#!/usr/bin/env python
"""Daily share of land area with precipitation above a threshold, 30-60N, CMIP6 daily precipitation.

For every pr file in INPUT_DIR whose model is listed in model_list.txt and whose experiment is
--experiment (default "historical"), this computes for each day in MONTHS (May-Oct) and YEARS the
percentage of the land area in LATITUDE with pr > THRESHOLDS (mm/day):

    pct = 100 * sum(cos(lat) over land cells with pr > threshold) / sum(cos(lat) over land cells)

Land is sftlf > 0 (SFTLF_DIR), matched to the pr grid by nearest neighbour; models without an
sftlf file use the mask of FALLBACK_MASK_MODEL. Same method as cesm2_ens_histo_calc.ipynb plus the
division by the land area done in land_histo_NH_plot_new.ipynb.

Output: one small file per member, OUTPUT_DIR/precip_cover_<model>_<experiment>_<member>_30-60N.nc,
variable 'wet_land_pct' (time, threshold). Only members that also have pr and tas files for both
historical and ssp370 (ALL_MATCHES) and pass MEMBER_FILTERS are used.

Run in the background, e.g.:
    nohup python cmip6_precip_cover_calc.py --workers 8 > precip_cover.log 2>&1 &
    nohup python cmip6_precip_cover_calc.py --experiment ssp370 --years 2015 2049 --workers 8 > precip_cover_ssp370.log 2>&1 &
"""
import argparse
import os
import re
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

warnings.filterwarnings(action='ignore')

# ----------------------------------------------------------------------------- settings
INPUT_DIR = '/net/atmos/data/cmip6-ng/pr/day/native'
TAS_DIR = '/net/atmos/data/cmip6-ng/tas/ann/native'
SFTLF_DIR = '/net/atmos/data/cmip6-ng/sftlf/fx/native'
OUTPUT_DIR = '/home/jwille/xenon_data/NextGEMS/mid_lat_precip/CMIP6'
MODEL_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_list.txt')
EXPERIMENT = 'historical'              # default pr experiment to process (--experiment to change)
# A member is only used if the same model + member exists in ALL of these
# (directory setting, filename prefix, experiment); the processed pr experiment itself is
# skipped in this check. The member label must match exactly.
ALL_MATCHES = [
    ('input_dir', 'pr_day', 'historical'),
    ('input_dir', 'pr_day', 'ssp370'),
    ('tas_dir', 'tas_ann', 'historical'),
    ('tas_dir', 'tas_ann', 'ssp370'),
]


def required_matches(experiment):
    return [m for m in ALL_MATCHES if not (m[1] == 'pr_day' and m[2] == experiment)]


# Optional per-model regex the member label must match
MEMBER_FILTERS = {
    'CanESM5': r'^r\d+i\d+p2f\d+$',    # only physics version p2
}
FALLBACK_MASK_MODEL = 'MPI-ESM1-2-HR'  # land mask for models without an sftlf file (BCC-CSM2-MR)

MONTHS = [5, 6, 7, 8, 9, 10]
YEARS = (1850, 2014)                   # inclusive, default for --years; None = all years in the file
LATITUDE = [30, 60]
REGION_NAME = '30-60N'                 # used in the output filename
THRESHOLDS = [1.0]                     # mm/day; more thresholds cost almost nothing, e.g. [0.1, 0.2, 0.5, 1.0]
BLOCK_YEARS = 10                       # years read from disk at once (memory vs. speed)

# Conversion factors to mm/day, keyed by the units attribute of the pr variable
UNIT_FACTORS = {
    'kg m-2 s-1': 86400,     # 1 kg m-2 of water = 1 mm
    'kg/m2/s': 86400,
    'mm s-1': 86400,
    'm s-1': 86400 * 1000,
    'm/s': 86400 * 1000,
    'mm/day': 1,
    'mm day-1': 1,
}


def filename_pattern(prefix):
    # <prefix>_<model>_<experiment>_<member>_<grid>.nc
    return re.compile(rf'^{prefix}_(?P<model>[^_]+)_(?P<experiment>[^_]+)_(?P<member>[^_]+)_(?P<grid>[^_]+)\.nc$')


FILENAME_PATTERN = filename_pattern('pr_day')


# ----------------------------------------------------------------------------- file selection
def read_model_list(path):
    """{model: expected member count} from lines like 'ACCESS-ESM1-5 (40)'; duplicates
    removed, order kept. The count is None if a line has no number."""
    models = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'^\s*([^\s(]+)\s*(?:\((\d+)\))?', line)
            if m and m.group(1) not in models:
                models[m.group(1)] = int(m.group(2)) if m.group(2) else None
    return models


def available_members(directory, prefix, experiment):
    """(model, member) pairs that have a <prefix> file for this experiment in directory."""
    pattern = filename_pattern(prefix)
    pairs = set()
    for fname in os.listdir(directory):
        m = pattern.match(fname)
        if m and m['experiment'] == experiment:
            pairs.add((m['model'], m['member']))
    return pairs


def select_files(dirs, models, experiment, required):
    """pr files in dirs['input_dir'] for the listed models and this experiment, keeping only
    members that pass MEMBER_FILTERS and have every file in `required`.
    Returns (selected, {skipped fname: reason})."""
    required_pairs = [(f'{prefix} {exp}', available_members(dirs[key], prefix, exp))
                      for key, prefix, exp in required]
    selected, skipped = [], {}
    for fname in sorted(os.listdir(dirs['input_dir'])):
        m = FILENAME_PATTERN.match(fname)
        if not (m and m['model'] in models and m['experiment'] == experiment):
            continue
        member_filter = MEMBER_FILTERS.get(m['model'])
        if member_filter and not re.match(member_filter, m['member']):
            skipped[fname] = 'excluded by MEMBER_FILTERS'
            continue
        missing = [label for label, pairs in required_pairs if (m['model'], m['member']) not in pairs]
        if missing:
            skipped[fname] = 'no ' + ', no '.join(missing)
        else:
            selected.append(fname)
    return selected, skipped


# ----------------------------------------------------------------------------- land mask
def find_sftlf(sftlf_dir, model):
    """sftlf file of a model: historical preferred, else any experiment; None if there is none."""
    pat = re.compile(rf'^sftlf_fx_{re.escape(model)}_(?P<exp>[^_]+)_(?P<mem>[^_]+)_[^_]+\.nc$')
    cands = [(m['exp'] != 'historical', [int(n) for n in re.findall(r'\d+', m['mem'])], f)
             for f in os.listdir(sftlf_dir) if (m := pat.match(f))]
    return os.path.join(sftlf_dir, sorted(cands)[0][2]) if cands else None


def nearest_regrid(src, lat, lon):
    """Nearest-neighbour match of a lat/lon field to the given lat/lon (lon periodic).
    Identical grids give exactly the original values."""
    ilat = np.abs(src.lat.values[:, None] - np.asarray(lat)[None, :]).argmin(axis=0)
    dlon = (src.lon.values[:, None] - np.asarray(lon)[None, :] + 180) % 360 - 180
    ilon = np.abs(dlon).argmin(axis=0)
    return src.values[np.ix_(ilat, ilon)]


def land_mask(sftlf_dir, model, lat, lon):
    path = find_sftlf(sftlf_dir, model)
    name = os.path.basename(path) if path else f'{FALLBACK_MASK_MODEL} (fallback)'
    path = path or find_sftlf(sftlf_dir, FALLBACK_MASK_MODEL)
    with xr.open_dataset(path) as ds:
        sftlf = ds['sftlf'].load()
    return nearest_regrid(sftlf, lat, lon) > 0, name


# ----------------------------------------------------------------------------- calculation
def precip_cover_for_file(in_path, sftlf_dir, years=YEARS):
    model = FILENAME_PATTERN.match(os.path.basename(in_path))['model']
    ds = xr.open_dataset(in_path, use_cftime=True)
    da = ds['pr']

    units = da.attrs.get('units', '').strip()
    if units not in UNIT_FACTORS:
        raise ValueError(f"Unknown precipitation units '{units}' in {in_path}")
    factor = UNIT_FACTORS[units]

    # Subset before loading. No calendar conversion: Feb 29 is outside the season,
    # and some models use a 360_day calendar that cannot be converted cleanly.
    lat = da.lat.values
    da = da.isel(lat=np.where((lat >= LATITUDE[0]) & (lat <= LATITUDE[1]))[0]).transpose('time', 'lat', 'lon')
    keep = da.time.dt.month.isin(MONTHS).values
    if years is not None:
        keep &= (da.time.dt.year.values >= years[0]) & (da.time.dt.year.values <= years[1])
    da = da.isel(time=np.where(keep)[0])
    if da.sizes['time'] == 0 or da.sizes['lat'] == 0:
        raise ValueError(f"No data in months {MONTHS} / years {years} / latitudes {LATITUDE} in {in_path}")

    land, mask_name = land_mask(sftlf_dir, model, da.lat.values, da.lon.values)
    w = (np.cos(np.deg2rad(da.lat.values))[:, None] * np.ones(da.sizes['lon']))[land]   # land cells only
    w_total = w.sum()
    thresholds = np.asarray(THRESHOLDS, dtype=float)

    # Read BLOCK_YEARS years at a time; per day: cos(lat)-weighted land area with pr > threshold
    day_year = da.time.dt.year.values
    uniq = np.unique(day_year)
    out = []
    for i in range(0, uniq.size, BLOCK_YEARS):
        idx = np.where((day_year >= uniq[i]) & (day_year <= uniq[min(i + BLOCK_YEARS, uniq.size) - 1]))[0]
        p = (da.isel(time=idx).values * factor)[:, land]                              # (day, land cell), mm/day
        wet = p[:, :, None] > thresholds                                              # NaN counts as not wet
        out.append(100.0 * np.einsum('dct,c->dt', wet, w) / w_total)
    pct = np.concatenate(out)

    result = xr.DataArray(
        pct,
        dims=['time', 'threshold'],
        coords={'time': da.time.values, 'threshold': ('threshold', thresholds, {'units': 'mm/day'})},
        name='wet_land_pct',
        attrs={
            'long_name': 'Percentage of land area with daily precipitation above threshold',
            'units': '%',
            'definition': 'pr > threshold; cos(lat)-weighted share of land cells (sftlf > 0)',
            'latitude': f'{LATITUDE[0]} to {LATITUDE[1]}',
            'months': ','.join(str(m) for m in MONTHS),
            'land_mask': mask_name,
            'source_file': os.path.basename(in_path),
            'source_units': units,
        },
    )
    ds.close()
    return result


def write_atomic(da, out_path):
    tmp_path = out_path + '.tmp'
    da.to_netcdf(tmp_path)
    os.replace(tmp_path, out_path)   # only complete files get the final name


def output_name(fname):
    m = FILENAME_PATTERN.match(fname)
    return f"precip_cover_{m['model']}_{m['experiment']}_{m['member']}_{REGION_NAME}.nc"


def process_one(fname, input_dir, output_dir, sftlf_dir, overwrite, years=YEARS):
    out_path = os.path.join(output_dir, output_name(fname))
    if os.path.exists(out_path) and not overwrite:
        return fname, 'skipped (output exists)', 0.0
    t0 = time.time()
    result = precip_cover_for_file(os.path.join(input_dir, fname), sftlf_dir, years)
    write_atomic(result, out_path)
    yrs = result.time.dt.year.values
    info = (f"{result.sizes['time']} days ({yrs[0]}-{yrs[-1]}), median "
            + ', '.join(f'{float(result.sel(threshold=t).median()):.1f}% (>{t:g} mm)' for t in result.threshold.values)
            + f", mask: {result.attrs['land_mask']}")
    return fname, info, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', default=INPUT_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--model-list', default=MODEL_LIST)
    parser.add_argument('--tas-dir', default=TAS_DIR)
    parser.add_argument('--sftlf-dir', default=SFTLF_DIR)
    parser.add_argument('--workers', type=int, default=8, help='files processed in parallel')
    parser.add_argument('--overwrite', action='store_true', help='recompute files whose output already exists')
    parser.add_argument('--dry-run', action='store_true', help='only list the files that would be processed')
    parser.add_argument('--experiment', default=EXPERIMENT, help="pr experiment to process, e.g. 'ssp370'")
    parser.add_argument('--years', type=int, nargs=2, default=YEARS, metavar=('FIRST', 'LAST'),
                        help=f'first and last year to keep (inclusive), default {YEARS}')
    args = parser.parse_args()
    years = tuple(args.years) if args.years is not None else None
    experiment = args.experiment
    required = required_matches(experiment)

    models = read_model_list(args.model_list)
    dirs = {'input_dir': args.input_dir, 'tas_dir': args.tas_dir}
    files, skipped = select_files(dirs, models, experiment, required)

    print(f"Models ({len(models)}): {', '.join(models)}", flush=True)
    print(f"Processing pr_day {experiment}, years {years}, months {MONTHS}, thresholds {THRESHOLDS} mm/day; "
          f"members must also have: " + ', '.join(f'{p} {e}' for _, p, e in required), flush=True)
    for model, expected in models.items():
        n = sum(FILENAME_PATTERN.match(f)['model'] == model for f in files)
        note = '' if expected is None or n == expected else f'   <-- model_list.txt says {expected}'
        print(f"  {model}: {n} members{note}", flush=True)
    print(f"Total: {len(files)} files", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} {experiment} pr files of listed models:", flush=True)
        for f, reason in skipped.items():
            print(f"   {f}  ({reason})", flush=True)
    if args.dry_run:
        print("Files that would be processed:")
        for f in files:
            print('  ', f)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    t_total = time.time()
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, f, args.input_dir, args.output_dir, args.sftlf_dir, args.overwrite, years): f
                   for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                _, info, dt = fut.result()
                print(f"[{i}/{len(files)}] {fname}: {info}, {dt:.1f}s", flush=True)
            except Exception:
                failed.append(fname)
                print(f"[{i}/{len(files)}] {fname}: FAILED\n{traceback.format_exc()}", flush=True)

    print(f"Done in {time.time() - t_total:.1f}s. {len(files) - len(failed)} succeeded, {len(failed)} failed.", flush=True)
    for f in failed:
        print('  failed:', f, flush=True)


if __name__ == '__main__':
    main()
