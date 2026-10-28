#!/usr/bin/env python
"""Markov dry-to-wet transition probability of daily precipitation per season, 30-60N, CMIP6.

For every pr file in INPUT_DIR whose model is listed in model_list.txt and whose experiment is
--experiment (default "historical"), this computes for each year in YEARS and each grid point in
LATITUDE, using the days in MONTHS (May-Oct) of that year only (no pair crosses from one season
into the next):

    wet = pr > WET_THRESHOLD (1 mm/day), dry otherwise
    n_D  = number of dry days t that are followed by a day t+1
    P_DW = (number of dry days t followed by a wet day t+1) / n_D      (NaN where n_D = 0)

Same P_DW and n_D as markov_transitions_wetdry() in markov_calc.ipynb, with the wet/dry series of
its CESM2 cell (pr > 1 mm/day). Missing days are left out of the counts (pairs with a NaN are skipped).
All grid points are kept (land and ocean); the land mask is applied later in the plotting.

Output: one file per member, OUTPUT_DIR/markov_<model>_<experiment>_<member>_30-60N.nc,
variables 'P_DW' and 'n_D' (year, lat, lon). Only members that also have pr and tas files for both
historical and ssp370 (ALL_MATCHES) and pass MEMBER_FILTERS are used.

Run in the background, e.g.:
    nohup python cmip6_markov_calc.py --workers 8 > markov.log 2>&1 &
    nohup python cmip6_markov_calc.py --experiment ssp370 --years 2020 2049 --workers 8 > markov_ssp370.log 2>&1 &
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
OUTPUT_DIR = '/home/jwille/xenon_data/NextGEMS/mid_lat_correlation/CMIP6'
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

MONTHS = [5, 6, 7, 8, 9, 10]
YEARS = (1850, 2014)                   # inclusive, default for --years; None = all years in the file
LATITUDE = [30, 60]
REGION_NAME = '30-60N'                 # used in the output filename
BLOCK_YEARS = 10                       # years read from disk at once (memory vs. speed)
WET_THRESHOLD = 1.0                    # mm/day: wet = pr > WET_THRESHOLD, dry otherwise

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


# ----------------------------------------------------------------------------- calculation
def dry_to_wet(x):
    """x (days, ...) in mm/day. Returns (P_DW, n_D) per grid point: n_D = dry days t with a valid day t+1,
    P_DW = share of them followed by a wet day (NaN where n_D = 0). Pairs with a NaN are left out."""
    x0, x1 = x[:-1], x[1:]
    valid = np.isfinite(x0) & np.isfinite(x1)
    dry0 = valid & ~(x0 > WET_THRESHOLD)
    wet1 = valid & (x1 > WET_THRESHOLD)
    n_d = dry0.sum(axis=0)
    n_dw = (dry0 & wet1).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        p_dw = np.where(n_d > 0, n_dw / n_d, np.nan)
    return p_dw, n_d


def markov_for_file(in_path, years=YEARS):
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

    # Read BLOCK_YEARS years at a time; one map per year (season)
    day_year = da.time.dt.year.values
    uniq = np.unique(day_year)
    out_years, out_p, out_n = [], [], []
    for i in range(0, uniq.size, BLOCK_YEARS):
        block_years = uniq[i:i + BLOCK_YEARS]
        idx = np.where(np.isin(day_year, block_years))[0]
        block = da.isel(time=idx).values * factor                      # mm/day
        block_day_year = day_year[idx]
        for y in block_years:
            season = block[block_day_year == y]
            if season.shape[0] < 3:
                continue
            p_dw, n_d = dry_to_wet(season)
            out_years.append(int(y))
            out_p.append(p_dw)
            out_n.append(n_d)

    dims = ['year', 'lat', 'lon']
    coords = {'year': ('year', np.array(out_years)), 'lat': da.lat.values, 'lon': da.lon.values}
    result = xr.Dataset(
        {
            'P_DW': (dims, np.stack(out_p).astype(np.float32), {
                'long_name': 'Probability that a dry day is followed by a wet day (within the season)',
                'definition': f'wet = pr > {WET_THRESHOLD} mm/day; P_DW = n(dry -> wet) / n_D'}),
            'n_D': (dims, np.stack(out_n).astype(np.int16), {
                'long_name': 'Number of dry days followed by a day in the same season (denominator of P_DW)'}),
        },
        coords=coords,
        attrs={
            'wet_threshold_mm_per_day': WET_THRESHOLD,
            'latitude': f'{LATITUDE[0]} to {LATITUDE[1]}',
            'months': ','.join(str(m) for m in MONTHS),
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
    return f"markov_{m['model']}_{m['experiment']}_{m['member']}_{REGION_NAME}.nc"


def process_one(fname, input_dir, output_dir, overwrite, years=YEARS):
    out_path = os.path.join(output_dir, output_name(fname))
    if os.path.exists(out_path) and not overwrite:
        return fname, 'skipped (output exists)', 0.0
    t0 = time.time()
    result = markov_for_file(os.path.join(input_dir, fname), years)
    write_atomic(result, out_path)
    yrs = result.year.values
    info = (f"{yrs.size} years ({yrs[0]}-{yrs[-1]}), grid {result.sizes['lat']}x{result.sizes['lon']}, "
            f"median P_DW {float(result['P_DW'].median()):.2f}, NaN {float(result['P_DW'].isnull().mean()) * 100:.1f}%")
    return fname, info, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', default=INPUT_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--model-list', default=MODEL_LIST)
    parser.add_argument('--tas-dir', default=TAS_DIR)
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
    print(f"Processing pr_day {experiment}, years {years}, months {MONTHS}, wet = pr > {WET_THRESHOLD} mm/day; "
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
        futures = {pool.submit(process_one, f, args.input_dir, args.output_dir, args.overwrite, years): f
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
