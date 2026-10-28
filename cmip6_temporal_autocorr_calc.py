#!/usr/bin/env python
"""Lag-1 temporal autocorrelation of daily precipitation per season, 30-60N, CMIP6 daily precipitation.

For every pr file in INPUT_DIR whose model is listed in model_list.txt and whose experiment is
--experiment (default "historical"), this computes for each year in YEARS and each grid point in
LATITUDE the Pearson correlation between precipitation on day t and on day t-1, using the days in
MONTHS (May-Oct) of that year only (no pair crosses from one season into the next). Same quantity as
temporal_auto() in autocorrelation_temporal_new_calc.ipynb:

    xr.corr(data, data.shift(time=1), dim='time')     (raw precipitation, not binary)

Grid points with no variability in a season (e.g. dry on every day) get NaN, as with xr.corr.
All grid points are kept (land and ocean); the land mask is applied later in the plotting.

Output: one file per member, OUTPUT_DIR/temporal_corr_<model>_<experiment>_<member>_30-60N.nc,
variable 'temporal_corr' (year, lat, lon). Only members that also have pr and tas files for both
historical and ssp370 (ALL_MATCHES) and pass MEMBER_FILTERS are used.

Run in the background, e.g.:
    nohup python cmip6_temporal_autocorr_calc.py --workers 8 > temporal_corr.log 2>&1 &
    nohup python cmip6_temporal_autocorr_calc.py --experiment ssp370 --years 2020 2049 --workers 8 > temporal_corr_ssp370.log 2>&1 &
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
def lag1_corr(x):
    """Pearson correlation of x[t] with x[t-1] along axis 0 (days), for every grid point; pairs with a
    NaN are left out, as in xr.corr(data, data.shift(time=1), dim='time'). No variability -> NaN."""
    a, b = x[1:].astype(np.float64), x[:-1].astype(np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(valid, a, np.nan), np.where(valid, b, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        da_ = a - np.nanmean(a, axis=0)
        db_ = b - np.nanmean(b, axis=0)
        return np.nansum(da_ * db_, axis=0) / np.sqrt(np.nansum(da_ ** 2, axis=0) * np.nansum(db_ ** 2, axis=0))


def temporal_corr_for_file(in_path, years=YEARS):
    ds = xr.open_dataset(in_path, use_cftime=True)
    da = ds['pr']

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

    # Read BLOCK_YEARS years at a time; one correlation map per year (season)
    day_year = da.time.dt.year.values
    uniq = np.unique(day_year)
    out_years, out_maps = [], []
    for i in range(0, uniq.size, BLOCK_YEARS):
        block_years = uniq[i:i + BLOCK_YEARS]
        idx = np.where(np.isin(day_year, block_years))[0]
        block = da.isel(time=idx).values
        block_day_year = day_year[idx]
        for y in block_years:
            season = block[block_day_year == y]
            if season.shape[0] < 3:
                continue
            out_years.append(int(y))
            out_maps.append(lag1_corr(season))

    result = xr.DataArray(
        np.stack(out_maps).astype(np.float32),
        dims=['year', 'lat', 'lon'],
        coords={'year': ('year', np.array(out_years)), 'lat': da.lat.values, 'lon': da.lon.values},
        name='temporal_corr',
        attrs={
            'long_name': 'Lag-1 temporal autocorrelation of daily precipitation within the season',
            'definition': 'Pearson correlation of pr(t) and pr(t-1) over the days in months of each year; '
                          'same as xr.corr(pr, pr.shift(time=1), dim="time")',
            'latitude': f'{LATITUDE[0]} to {LATITUDE[1]}',
            'months': ','.join(str(m) for m in MONTHS),
            'source_file': os.path.basename(in_path),
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
    return f"temporal_corr_{m['model']}_{m['experiment']}_{m['member']}_{REGION_NAME}.nc"


def process_one(fname, input_dir, output_dir, overwrite, years=YEARS):
    out_path = os.path.join(output_dir, output_name(fname))
    if os.path.exists(out_path) and not overwrite:
        return fname, 'skipped (output exists)', 0.0
    t0 = time.time()
    result = temporal_corr_for_file(os.path.join(input_dir, fname), years)
    write_atomic(result, out_path)
    yrs = result.year.values
    info = (f"{yrs.size} years ({yrs[0]}-{yrs[-1]}), grid {result.sizes['lat']}x{result.sizes['lon']}, "
            f"median r {float(result.median()):.2f}, NaN {float(result.isnull().mean()) * 100:.1f}%")
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
    print(f"Processing pr_day {experiment}, years {years}, months {MONTHS}; "
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
