#!/usr/bin/env python
"""Longest consecutive dry spell per summer season for CMIP6 historical daily precipitation.

For every file in INPUT_DIR whose model is listed in model_list.txt and whose experiment
is "historical", this computes, per year and grid point, the longest run of consecutive
days with precipitation < 1 mm/day within the season (May-Oct by default), and writes it
to OUTPUT_DIR with "pr_day" replaced by "dry_spell" in the filename. Only members that
also have every file in REQUIRED_MATCHES (pr ssp370, tas historical, tas ssp370) and pass
MEMBER_FILTERS are used. A copy clipped to each region in
CLIP_REGIONS is written to OUTPUT_DIR/<region name>/ with the same filename.

Same method as cesm2_dry_consect_calc-fast.ipynb.

Run in the background, e.g.:
    nohup python cmip6_dry_spell_calc.py --workers 8 > dry_spell.log 2>&1 &
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
OUTPUT_DIR = '/home/jwille/xenon_data/NextGEMS/dry_spells/CMIP6'
MODEL_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_list.txt')
EXPERIMENT = 'ssp370'              # the pr experiment that is processed
# A member is only used if the same model + member also exists in ALL of these
# (directory setting, filename prefix, experiment). The member label must match exactly.
REQUIRED_MATCHES = [
    ('input_dir', 'pr_day', 'ssp370'),
    ('tas_dir', 'tas_ann', 'historical'),
    ('tas_dir', 'tas_ann', 'ssp370'),
]
# Optional per-model regex the member label must match
MEMBER_FILTERS = {
    'CanESM5': r'^r\d+i\d+p2f\d+$',    # only physics version p1
}

MONTHS = [5, 6, 7, 8, 9, 10]
LATITUDE = [-90, 90]         # domain of the main output files
# Extra clipped copies, written to OUTPUT_DIR/<name>/ with the same filename
CLIP_REGIONS = {'30-60N': [30, 60]}
DRY_THRESHOLD = 1            # mm/day

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


# ----------------------------------------------------------------------------- functions
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


def select_files(dirs, models, experiment, required=REQUIRED_MATCHES):
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


def longest_run_vectorized(mask, axis=0):
    """Longest run of True/1 values along `axis`, fully vectorized.

    c     = running count of ones
    reset = value of c at the most recent zero (carried forward)
    c - reset = length of the current run of ones at each step
    """
    m = np.moveaxis(np.asarray(mask, dtype=bool), axis, 0)
    c = np.cumsum(m, axis=0, dtype=np.int32)
    reset = np.maximum.accumulate(np.where(m, 0, c), axis=0)
    return (c - reset).max(axis=0)


def dry_spell_for_file(in_path):
    ds = xr.open_dataset(in_path, use_cftime=True)
    da = ds['pr']

    units = da.attrs.get('units', '').strip()
    if units not in UNIT_FACTORS:
        raise ValueError(f"Unknown precipitation units '{units}' in {in_path}")
    factor = UNIT_FACTORS[units]

    # Subset before loading. No calendar conversion: Feb 29 is outside the season,
    # and some models use a 360_day calendar that cannot be converted cleanly.
    lat = da.lat.values
    lat_idx = np.where((lat >= LATITUDE[0]) & (lat <= LATITUDE[1]))[0]
    da = da.isel(lat=lat_idx)
    da = da.isel(time=np.where(da.time.dt.month.isin(MONTHS).values)[0])
    if da.sizes['time'] == 0 or lat_idx.size == 0:
        raise ValueError(f"No data in months {MONTHS} / latitudes {LATITUDE} in {in_path}")

    years = da.time.dt.year.values

    # One season per calendar year
    uniq_years = np.unique(years)
    time_size = np.array([(years == y).sum() for y in uniq_years])
    keep = time_size > 1
    uniq_years, time_size = uniq_years[keep], time_size[keep]

    # Load one season at a time to keep memory low on global high-resolution grids
    result = np.stack([
        longest_run_vectorized((da.isel(time=np.where(years == y)[0]) * factor).values < DRY_THRESHOLD)  # mm/day
        for y in uniq_years
    ]).astype(np.float64)

    # Drop incomplete seasons (more than 2 days shorter than the longest one)
    time_mask = time_size >= time_size.max() - 2
    result_final = result[time_mask]

    dry_spell = xr.DataArray(
        result_final,
        dims=['year', 'lat', 'lon'],
        coords={
            'year': ('year', uniq_years[time_mask]),
            'lat': ('lat', da.lat.values),
            'lon': ('lon', da.lon.values),
        },
        name='dry_spell',
        attrs={
            # No 'units: days' attribute: xarray would decode the values as timedeltas on reading
            'long_name': 'Longest consecutive dry spell in season (number of days)',
            'dry_day_definition': f'pr < {DRY_THRESHOLD} mm/day',
            'months': ','.join(str(m) for m in MONTHS),
            'source_file': os.path.basename(in_path),
            'source_units': units,
        },
    )
    ds.close()
    return dry_spell


def clip_lat(da, lat_range):
    lat = da.lat.values
    return da.isel(lat=np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0])


def write_atomic(da, out_path):
    tmp_path = out_path + '.tmp'
    da.to_netcdf(tmp_path)
    os.replace(tmp_path, out_path)   # only complete files get the final name


def process_one(fname, input_dir, output_dir, overwrite):
    in_path = os.path.join(input_dir, fname)
    out_name = fname.replace('pr_day', 'dry_spell', 1)
    out_paths = {None: os.path.join(output_dir, out_name)}
    for name in CLIP_REGIONS:
        out_paths[name] = os.path.join(output_dir, name, out_name)
    if all(os.path.exists(p) for p in out_paths.values()) and not overwrite:
        return fname, 'skipped (output exists)', 0.0

    t0 = time.time()
    dry_spell = dry_spell_for_file(in_path)
    write_atomic(dry_spell, out_paths[None])
    for name, lat_range in CLIP_REGIONS.items():
        os.makedirs(os.path.dirname(out_paths[name]), exist_ok=True)
        write_atomic(clip_lat(dry_spell, lat_range), out_paths[name])
    info = (f"{dry_spell.sizes['year']} years ({int(dry_spell.year[0])}-{int(dry_spell.year[-1])}), "
            f"mean {float(dry_spell.mean()):.2f} days")
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
    args = parser.parse_args()

    models = read_model_list(args.model_list)
    dirs = {'input_dir': args.input_dir, 'tas_dir': args.tas_dir}
    files, skipped = select_files(dirs, models, EXPERIMENT)

    print(f"Models ({len(models)}): {', '.join(models)}", flush=True)
    print(f"Processing pr_day {EXPERIMENT}; members must also have: "
          + ', '.join(f'{p} {e}' for _, p, e in REQUIRED_MATCHES), flush=True)
    for model, expected in models.items():
        n = sum(FILENAME_PATTERN.match(f)['model'] == model for f in files)
        note = '' if expected is None or n == expected else f'   <-- model_list.txt says {expected}'
        print(f"  {model}: {n} members{note}", flush=True)
    print(f"Total: {len(files)} files", flush=True)
    if skipped:
        print(f"Skipped {len(skipped)} {EXPERIMENT} pr files of listed models:", flush=True)
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
        futures = {pool.submit(process_one, f, args.input_dir, args.output_dir, args.overwrite): f for f in files}
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
