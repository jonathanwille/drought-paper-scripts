#!/usr/bin/env python
"""Hourly share of land area with precipitation above thresholds, 30-60N, CESM2-LE ensemble (historical or ssp370).

For every member in INPUT_DIR (b.e21.<compset>.f09_g17.LE2-<init>.<member>.cam.h4.PRECT.<start>-<end>.nc, compset
BHISTsmbb for --experiment historical, BSSP370smbb for ssp370),
this computes for each hour in MONTHS (May-Oct) and YEARS the percentage of the land area and of the sea
area in LATITUDE with hourly precipitation > THRESHOLDS (mm per hour):

    pct = 100 * sum(cos(lat) over land (sea) cells with pr > threshold) / sum(cos(lat) over land (sea) cells)

Land is MASK > 0, sea is MASK == 0 in LAND_MASK (same CESM2 grid). Same quantity as cesm2_ens_histo_calc.ipynb (daily, > 1 mm)
divided by the land area as in land_histo_NH_plot_new.ipynb, here for hourly data.

Every input file (one member-decade) is processed separately and written to OUTPUT_DIR/parts/; when all
files of a member are done they are merged into one file per member:
    OUTPUT_DIR/cesm2_hourly_precip_cover_<init>.<member>_30-60N.nc          (historical)
    OUTPUT_DIR/cesm2_hourly_precip_cover_<init>.<member>_ssp370_30-60N.nc   (ssp370)
with variables 'wet_land_pct' and 'wet_sea_pct' (time, threshold). Only input files that overlap the years are read.
Finished parts are skipped on a rerun.

Run in the background, e.g.:
    nohup python cesm2_hourly_precip_cover_calc.py --workers 8 > cesm2_hourly_precip_cover.log 2>&1 &
    nohup python cesm2_hourly_precip_cover_calc.py --experiment ssp370 --years 2021 2049 --workers 8 > cesm2_hourly_precip_cover_ssp370.log 2>&1 &
"""
import argparse
import os
import re
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

warnings.filterwarnings(action='ignore')

# ----------------------------------------------------------------------------- settings
INPUT_DIR = '/home/jwille/xenon_data/NextGEMS/CESM2/precip/1H/database'
OUTPUT_DIR = '/home/jwille/xenon_data/NextGEMS/mid_lat_precip/cesm2_ens_hourly'
LAND_MASK = '/home/jwille/xenon_data/NextGEMS/CESM2/precip/24H/mask.nc'   # variable MASK, land = MASK > 0
COMPSETS = {'historical': 'BHISTsmbb', 'ssp370': 'BSSP370smbb'}   # experiment -> compset in the filenames
EXPERIMENT = 'historical'              # default for --experiment

MONTHS = [5, 6, 7, 8, 9, 10]
DEFAULT_YEARS = {'historical': (1850, 2014), 'ssp370': (2021, 2049)}   # inclusive, default for --years
LATITUDE = [30, 60]
REGION_NAME = '30-60N'                 # used in the output filename
THRESHOLDS = [0.1, 0.5]                # mm per hour
UNIT_FACTOR = 3600 * 1000              # PRECT m/s -> mm per hour
BLOCK_YEARS = 1                        # years read from disk at once (memory vs. speed)

FILENAME_PATTERN = re.compile(
    r'^b\.e21\.(?P<exp>[^.]+)\.f09_g17\.LE2-(?P<init>\d+)\.(?P<member>\d+)\.cam\.h4\.PRECT\.'
    r'(?P<start>\d+)-(?P<end>\d+)\.nc$')


# ----------------------------------------------------------------------------- helpers
def member_name(m):
    return f"{m['init']}.{m['member']}"


def exp_tag(experiment):
    return '' if experiment == 'historical' else f'_{experiment}'      # historical file names unchanged


def part_name(m, experiment='historical'):
    return f"cesm2_hourly_precip_cover_{member_name(m)}{exp_tag(experiment)}_{m['start']}-{m['end']}_{REGION_NAME}.nc"


def member_out_name(member, experiment='historical'):
    return f'cesm2_hourly_precip_cover_{member}{exp_tag(experiment)}_{REGION_NAME}.nc'


def overlaps(m, years):
    """Does the file's date range (from its name) overlap the years?"""
    return int(m['start'][:4]) <= years[1] and int(m['end'][:4]) >= years[0]


def write_atomic(da, out_path):
    tmp_path = out_path + '.tmp'
    da.to_netcdf(tmp_path)
    os.replace(tmp_path, out_path)   # only complete files get the final name


def load_mask(lat, lon):
    """MASK on the data grid (land = MASK > 0, sea = MASK == 0): the mask points at the data latitudes/longitudes."""
    with xr.open_dataset(LAND_MASK) as ds:
        mask = ds['MASK'].load()
    if 'time' in mask.dims:
        mask = mask.isel(time=0)
    ilat = np.abs(mask.lat.values[:, None] - lat[None, :]).argmin(axis=0)
    ilon = np.abs(((mask.lon.values[:, None] - lon[None, :]) + 180) % 360 - 180).argmin(axis=0)
    if max(np.abs(mask.lat.values[ilat] - lat).max(), np.abs(mask.lon.values[ilon] - lon).max()) > 0.01:
        raise ValueError('land mask grid does not match the PRECT grid')
    return mask.values[np.ix_(ilat, ilon)]


def mid_interval_times(ds):
    """Time stamps at the middle of each averaging interval if time bounds exist (CESM stamps the end of
    the interval, so 00:00 of 1 May would otherwise count as May); else the time coordinate as is."""
    bname = ds['time'].attrs.get('bounds', 'time_bnds')
    if bname in ds:
        b = ds[bname].values
        return np.array([b0 + (b1 - b0) / 2 for b0, b1 in b])
    return ds['time'].values


# ----------------------------------------------------------------------------- calculation
def precip_cover_for_file(in_path, years):
    ds = xr.open_dataset(in_path, use_cftime=True)
    da = ds['PRECT']
    lat = da.lat.values
    da = da.isel(lat=np.where((lat >= LATITUDE[0]) & (lat <= LATITUDE[1]))[0]).transpose('time', 'lat', 'lon')

    times = mid_interval_times(ds)
    months = np.array([t.month for t in times])
    yr = np.array([t.year for t in times])
    keep = np.isin(months, MONTHS) & (yr >= years[0]) & (yr <= years[1])
    idx_all = np.where(keep)[0]
    if idx_all.size == 0:
        ds.close()
        return None                                   # e.g. a file with no May-Oct hours in YEARS

    mask = load_mask(da.lat.values, da.lon.values)
    surfaces = {'land': mask > 0, 'sea': mask == 0}
    w2d = np.cos(np.deg2rad(da.lat.values))[:, None] * np.ones(da.sizes['lon'])
    thresholds = np.asarray(THRESHOLDS, dtype=float)

    out = {name: [] for name in surfaces}
    sel_years = yr[idx_all]
    for y0 in range(sel_years.min(), sel_years.max() + 1, BLOCK_YEARS):
        idx = idx_all[(sel_years >= y0) & (sel_years < y0 + BLOCK_YEARS)]
        if idx.size == 0:
            continue
        p = da.isel(time=idx).values * UNIT_FACTOR                         # (hour, lat, lon), mm per hour
        for name, cells in surfaces.items():
            w = w2d[cells]
            wet = p[:, cells][:, :, None] > thresholds                     # NaN counts as not wet
            out[name].append(100.0 * np.einsum('hct,c->ht', wet, w) / w.sum())

    coords = {'time': times[idx_all], 'threshold': ('threshold', thresholds, {'units': 'mm/h'})}
    time_stamp = ('middle of the averaging interval' if 'time_bnds' in ds or ds['time'].attrs.get('bounds') in ds
                  else 'as in the input file')
    result = xr.Dataset(
        {f'wet_{name}_pct': (('time', 'threshold'), np.concatenate(out[name]), {
            'long_name': f'Percentage of {name} area with hourly precipitation above threshold',
            'units': '%',
            'definition': f'PRECT * 3600 * 1000 > threshold (mm/h); cos(lat)-weighted share of {name} cells '
                          f'({"MASK > 0" if name == "land" else "MASK == 0"})'}) for name in surfaces},
        coords=coords,
        attrs={'latitude': f'{LATITUDE[0]} to {LATITUDE[1]}', 'months': ','.join(str(m) for m in MONTHS),
               'time_stamp': time_stamp},
    )
    ds.close()
    return result


def process_part(fname, input_dir, parts_dir, overwrite, years, experiment):
    out_path = os.path.join(parts_dir, part_name(FILENAME_PATTERN.match(fname), experiment))
    if os.path.exists(out_path) and not overwrite:
        return fname, 'skipped (part exists)', 0.0
    t0 = time.time()
    result = precip_cover_for_file(os.path.join(input_dir, fname), years)
    if result is None:
        # empty marker so the file counts as done
        empty = (('time', 'threshold'), np.empty((0, len(THRESHOLDS))))
        result = xr.Dataset({'wet_land_pct': empty, 'wet_sea_pct': empty}, coords={'threshold': THRESHOLDS})
        write_atomic(result, out_path)
        return fname, 'no hours in MONTHS/YEARS', time.time() - t0
    write_atomic(result, out_path)
    yrs = result.time.dt.year.values
    info = (f"{result.sizes['time']} hours ({yrs[0]}-{yrs[-1]}), median land / sea "
            + ', '.join(f"{float(result['wet_land_pct'].sel(threshold=t).median()):.1f} / "
                        f"{float(result['wet_sea_pct'].sel(threshold=t).median()):.1f}% (>{t:g} mm/h)"
                        for t in result.threshold.values))
    return fname, info, time.time() - t0


def merge_member(member, part_files, parts_dir, output_dir, experiment):
    das = []
    for f in sorted(part_files):
        with xr.open_dataset(os.path.join(parts_dir, f), use_cftime=True) as ds:
            if 'wet_sea_pct' not in ds:
                raise ValueError(f'{f} has no wet_sea_pct (made by an older version): rerun with --overwrite')
            da = ds[['wet_land_pct', 'wet_sea_pct']].load()
        if da.sizes['time']:
            das.append(da)
    da = xr.concat(das, dim='time').sortby('time')
    _, first = np.unique(da.time.values, return_index=True)          # drop hours that occur in two files
    da = da.isel(time=np.sort(first))
    da.attrs['member'] = member
    out_path = os.path.join(output_dir, member_out_name(member, experiment))
    write_atomic(da, out_path)
    yrs = da.time.dt.year.values
    return out_path, da.sizes['time'], yrs[0], yrs[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', default=INPUT_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--workers', type=int, default=8, help='files processed in parallel')
    parser.add_argument('--overwrite', action='store_true', help='recompute parts whose output already exists')
    parser.add_argument('--dry-run', action='store_true', help='only list the files that would be processed')
    parser.add_argument('--experiment', default=EXPERIMENT, choices=sorted(COMPSETS))
    parser.add_argument('--years', type=int, nargs=2, metavar=('FIRST', 'LAST'),
                        help=f'first and last year (inclusive); default {DEFAULT_YEARS}')
    args = parser.parse_args()
    experiment = args.experiment
    years = tuple(args.years) if args.years else DEFAULT_YEARS[experiment]

    members = defaultdict(list)                                           # member -> input files
    for fname in sorted(os.listdir(args.input_dir)):
        m = FILENAME_PATTERN.match(fname)
        if m and m['exp'] == COMPSETS[experiment] and overlaps(m, years):
            members[member_name(m)].append(fname)
    files = [f for fl in members.values() for f in fl]

    print(f"{experiment} ({COMPSETS[experiment]}): {len(members)} members, {len(files)} files; years {years}, months {MONTHS}, "
          f"thresholds {THRESHOLDS} mm/h", flush=True)
    for member, fl in members.items():
        print(f"  {member}: {len(fl)} files", flush=True)
    if args.dry_run:
        for f in files:
            print('  ', f)
        return

    parts_dir = os.path.join(args.output_dir, 'parts')
    os.makedirs(parts_dir, exist_ok=True)
    t_total = time.time()
    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_part, f, args.input_dir, parts_dir, args.overwrite, years, experiment): f
                   for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                _, info, dt = fut.result()
                print(f"[{i}/{len(files)}] {fname}: {info}, {dt:.1f}s", flush=True)
            except Exception:
                failed.append(fname)
                print(f"[{i}/{len(files)}] {fname}: FAILED\n{traceback.format_exc()}", flush=True)

    # merge the parts of every member whose files all succeeded
    for member, fl in members.items():
        if any(f in failed for f in fl):
            print(f"{member}: not merged, {sum(f in failed for f in fl)} file(s) failed", flush=True)
            continue
        parts = [part_name(FILENAME_PATTERN.match(f), experiment) for f in fl]
        try:
            out_path, n, y0, y1 = merge_member(member, parts, parts_dir, args.output_dir, experiment)
            print(f"{member}: {n} hours ({y0}-{y1}) -> {out_path}", flush=True)
        except Exception:
            failed.append(member)
            print(f"{member}: merge FAILED\n{traceback.format_exc()}", flush=True)

    print(f"Done in {time.time() - t_total:.1f}s. {len(failed)} failure(s).", flush=True)
    for f in failed:
        print('  failed:', f, flush=True)


if __name__ == '__main__':
    main()
