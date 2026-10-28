#!/usr/bin/env python
"""Mean May-Oct precipitation total, 30-60N, CESM2-LE historical ensemble (hourly PRECT), 1850-2014.

For every member in INPUT_DIR (b.e21.BHISTsmbb.f09_g17.LE2-<init>.<member>.cam.h4.PRECT.<start>-<end>.nc),
this sums the hourly precipitation (PRECT * 3600 * 1000 = mm per hour) over the hours in MONTHS (May-Oct)
of each year in YEARS, at every grid point in LATITUDE. Same quantity as the CESM2 cell of
precip_total_calc.ipynb (daily mm summed over May-Oct of each year), here from hourly data.

Every input file (one member-decade) is processed separately and written to OUTPUT_DIR/parts/; when all
files of a member are done they are merged into one file per member:
    OUTPUT_DIR/cesm2_hourly_precip_total_<init>.<member>_30-60N.nc   variable 'pr_total' (year, lat, lon), mm
At the end, the script writes the mean over YEARS of every member and of the ensemble,
    OUTPUT_DIR/cesm2_hourly_precip_total_ensmean_1850-2014_30-60N.nc   (lat, lon), mm
and prints and saves (csv) the cos(lat)-area-weighted land and sea means (m) per member and for the
ensemble. Land is MASK > 0 in LAND_MASK. Finished parts are skipped on a rerun.

Run in the background, e.g.:
    nohup python cesm2_hourly_precip_total_calc.py --workers 8 > cesm2_hourly_precip_total.log 2>&1 &
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
OUTPUT_DIR = '/home/jwille/xenon_data/NextGEMS/mid_lat_precip/cesm2_ens_hourly_total'
LAND_MASK = '/home/jwille/xenon_data/NextGEMS/CESM2/precip/24H/mask.nc'   # variable MASK, land = MASK > 0
EXPERIMENT = 'BHISTsmbb'               # historical compset in the filenames

MONTHS = [5, 6, 7, 8, 9, 10]
YEARS = (1850, 2014)                   # inclusive
LATITUDE = [30, 60]
REGION_NAME = '30-60N'                 # used in the output filename
UNIT_FACTOR = 3600 * 1000              # PRECT m/s -> mm per hour
BLOCK_YEARS = 1                        # years read from disk at once (memory vs. speed)

FILENAME_PATTERN = re.compile(
    r'^b\.e21\.(?P<exp>[^.]+)\.f09_g17\.LE2-(?P<init>\d+)\.(?P<member>\d+)\.cam\.h4\.PRECT\.'
    r'(?P<start>\d+)-(?P<end>\d+)\.nc$')


# ----------------------------------------------------------------------------- helpers
def member_name(m):
    return f"{m['init']}.{m['member']}"


def part_name(m):
    return f"cesm2_hourly_precip_total_{member_name(m)}_{m['start']}-{m['end']}_{REGION_NAME}.nc"


def member_out_name(member):
    return f'cesm2_hourly_precip_total_{member}_{REGION_NAME}.nc'


def write_atomic(obj, out_path):
    tmp_path = out_path + '.tmp'
    obj.to_netcdf(tmp_path)
    os.replace(tmp_path, out_path)   # only complete files get the final name


def load_land(lat, lon):
    """Land mask (MASK > 0) on the data grid: the mask points at the data latitudes/longitudes."""
    with xr.open_dataset(LAND_MASK) as ds:
        mask = ds['MASK'].load()
    if 'time' in mask.dims:
        mask = mask.isel(time=0)
    ilat = np.abs(mask.lat.values[:, None] - lat[None, :]).argmin(axis=0)
    ilon = np.abs(((mask.lon.values[:, None] - lon[None, :]) + 180) % 360 - 180).argmin(axis=0)
    if max(np.abs(mask.lat.values[ilat] - lat).max(), np.abs(mask.lon.values[ilon] - lon).max()) > 0.01:
        raise ValueError('land mask grid does not match the PRECT grid')
    return mask.values[np.ix_(ilat, ilon)] > 0


def mid_interval_times(ds):
    """Time stamps at the middle of each averaging interval if time bounds exist (CESM stamps the end of
    the interval, so 00:00 of 1 May would otherwise count as May); else the time coordinate as is."""
    bname = ds['time'].attrs.get('bounds', 'time_bnds')
    if bname in ds:
        b = ds[bname].values
        return np.array([b0 + (b1 - b0) / 2 for b0, b1 in b])
    return ds['time'].values


# ----------------------------------------------------------------------------- calculation
def precip_total_for_file(in_path):
    """May-Oct precipitation total (mm) per year and grid point, plus the number of hours per year,
    for the hours of this file. Returns None if the file has no hours in MONTHS/YEARS."""
    ds = xr.open_dataset(in_path, use_cftime=True)
    da = ds['PRECT']
    lat = da.lat.values
    da = da.isel(lat=np.where((lat >= LATITUDE[0]) & (lat <= LATITUDE[1]))[0]).transpose('time', 'lat', 'lon')

    times = mid_interval_times(ds)
    months = np.array([t.month for t in times])
    years = np.array([t.year for t in times])
    keep = np.isin(months, MONTHS) & (years >= YEARS[0]) & (years <= YEARS[1])
    idx_all = np.where(keep)[0]
    if idx_all.size == 0:
        ds.close()
        return None

    sel_years = years[idx_all]
    out_years, out_sum, out_n = [], [], []
    for y0 in range(sel_years.min(), sel_years.max() + 1, BLOCK_YEARS):
        block_years = np.arange(y0, y0 + BLOCK_YEARS)
        idx = idx_all[np.isin(sel_years, block_years)]
        if idx.size == 0:
            continue
        p = da.isel(time=idx).values.astype(np.float64) * UNIT_FACTOR       # (hour, lat, lon), mm per hour
        yrs = years[idx]
        for y in np.unique(yrs):
            out_years.append(int(y))
            out_sum.append(p[yrs == y].sum(axis=0))
            out_n.append(int((yrs == y).sum()))

    times_sel = times[idx_all]
    result = xr.Dataset(
        {
            'pr_total': (('year', 'lat', 'lon'), np.stack(out_sum),
                         {'long_name': 'Precipitation total over the months', 'units': 'mm'}),
            'n_hours': (('year',), np.array(out_n), {'long_name': 'Number of hours summed'}),
        },
        coords={'year': np.array(out_years), 'lat': da.lat.values, 'lon': da.lon.values},
        attrs={'months': ','.join(str(m) for m in MONTHS), 'latitude': f'{LATITUDE[0]} to {LATITUDE[1]}',
               'source_file': os.path.basename(in_path),
               'first_hour': str(times_sel[0]), 'last_hour': str(times_sel[-1])},
    )
    ds.close()
    return result


def process_part(fname, input_dir, parts_dir, overwrite):
    out_path = os.path.join(parts_dir, part_name(FILENAME_PATTERN.match(fname)))
    if os.path.exists(out_path) and not overwrite:
        return fname, 'skipped (part exists)', 0.0
    t0 = time.time()
    result = precip_total_for_file(os.path.join(input_dir, fname))
    if result is None:
        result = xr.Dataset({'pr_total': (('year', 'lat', 'lon'), np.empty((0, 0, 0))),
                             'n_hours': (('year',), np.empty(0, dtype=int))})     # empty marker: file done
        write_atomic(result, out_path)
        return fname, 'no hours in MONTHS/YEARS', time.time() - t0
    write_atomic(result, out_path)
    yrs = result.year.values
    info = f"{yrs.size} years ({yrs[0]}-{yrs[-1]}), hours per year {sorted(set(result.n_hours.values.tolist()))}"
    return fname, info, time.time() - t0


def merge_member(member, part_files, parts_dir, output_dir):
    """Combine the parts of a member; a year split over two files (should not happen for May-Oct) is summed."""
    parts = []
    for f in sorted(part_files):
        with xr.open_dataset(os.path.join(parts_dir, f)) as ds:
            ds = ds.load()
        if ds.sizes.get('year', 0):
            parts.append(ds)
    ds = xr.concat(parts, dim='year')
    ds = ds.groupby('year').sum()
    ds.attrs['member'] = member
    expected = 31 + 30 + 31 + 31 + 30 + 31       # May-Oct days (no-leap calendar)
    bad = ds.year.values[ds.n_hours.values != expected * 24]
    out_path = os.path.join(output_dir, member_out_name(member))
    write_atomic(ds, out_path)
    return out_path, ds, bad


def area_means(field, land):
    """cos(lat)-weighted mean over land and over sea of a (lat, lon) field."""
    w = np.cos(np.deg2rad(field.lat.values))[:, None] * np.ones(field.sizes['lon'])
    v = field.values
    ok = np.isfinite(v)
    land_mean = np.average(v[land & ok], weights=w[land & ok])
    sea_mean = np.average(v[~land & ok], weights=w[~land & ok])
    return land_mean, sea_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', default=INPUT_DIR)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--workers', type=int, default=8, help='files processed in parallel')
    parser.add_argument('--overwrite', action='store_true', help='recompute parts whose output already exists')
    parser.add_argument('--dry-run', action='store_true', help='only list the files that would be processed')
    args = parser.parse_args()

    members = defaultdict(list)                                           # member -> input files
    for fname in sorted(os.listdir(args.input_dir)):
        m = FILENAME_PATTERN.match(fname)
        if m and m['exp'] == EXPERIMENT:
            members[member_name(m)].append(fname)
    files = [f for fl in members.values() for f in fl]

    print(f"{EXPERIMENT}: {len(members)} members, {len(files)} files; years {YEARS}, months {MONTHS}", flush=True)
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
        futures = {pool.submit(process_part, f, args.input_dir, parts_dir, args.overwrite): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fname = futures[fut]
            try:
                _, info, dt = fut.result()
                print(f"[{i}/{len(files)}] {fname}: {info}, {dt:.1f}s", flush=True)
            except Exception:
                failed.append(fname)
                print(f"[{i}/{len(files)}] {fname}: FAILED\n{traceback.format_exc()}", flush=True)

    # merge per member, then the mean over YEARS per member and for the ensemble
    member_means = {}
    for member, fl in members.items():
        if any(f in failed for f in fl):
            print(f"{member}: not merged, {sum(f in failed for f in fl)} file(s) failed", flush=True)
            continue
        try:
            out_path, ds, bad = merge_member(member, [part_name(FILENAME_PATTERN.match(f)) for f in fl],
                                             parts_dir, args.output_dir)
        except Exception:
            failed.append(member)
            print(f"{member}: merge FAILED\n{traceback.format_exc()}", flush=True)
            continue
        n_years = ds.sizes['year']
        note = '' if n_years == YEARS[1] - YEARS[0] + 1 else f'  WARNING: {n_years} years'
        note += f'  WARNING: incomplete May-Oct in {list(bad)}' if bad.size else ''
        print(f"{member}: {n_years} years -> {out_path}{note}", flush=True)
        member_means[member] = ds['pr_total'].mean('year')

    if member_means:
        any_field = next(iter(member_means.values()))
        land = load_land(any_field.lat.values, any_field.lon.values)
        stack = xr.concat(list(member_means.values()), dim=xr.DataArray(list(member_means), dims='member', name='member'))
        ens = xr.Dataset({'pr_total_mean': stack.mean('member'), 'pr_total_member_mean': stack})
        ens['pr_total_mean'].attrs = {'long_name': f'Ensemble-mean May-Oct precipitation total, mean {YEARS[0]}-{YEARS[1]}', 'units': 'mm'}
        ens['pr_total_member_mean'].attrs = {'long_name': f'May-Oct precipitation total per member, mean {YEARS[0]}-{YEARS[1]}', 'units': 'mm'}
        ens_path = os.path.join(args.output_dir, f'cesm2_hourly_precip_total_ensmean_{YEARS[0]}-{YEARS[1]}_{REGION_NAME}.nc')
        write_atomic(ens, ens_path)

        rows = []
        print(f"\nMean May-Oct precipitation {YEARS[0]}-{YEARS[1]}, 30-60N, cos(lat) area-weighted (m):", flush=True)
        print(f"  {'member':12s} {'land':>8s} {'sea':>8s}", flush=True)
        for member, field in member_means.items():
            lm, sm = area_means(field, land)
            rows.append((member, lm / 1000, sm / 1000))
            print(f"  {member:12s} {lm / 1000:8.4f} {sm / 1000:8.4f}", flush=True)
        lm, sm = area_means(ens['pr_total_mean'], land)
        lands = np.array([r[1] for r in rows])
        seas = np.array([r[2] for r in rows])
        print(f"  {'ensemble':12s} {lm / 1000:8.4f} {sm / 1000:8.4f}", flush=True)
        print(f"  member range: land {lands.min():.4f}-{lands.max():.4f}, sea {seas.min():.4f}-{seas.max():.4f}", flush=True)
        csv_path = os.path.join(args.output_dir, f'cesm2_hourly_precip_total_{YEARS[0]}-{YEARS[1]}_{REGION_NAME}_area_means.csv')
        with open(csv_path, 'w') as f:
            f.write('member,land_m,sea_m\n')
            for r in rows:
                f.write(f'{r[0]},{r[1]:.6f},{r[2]:.6f}\n')
            f.write(f'ensemble,{lm / 1000:.6f},{sm / 1000:.6f}\n')
        print(f"-> {ens_path}\n-> {csv_path}", flush=True)

    print(f"Done in {time.time() - t_total:.1f}s. {len(failed)} failure(s).", flush=True)
    for f in failed:
        print('  failed:', f, flush=True)


if __name__ == '__main__':
    main()
