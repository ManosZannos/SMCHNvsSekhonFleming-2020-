"""
preprocess_sf.py — Exact replication of Sekhon & Fleming 2020 preprocessing pipeline

Exact pipeline order matching S&F:
  1. preprocess_data.py logic: resample/interpolate PER DAY, NO filtering here
     Output: one processed DataFrame per day, then concat → full month zone file
  2. grid.py logic: ALL filtering on the full month zone file
     - geographic bounds [32,33]x[-118,-117], SOG, ocean mask, anchored, crowd
     - grid split with size=0.2°, step=0.1° (overlapping)
  3. data.py normalize(): uses geographic_utils.py bounds [32,35]x[-120,-117]
     LAT/LON radians → min-max [0,1] with WIDER bounds, SOG/22, Heading/360

KEY FIX vs previous versions:
  - Normalization bounds come from geographic_utils.py (active lines):
      min_lat, max_lat = 32, 35  (NOT 32, 33)
      min_lon, max_lon = -120, -117  (NOT -118, -117)
    This is what data.py normalize() and scale_values() use for ADE/FDE.
  - Grid filtering still uses [32,33]x[-118,-117] (grid.py get_grids)
  - Grid step = 0.1° (overlapping), grid size = 0.2°
  - Step 1 (resample) PER DAY — bounded index, no hang

Output: dataset/noaa_jan2017_sf/processed/grid_NNN.csv

Usage:
  python preprocess_sf.py
"""

import os
import re
import glob
import math
import shutil
import zipfile
import warnings
import numpy as np
import pandas as pd
from global_land_mask import globe
warnings.filterwarnings("ignore")

# ============================================================================
# geographic_utils.py bounds — used for grid.py geographic filtering
# ============================================================================
GRID_LAT_MIN = 32.0
GRID_LAT_MAX = 33.0
GRID_LON_MIN = -118.0
GRID_LON_MAX = -117.0

# ============================================================================
# geographic_utils.py normalization bounds (ACTIVE lines in geographic_utils.py)
# min_lat, max_lat, min_lon, max_lon = 32, 35, -120, -117  → converted to radians
# This is what data.py normalize() uses for min-max scaling
# ============================================================================
NORM_LAT_MIN_DEG = 32.0
NORM_LAT_MAX_DEG = 35.0
NORM_LON_MIN_DEG = -120.0
NORM_LON_MAX_DEG = -117.0

MIN_LAT = (math.pi / 180) * NORM_LAT_MIN_DEG
MAX_LAT = (math.pi / 180) * NORM_LAT_MAX_DEG
MIN_LON = (math.pi / 180) * NORM_LON_MIN_DEG
MAX_LON = (math.pi / 180) * NORM_LON_MAX_DEG

# grid.py parameters
GRID_SIZE      = 0.2    # degrees — window size
GRID_STEP      = 0.1    # degrees — step (overlapping, matches grid.py l+=0.1)
MIN_TIMESTAMPS = 1000   # minimum timestamps per valid grid cell
MIN_VESSELS    = 3      # minimum vessels per valid grid cell (strictly >3)


def get_day_from_filename(filename):
    match = re.search(r'AIS_\d{4}_\d{2}_(\d{2})\.zip', filename)
    return int(match.group(1)) if match else None


def load_zip(zip_path):
    """Load one daily AIS zip → DataFrame."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV in {zip_path}")
        with z.open(names[0]) as f:
            return pd.read_csv(
                f,
                usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'Heading'],
                parse_dates=['BaseDateTime']
            )


# ============================================================================
# Step 1: preprocess_data.py replication — PER DAY, NO filtering
# ============================================================================
def preprocess_step1_day(df):
    """
    Exact replication of S&F preprocess_data.py for one daily file.

    Per vessel:
      - ceil BaseDateTime to minute
      - drop duplicate timestamps (keep first)
      - resample to 1min, interpolate(limit=5)
      - dropna on LAT/LON  ← gaps >5min become NaN → trajectory splits naturally
      - fix heading 511 (invalid → ffill/bfill)

    NO geographic or SOG filtering here — that is grid.py's job.
    Running per day keeps resample index bounded to ~1440 minutes (no hang).
    """
    df = df.copy()
    df.sort_values(['BaseDateTime'], inplace=True)
    vessels = df['MMSI'].unique()

    out_frames = []
    for vessel in vessels:
        vessel_data = df.loc[df['MMSI'] == vessel].copy()

        vessel_data['BaseDateTime'] = vessel_data['BaseDateTime'].dt.ceil('min')
        vessel_data = vessel_data.loc[~vessel_data['BaseDateTime'].duplicated(keep='first')]

        vessel_data = (vessel_data
                       .set_index('BaseDateTime')
                       .resample('1min')
                       .interpolate(limit=5))
        vessel_data.reset_index('BaseDateTime', inplace=True)
        vessel_data = vessel_data.dropna(subset=['LAT', 'LON'])

        try:
            vessel_data.set_index('BaseDateTime', inplace=True)
            vessel_data['Heading'] = vessel_data['Heading'].astype('int32')
            # S&F preprocess_data.py: _append is INSIDE the 'if not unique==1' block
            # Vessels with constant heading (e.g. always 0 or always 511) are excluded
            if not len(vessel_data['Heading'].unique()) == 1:
                if int(511) in vessel_data['Heading'].values:
                    vessel_data['Heading'].replace(to_replace=511, method='ffill', inplace=True)
                    vessel_data['Heading'].replace(to_replace=511, method='bfill', inplace=True)
                vessel_data['MMSI'] = vessel
                out_frames.append(vessel_data)   # ← ONLY here, matching S&F
        except ValueError:
            continue

    if not out_frames:
        return pd.DataFrame()

    result = pd.concat(out_frames)
    result.index.name = 'BaseDateTime'
    result.reset_index(inplace=True)
    result.sort_values(['BaseDateTime'], inplace=True)
    return result


# ============================================================================
# Step 2: grid.py filtering — on FULL MONTH concat
# ============================================================================
def preprocess_step2_grid(df):
    """
    Exact replication of grid.py get_grids() filtering sequence.

    Geographic filter uses GRID bounds [32,33]x[-118,-117] (grid.py).
    Order matches grid.py exactly:
      1. Geographic bounds
      2. SOG <= 22
      3. Ocean mask
      4. Remove anchored vessels (max SOG <= 1 per vessel)
      5. Remove timestamps with <= 3 vessels
      6. Remove timestamps where max SOG <= 1
    """
    df = df.loc[
        (df['LAT'] >= GRID_LAT_MIN) & (df['LAT'] <= GRID_LAT_MAX) &
        (df['LON'] >= GRID_LON_MIN) & (df['LON'] <= GRID_LON_MAX)
    ].copy()
    if df.empty:
        return df

    df = df.loc[abs(df['SOG']) <= 22]
    if df.empty:
        return df

    df = ocean_mask(df)
    if df.empty:
        return df

    groups = df.groupby(['MMSI'])
    df = groups.filter(lambda x: abs(x['SOG']).max() > 1.0)
    if df.empty:
        return df

    groups = df.groupby(['BaseDateTime'])
    df = groups.filter(lambda x: len(x['MMSI']) > 3)
    if df.empty:
        return df

    groups = df.groupby(['BaseDateTime'])
    df = groups.filter(lambda x: abs(x['SOG'].max()) > 1.0)

    return df


def ocean_mask(df):
    """Exact replication of grid.py ocean_mask()."""
    df = df.copy()
    df['is_ocean'] = globe.is_ocean(df['LAT'], df['LON'])
    df = df[df['is_ocean'] == True]
    return df.drop(['is_ocean'], axis=1)


def split_into_grids(df):
    """
    Exact replication of grid.py get_grids() grid splitting loop.

    size=0.2°, step=0.1° (overlapping) — matches grid.py:
      l2 += 0.1 / l += 0.1 with grid_size=0.2
    Bounds: inclusive on both sides (>= and <=) — matches grid.py.
    Valid: len(timestamps) >= 1000 AND len(vessels) > 3.
    """
    if df.empty:
        return []

    grids = []
    min_lat = int(np.floor(df['LAT'].min()))
    max_lat = int(np.ceil(df['LAT'].max()))
    min_lon = int(np.floor(df['LON'].min()))
    max_lon = int(np.ceil(df['LON'].max()))

    l = min_lat
    while not l >= max_lat:
        l2 = min_lon
        df_lat = df.loc[(df['LAT'] >= l) & (df['LAT'] <= (l + GRID_SIZE))]
        while not l2 >= max_lon:
            df_grid = df_lat.loc[
                (df_lat['LON'] >= l2) & (df_lat['LON'] <= (l2 + GRID_SIZE))
            ]

            if not df_grid.empty:
                groups = df_grid.groupby(['BaseDateTime'])
                df_grid = groups.filter(lambda x: len(x['MMSI']) > 2)

            if not df_grid.empty:
                vessels    = df_grid['MMSI'].unique()
                timestamps = df_grid['BaseDateTime'].unique()

                if len(timestamps) >= MIN_TIMESTAMPS and len(vessels) > MIN_VESSELS:
                    print(f"  Grid LAT [{l:.1f},{l+GRID_SIZE:.1f}] "
                          f"LON [{l2:.1f},{l2+GRID_SIZE:.1f}]: "
                          f"{len(vessels)} vessels, {len(timestamps)} timestamps")
                    grids.append(df_grid.copy())

            l2 = round(l2 + GRID_STEP, 10)
        l = round(l + GRID_STEP, 10)

    return grids


# ============================================================================
# Step 3: data.py normalize() — with geographic_utils.py WIDER bounds
# ============================================================================
def normalize_sf(df):
    """
    Exact replication of data.py normalize() using geographic_utils.py bounds.

    IMPORTANT: bounds are from geographic_utils.py ACTIVE lines:
      min_lat, max_lat, min_lon, max_lon = 32, 35, -120, -117
    NOT the commented-out Zone 11 bounds [32,33]x[-118,-117].

    These wider bounds are also used by scale_values() in geographic_utils.py
    when computing equirectangular_distance for ADE/FDE — so normalization
    and error computation must use the same bounds.
    """
    df = df.copy()
    df['LAT'] = (math.pi / 180) * df['LAT']
    df['LON'] = (math.pi / 180) * df['LON']

    # data.py normalize(): filter to normalization bounds
    df = df.loc[
        (df['LAT'] <= MAX_LAT) & (df['LAT'] >= MIN_LAT) &
        (df['LON'] <= MAX_LON) & (df['LON'] >= MIN_LON)
    ]
    if df.empty:
        return df

    df['LAT'] = (df['LAT'] - MIN_LAT) / (MAX_LAT - MIN_LAT)
    df['LON'] = (df['LON'] - MIN_LON) / (MAX_LON - MIN_LON)
    df['SOG']     = df['SOG'] / 22
    df['Heading'] = df['Heading'] / 360

    return df


def to_frame_format(df):
    """
    Convert BaseDateTime → integer frame_id (minutes from grid start).
    Rename MMSI → vessel_id.
    Output columns: frame_id, vessel_id, LON, LAT, SOG, Heading
    """
    df = df.copy().sort_values('BaseDateTime')
    t0 = df['BaseDateTime'].min()
    df['frame_id'] = (
        (df['BaseDateTime'] - t0).dt.total_seconds() / 60.0
    ).round().astype(int)
    df = df.rename(columns={'MMSI': 'vessel_id'})
    return df[['frame_id', 'vessel_id', 'LON', 'LAT', 'SOG', 'Heading']]


# ============================================================================
# Main
# ============================================================================
def main():
    raw_data_folder = "data/raw/2017_01"
    dataset_name    = "noaa_jan2017_sf"
    dataset_base    = os.path.join("dataset", dataset_name)
    processed_dir   = os.path.join(dataset_base, "processed")

    if os.path.exists(processed_dir):
        print(f"Clearing existing processed data: {processed_dir}")
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    zip_files = sorted(glob.glob(os.path.join(raw_data_folder, "AIS_*.zip")))
    if not zip_files:
        print(f"ERROR: No AIS_*.zip files found in {raw_data_folder}")
        return

    print(f"\n{'='*70}")
    print(f"S&F PIPELINE — NOAA AIS January 2017")
    print(f"{'='*70}")
    print(f"Grid filter bounds:  LAT [{GRID_LAT_MIN},{GRID_LAT_MAX}], "
          f"LON [{GRID_LON_MIN},{GRID_LON_MAX}]")
    print(f"Normalization bounds: LAT [{NORM_LAT_MIN_DEG},{NORM_LAT_MAX_DEG}], "
          f"LON [{NORM_LON_MIN_DEG},{NORM_LON_MAX_DEG}] (geographic_utils.py)")
    print(f"Grid: size={GRID_SIZE}°, step={GRID_STEP}° (overlapping)")
    print(f"Step 1: resample PER DAY | Step 2: filter on FULL MONTH concat")
    print(f"{'='*70}\n")

    # =========================================================================
    # PASS 1 — preprocess_data.py: resample each day, no filtering
    #
    # NOTE: S&F raw CSVs are pre-filtered per zone before preprocess_data.py.
    # Our raw data covers the entire US coast (218M rows/month). We apply a
    # loose geographic pre-filter (wider than grid bounds) BEFORE resampling
    # to reduce memory and runtime — equivalent to S&F zone-split input files.
    # Pre-filter: LAT [31,34], LON [-119,-116] — wider than grid [32,33]x[-118,-117]
    # =========================================================================
    PRE_LAT_MIN, PRE_LAT_MAX = 31.0, 34.0
    PRE_LON_MIN, PRE_LON_MAX = -119.0, -116.0

    print("PASS 1: Resampling each day (preprocess_data.py)...")
    print(f"  Geographic pre-filter: LAT [{PRE_LAT_MIN},{PRE_LAT_MAX}], "
          f"LON [{PRE_LON_MIN},{PRE_LON_MAX}] (zone-equivalent, wider than grid)\n")
    all_days = []

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        day_num  = get_day_from_filename(filename)
        if day_num is None:
            continue
        try:
            df_raw = load_zip(zip_path)

            # Pre-filter: reduces ~7M rows/day to manageable size before resample
            df_raw = df_raw.loc[
                (df_raw['LAT'] >= PRE_LAT_MIN) & (df_raw['LAT'] <= PRE_LAT_MAX) &
                (df_raw['LON'] >= PRE_LON_MIN) & (df_raw['LON'] <= PRE_LON_MAX)
            ]
            if df_raw.empty:
                print(f"  Day {day_num:02d}: empty after pre-filter, skip")
                continue

            df_day = preprocess_step1_day(df_raw)
            if df_day.empty:
                print(f"  Day {day_num:02d}: empty after resample, skip")
                continue
            print(f"  Day {day_num:02d}: {len(df_raw):,} pre-filtered → "
                  f"{len(df_day):,} resampled rows")
            all_days.append(df_day)
        except Exception as e:
            print(f"  Day {day_num:02d}: ERROR — {e}")
            continue

    if not all_days:
        print("ERROR: No data after Step 1!")
        return

    # Concat → full month (equivalent to S&F processed_data/11.csv)
    print(f"\nConcatenating {len(all_days)} days...")
    df_month = pd.concat(all_days, ignore_index=True)
    df_month.sort_values('BaseDateTime', inplace=True)
    print(f"Full month: {len(df_month):,} rows, "
          f"{df_month['MMSI'].nunique()} vessels, "
          f"{df_month['BaseDateTime'].nunique():,} unique timestamps")

    # =========================================================================
    # PASS 2 — grid.py: filter + grid split on full month
    # =========================================================================
    print(f"\nPASS 2: Filtering (grid.py) on full month data...")
    df_filtered = preprocess_step2_grid(df_month)
    if df_filtered.empty:
        print("ERROR: empty after filtering!")
        return
    print(f"After filtering: {len(df_filtered):,} rows, "
          f"{df_filtered['MMSI'].nunique()} vessels, "
          f"{df_filtered['BaseDateTime'].nunique():,} timestamps")

    print(f"\nGrid split (size={GRID_SIZE}°, step={GRID_STEP}°)...")
    grids = split_into_grids(df_filtered)
    print(f"Valid grids: {len(grids)}")

    if not grids:
        print("ERROR: No valid grids found!")
        return

    # =========================================================================
    # PASS 3 — data.py normalize + save
    # =========================================================================
    print(f"\nPASS 3: Normalizing (geographic_utils.py bounds) and saving...")
    total_saved = 0

    for g_idx, grid_df in enumerate(grids):
        grid_norm = normalize_sf(grid_df)
        if grid_norm.empty:
            print(f"  grid_{g_idx:03d}: empty after normalization, skip")
            continue

        frames  = to_frame_format(grid_norm)
        out_csv = os.path.join(processed_dir, f"grid_{g_idx:03d}.csv")
        frames.to_csv(out_csv, index=False)
        total_saved += 1

        print(f"  grid_{g_idx:03d}: "
              f"{frames['vessel_id'].nunique()} vessels, "
              f"{frames['frame_id'].nunique()} frames → {out_csv}")

    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total grids saved: {total_saved}")
    print(f"Output: {processed_dir}/")
    print(f"\nNext: python train.py --dataset {dataset_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()