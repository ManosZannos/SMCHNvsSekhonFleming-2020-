"""
preprocess_sf.py — Exact replication of Sekhon & Fleming 2020 preprocessing pipeline

Replicates:
  1. preprocess_data.py: resample to 1min, interpolate, fix heading 511
     — on FULL MONTH data per vessel (not per day)
  2. grid.py: geographic filter, SOG filter, ocean mask, grid splitting
     — on FULL MONTH data, overlapping grids with step=0.1° size=0.2°
  3. data.py normalization: LAT/LON min-max to [0,1], SOG/22, Heading/360

FIXES vs previous version:
  Bug #1: Grid step corrected to 0.1° (overlapping) — was 0.2° (non-overlapping)
          S&F grid.py: l2 += 0.1, l += 0.1 with grid_size=0.2
  Bug #3: Step 1 (resample/interpolate) now runs on the FULL MONTH raw data
          per vessel, not per day. This avoids splitting vessel trajectories
          at day boundaries, matching S&F preprocess_data.py behaviour.

Output: frame-format CSV files in dataset/noaa_jan2017_sf/processed/

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
from global_land_mask import globe  # pyright: ignore[reportMissingImports]
warnings.filterwarnings("ignore")

# ============================================================================
# S&F geographic constants (from geographic_utils.py and grid.py)
# ============================================================================
SF_LAT_MIN = 32.0
SF_LAT_MAX = 33.0
SF_LON_MIN = -118.0
SF_LON_MAX = -117.0

# Min-max normalization bounds (radians, as in data.py)
MIN_LAT = (math.pi / 180) * SF_LAT_MIN
MAX_LAT = (math.pi / 180) * SF_LAT_MAX
MIN_LON = (math.pi / 180) * SF_LON_MIN
MAX_LON = (math.pi / 180) * SF_LON_MAX

# S&F grid.py parameters
GRID_SIZE      = 0.2    # degrees — window size (unchanged)
GRID_STEP      = 0.1    # degrees — step (Bug #1 fix: was 0.2, S&F uses 0.1)
MIN_TIMESTAMPS = 1000   # minimum timestamps per grid cell
MIN_VESSELS    = 3      # minimum vessels per grid cell (>3)


def get_day_from_filename(filename):
    match = re.search(r'AIS_\d{4}_\d{2}_(\d{2})\.zip', filename)
    return int(match.group(1)) if match else None


def load_zip(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"No CSV in {zip_path}")
        with z.open(names[0]) as f:
            return pd.read_csv(f,
                               usecols=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'Heading'],
                               parse_dates=['BaseDateTime'])


# ============================================================================
# Step 1: preprocess_data.py replication
# Bug #3 fix: runs on FULL MONTH raw data (not per day)
# ============================================================================
def preprocess_step1(df):
    """
    Replicates preprocess_data.py on the concatenated monthly raw data.

    Per vessel (across full month):
    - ceil to minute, remove duplicates
    - resample 1min, interpolate (limit=5)
    - fix heading 511 (invalid → ffill/bfill)

    Running this on the full month avoids splitting vessel trajectories
    at day boundaries (Bug #3).
    """
    df = df.copy()
    df.sort_values(['BaseDateTime'], inplace=True)
    vessels = df['MMSI'].unique()

    out_frames = []
    for vessel in vessels:
        vessel_data = df.loc[df['MMSI'] == vessel].copy()
        vessel_data['BaseDateTime'] = vessel_data['BaseDateTime'].dt.ceil('min')
        vessel_data = vessel_data.loc[~vessel_data['BaseDateTime'].duplicated(keep='first')]
        vessel_data = vessel_data.set_index(['BaseDateTime']).resample('1min').interpolate(limit=5)
        vessel_data.reset_index('BaseDateTime', inplace=True)
        vessel_data = vessel_data.dropna(subset=['LAT', 'LON'])

        try:
            vessel_data.set_index(['BaseDateTime'], inplace=True)
            vessel_data['Heading'] = vessel_data['Heading'].astype('int32')
            if not len(vessel_data['Heading'].unique()) == 1:
                if int(511) in vessel_data['Heading'].values:
                    vessel_data['Heading'].replace(to_replace=511, method='ffill', inplace=True)
                    vessel_data['Heading'].replace(to_replace=511, method='bfill', inplace=True)
            vessel_data['MMSI'] = vessel
            out_frames.append(vessel_data)
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
# Step 2: grid.py filtering (on FULL MONTH data)
# ============================================================================
def preprocess_step2(df):
    """
    Replicates grid.py filtering (in order):
    - LAT: [32, 33], LON: [-118, -117]
    - SOG <= 22
    - Ocean mask (remove land coordinates)
    - Remove anchored vessels (max SOG <= 1 per vessel)
    - Remove timestamps with <= 3 vessels
    - Remove timestamps where max SOG <= 1
    """
    df = df.loc[
        (df['LAT'] >= SF_LAT_MIN) & (df['LAT'] <= SF_LAT_MAX) &
        (df['LON'] >= SF_LON_MIN) & (df['LON'] <= SF_LON_MAX)
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
    """Replicates grid.py ocean_mask() — removes vessels on land."""
    df = df.copy()
    df['is_ocean'] = globe.is_ocean(df['LAT'], df['LON'])
    df = df[df['is_ocean'] == True]
    return df.drop(['is_ocean'], axis=1)


def split_into_grids(df, grid_size=GRID_SIZE, grid_step=GRID_STEP):
    """
    Replicates grid.py grid splitting.

    Bug #1 fix: step=0.1° (overlapping 50%), size=0.2°
    S&F grid.py uses: l2 += 0.1, l += 0.1 with grid_size=0.2
    Previously we used step=size=0.2 (non-overlapping) — this roughly
    halved the number of valid grids.

    Bounds: inclusive on both sides (matching S&F: >= l and <= l+grid_size).
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
        df_lat = df.loc[(df['LAT'] >= l) & (df['LAT'] <= (l + grid_size))]
        while not l2 >= max_lon:
            df_grid = df_lat.loc[(df_lat['LON'] >= l2) & (df_lat['LON'] <= (l2 + grid_size))]

            if not df_grid.empty:
                groups = df_grid.groupby(['BaseDateTime'])
                df_grid = groups.filter(lambda x: len(x['MMSI']) > 2)

            if not df_grid.empty:
                timestamps = df_grid['BaseDateTime'].unique()
                vessels    = df_grid['MMSI'].unique()

                if len(timestamps) >= MIN_TIMESTAMPS and len(vessels) > MIN_VESSELS:
                    grids.append(df_grid.copy())

            l2 += grid_step   # Bug #1 fix: 0.1° step (was 0.2°)
        l += grid_step         # Bug #1 fix: 0.1° step (was 0.2°)

    return grids


# ============================================================================
# Step 3: data.py normalization replication
# ============================================================================
def normalize_sf(df):
    """
    Replicates data.py normalize():
    - LAT/LON: convert to radians, then min-max normalize to [0,1]
    - SOG: divide by 22
    - Heading: divide by 360
    """
    df = df.copy()
    df['LAT'] = (math.pi / 180) * df['LAT']
    df['LON'] = (math.pi / 180) * df['LON']

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
    Convert to frame_id format for TrajectoryDataset.
    frame_id = minutes elapsed from the first timestamp in this grid.
    """
    df = df.copy().sort_values('BaseDateTime')
    t0 = df['BaseDateTime'].min()
    df['frame_id'] = ((df['BaseDateTime'] - t0).dt.total_seconds() / 60.0).round().astype(int)
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
    print(f"S&F PIPELINE — NOAA AIS January 2017 (Full Month)")
    print(f"{'='*70}")
    print(f"Files: {len(zip_files)} days")
    print(f"Geographic bounds: LAT [{SF_LAT_MIN},{SF_LAT_MAX}], LON [{SF_LON_MIN},{SF_LON_MAX}]")
    print(f"Grid size: {GRID_SIZE}°, step: {GRID_STEP}° (overlapping — Bug #1 fix)")
    print(f"Min timestamps: {MIN_TIMESTAMPS}, Min vessels: >{MIN_VESSELS}")
    print(f"Step 1 runs on FULL MONTH data per vessel (Bug #3 fix)")
    print(f"{'='*70}\n")

    # =========================================================================
    # PASS 1: Load all raw days and concatenate
    # Bug #3 fix: Step 1 (resample) runs on the full month per vessel,
    # not per day. This prevents splitting trajectories at day boundaries.
    # =========================================================================
    print("PASS 1: Loading all raw daily files...")
    all_raw = []

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        day_num  = get_day_from_filename(filename)
        if day_num is None:
            continue
        try:
            df_raw = load_zip(zip_path)
            print(f"  Day {day_num:02d}: {len(df_raw):,} rows, {df_raw['MMSI'].nunique()} vessels")
            all_raw.append(df_raw)
        except Exception as e:
            print(f"  Day {day_num:02d}: ERROR — {e}")
            continue

    if not all_raw:
        print("ERROR: No raw data loaded!")
        return

    print(f"\nConcatenating {len(all_raw)} days of raw data...")
    df_all_raw = pd.concat(all_raw, ignore_index=True)
    df_all_raw.sort_values('BaseDateTime', inplace=True)
    print(f"Total raw: {len(df_all_raw):,} rows, {df_all_raw['MMSI'].nunique()} vessels")

    # =========================================================================
    # Step 1: Resample/interpolate on full month per vessel
    # =========================================================================
    print(f"\nStep 1: Resampling full month per vessel (this may take a while)...")
    df = preprocess_step1(df_all_raw)
    if df.empty:
        print("ERROR: empty after Step 1!")
        return
    print(f"After Step 1: {len(df):,} rows, {df['MMSI'].nunique()} vessels")

    # =========================================================================
    # Step 2: Filtering (geographic, SOG, ocean mask, crowd filter)
    # =========================================================================
    print(f"\nStep 2: Filtering...")
    df = preprocess_step2(df)
    if df.empty:
        print("ERROR: empty after Step 2!")
        return
    print(f"After Step 2: {len(df):,} rows, {df['MMSI'].nunique()} vessels")
    print(f"Unique timestamps: {df['BaseDateTime'].nunique():,}")

    # =========================================================================
    # PASS 2: Grid split on full month data
    # Bug #1 fix: step=0.1° overlapping grids (was 0.2° non-overlapping)
    # =========================================================================
    print(f"\nPASS 2: Grid split (step={GRID_STEP}°, size={GRID_SIZE}°)...")
    grids = split_into_grids(df)
    print(f"Valid grids found: {len(grids)}")

    # =========================================================================
    # PASS 3: Normalize and save each grid
    # =========================================================================
    print(f"\nPASS 3: Normalizing and saving grids...")
    total_saved = 0

    for g_idx, grid_df in enumerate(grids):
        grid_norm = normalize_sf(grid_df)
        if grid_norm.empty:
            continue

        frames = to_frame_format(grid_norm)

        out_csv = os.path.join(processed_dir, f"grid_{g_idx:03d}.csv")
        frames.to_csv(out_csv, index=False)
        total_saved += 1

        vessels    = frames['vessel_id'].nunique()
        timestamps = frames['frame_id'].nunique()
        print(f"  grid_{g_idx:03d}: {vessels} vessels, {timestamps} frames → {out_csv}")

    print(f"\n{'='*70}")
    print(f"PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Dataset:     {dataset_name}")
    print(f"Location:    {processed_dir}/")
    print(f"Total grids: {total_saved}")
    print(f"\nNext: python train.py --dataset {dataset_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()