"""
preprocess_sf.py — Exact replication of Sekhon & Fleming 2020 preprocessing pipeline

Replicates:
  1. preprocess_data.py: resample to 1min, interpolate, fix heading 511
  2. grid.py: geographic filter, SOG filter, ocean mask, grid splitting
  3. data.py normalization: LAT/LON min-max to [0,1], SOG/22, Heading/360

NO day-level split here — split is done at sample level in TrajectoryDataset
(aligned with S&F data.py random_split).

Output: frame-format CSV files in dataset/noaa_jan2017_sf/processed/

Usage:
  python preprocess_sf.py
"""

import shutil
import os
import re
import glob
import math
import zipfile
import warnings
import numpy as np
import pandas as pd
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
GRID_SIZE      = 0.2    # degrees (non-overlapping)
MIN_TIMESTAMPS = 1000   # minimum timestamps per grid cell
MIN_VESSELS    = 3      # minimum vessels per grid cell (>3)


def get_day_from_filename(filename):
    match = re.search(r'AIS_\d{4}_\d{2}_(\d{2})\.zip', filename)
    return int(match.group(1)) if match else None


def get_date_str_from_filename(filename):
    match = re.search(r'AIS_(\d{4}_\d{2}_\d{2})\.zip', filename)
    return match.group(1) if match else None


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
# ============================================================================
def preprocess_step1(df):
    """
    Replicates preprocess_data.py:
    - Sort by BaseDateTime
    - Per vessel: ceil to minute, remove duplicates, resample 1min, interpolate
    - Fix heading 511 (invalid value → ffill/bfill)
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
# Step 2: grid.py replication
# ============================================================================
def preprocess_step2(df):
    """
    Replicates grid.py filtering:
    - LAT: [32, 33], LON: [-118, -117]
    - SOG <= 22
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


def split_into_grids(df, grid_size=GRID_SIZE):
    """
    Replicates grid.py grid splitting — non-overlapping (step = grid_size).
    Returns list of DataFrames, one per valid grid cell.
    """
    if df.empty:
        return []

    grids = []
    min_lat = int(np.floor(df['LAT'].min()))
    max_lat = int(np.ceil(df['LAT'].max()))
    min_lon = int(np.floor(df['LON'].min()))
    max_lon = int(np.ceil(df['LON'].max()))

    l = min_lat
    while l < max_lat:
        l2 = min_lon
        df_lat = df.loc[(df['LAT'] >= l) & (df['LAT'] <= (l + grid_size))]
        while l2 < max_lon:
            df_grid = df_lat.loc[(df_lat['LON'] >= l2) & (df_lat['LON'] <= (l2 + grid_size))]

            if not df_grid.empty:
                groups = df_grid.groupby(['BaseDateTime'])
                df_grid = groups.filter(lambda x: len(x['MMSI']) > 2)

            if not df_grid.empty:
                timestamps = df_grid['BaseDateTime'].unique()
                vessels    = df_grid['MMSI'].unique()

                if len(timestamps) >= MIN_TIMESTAMPS and len(vessels) > MIN_VESSELS:
                    grids.append(df_grid.copy())

            l2 += GRID_SIZE  # non-overlapping
        l += GRID_SIZE        # non-overlapping

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
    """Convert to frame_id format for TrajectoryDataset."""
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
    # All grids go to processed/ — split is done at sample level in TrajectoryDataset
    processed_dir   = os.path.join(dataset_base, "processed")
    # Auto-clean existing processed data
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
    print(f"Files: {len(zip_files)} days")
    print(f"Geographic bounds: LAT [{SF_LAT_MIN},{SF_LAT_MAX}], LON [{SF_LON_MIN},{SF_LON_MAX}]")
    print(f"Grid size: {GRID_SIZE}° (non-overlapping)")
    print(f"Min timestamps: {MIN_TIMESTAMPS}")
    print(f"Output: {processed_dir}/")
    print(f"NOTE: train/val/test split done at sample level in TrajectoryDataset")
    print(f"{'='*70}\n")

    total_grids = 0

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        day_num  = get_day_from_filename(filename)
        date_str = get_date_str_from_filename(filename)

        if day_num is None or date_str is None:
            continue

        print(f"Day {day_num:02d}: {filename}")

        try:
            df_raw = load_zip(zip_path)
            print(f"  Loaded: {len(df_raw):,} rows, {df_raw['MMSI'].nunique()} vessels")

            # Step 1: resample/interpolate
            df = preprocess_step1(df_raw)
            if df.empty:
                print(f"  SKIP: empty after Step 1")
                continue
            print(f"  After Step 1: {len(df):,} rows")

            # Step 2: filtering
            df = preprocess_step2(df)
            if df.empty:
                print(f"  SKIP: empty after Step 2")
                continue
            print(f"  After Step 2: {len(df):,} rows, {df['MMSI'].nunique()} vessels")

            # Step 2b: grid splitting
            grids = split_into_grids(df)
            print(f"  Grids: {len(grids)}")

            for g_idx, grid_df in enumerate(grids):
                # Step 3: normalize
                grid_norm = normalize_sf(grid_df)
                if grid_norm.empty:
                    continue

                # Convert to frame format
                frames = to_frame_format(grid_norm)

                # Save to processed/ (no split yet)
                out_csv = os.path.join(processed_dir, f"day_{date_str}_grid{g_idx:02d}.csv")
                frames.to_csv(out_csv, index=False)
                total_grids += 1

            print(f"  ✓ Saved {len(grids)} grids\n")

        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

    print(f"{'='*70}")
    print(f"PREPROCESSING COMPLETE!")
    print(f"{'='*70}")
    print(f"Dataset:     {dataset_name}")
    print(f"Location:    {processed_dir}/")
    print(f"Total grids: {total_grids}")
    print(f"\nNext: python train.py --dataset {dataset_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()