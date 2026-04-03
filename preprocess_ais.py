"""
NOAA AIS Data Preprocessing Script (S&F 2017 aligned)

Creates frame-format CSV files compatible with TrajectoryDataset:
  frame_id, vessel_id, LON, LAT, SOG, Heading

Processes all AIS files from a raw data folder and splits them into train/val/test.

TWO-PASS APPROACH (for dataset-level z-score normalization):
  Pass 1: Compute global statistics from train data only
  Pass 2: Apply those statistics to all splits (train/val/test)

Dataset: NOAA AIS January 2017, San Diego Harbor, Zone 11
Split: 80/10/10 (aligned with Sekhon & Fleming 2020)

Usage:
  python preprocess_ais.py
"""

import os
import json
import glob
import re
import numpy as np
import pandas as pd
from utils import (
    load_noaa_csv,
    preprocess_noaa_to_frames,
    save_frames_csv,
    clean_abnormal_data_noaa,
    resample_interpolate_1min,
    filter_timestamps_min_vessels,
)


def get_day_from_filename(filename):
    """Extract day number from AIS filename (e.g., AIS_2017_01_01.csv -> 1)"""
    match = re.search(r'AIS_\d{4}_\d{2}_(\d{2})\.csv', filename)
    if match:
        return int(match.group(1))
    return None


def get_date_str_from_filename(filename):
    """Extract full date string from AIS filename (e.g., AIS_2017_01_01.csv -> 2017_01_01)"""
    match = re.search(r'AIS_(\d{4}_\d{2}_\d{2})\.csv', filename)
    if match:
        return match.group(1)
    return None


def main():
    # ----------------------------
    # Input
    # ----------------------------
    raw_data_folder = "data/raw/2017_01"  # Folder with all AIS_*.csv files
    nrows = None  # Use None for full processing, or set limit for testing (e.g., 2_000_000)

    # ----------------------------
    # Train/Val/Test Split (Sekhon & Fleming 2020: 80/10/10)
    # 31 days in January 2017:
    # Train: days 1-25  (25 days → ~80%)
    # Val:   days 26-28  (3 days → ~10%)
    # Test:  days 29-31  (3 days → ~10%)
    # ----------------------------
    train_days = list(range(1, 26))    # Days 1-25  → train/ (80%)
    val_days   = list(range(26, 29))   # Days 26-28 → val/   (10%)
    test_days  = list(range(29, 32))   # Days 29-31 → test/  (10%)

    # ----------------------------
    # Paper preprocessing params (Sekhon & Fleming 2020)
    # ----------------------------
    lat_range = (30.0, 35.0)
    lon_range = (-120.0, -115.0)
    sog_range = (1.0, 22.0)
    heading_range = (0.0, 360.0)
    min_vessels_per_timestamp = 3  # Paper's value: >3 concurrent vessels
    max_gap_minutes = 10           # Gaps > 10 min split vessel trajectory

    # ----------------------------
    # Output
    # ----------------------------
    dataset_name = "noaa_jan2017"
    dataset_base = os.path.join("dataset", dataset_name)
    global_stats_path = os.path.join(dataset_base, "global_stats.json")

    # Find all AIS csv files
    csv_files = sorted(glob.glob(os.path.join(raw_data_folder, "AIS_*.csv")))

    if not csv_files:
        print(f"ERROR: No AIS_*.csv files found in {raw_data_folder}")
        return

    # Categorize files by split
    train_files = []
    val_files   = []
    test_files  = []

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        day_num = get_day_from_filename(filename)
        if day_num is None:
            continue
        if day_num in train_days:
            train_files.append(csv_path)
        elif day_num in val_days:
            val_files.append(csv_path)
        elif day_num in test_days:
            test_files.append(csv_path)

    print(f"\n{'='*80}")
    print(f"DATASET SPLIT OVERVIEW (Sekhon & Fleming 2020 aligned)")
    print(f"{'='*80}")
    print(f"Dataset:     NOAA AIS January 2017, San Diego Harbor (Zone 11)")
    print(f"Split ratio: 80/10/10")
    print(f"Train files: {len(train_files)} (days 1-25)")
    print(f"Val files:   {len(val_files)}   (days 26-28)")
    print(f"Test files:  {len(test_files)}   (days 29-31)")
    print(f"{'='*80}\n")

    # =========================================================================
    # PASS 1: Compute global statistics from TRAIN data only (streaming)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"PASS 1: Computing global statistics from TRAIN data (streaming)")
    print(f"{'='*80}\n")

    stats_cols = ["LON", "LAT", "SOG", "Heading"]
    streaming_stats = {col: {"count": 0, "mean": 0.0, "M2": 0.0} for col in stats_cols}

    total_train_rows = 0
    total_train_vessels = set()

    for csv_path in train_files:
        filename = os.path.basename(csv_path)
        day_num = get_day_from_filename(filename)
        date_str = get_date_str_from_filename(filename)

        if day_num is None or date_str is None:
            print(f"Skipping {filename} - couldn't parse date")
            continue

        print(f"[TRAIN] Processing day {day_num:02d}: {filename}")

        try:
            df_raw = load_noaa_csv(csv_path, nrows=nrows)
            print(f"  Loaded: {len(df_raw):,} rows")

            # Steps 1-3: clean, resample, filter (no z-score yet)
            df = clean_abnormal_data_noaa(df_raw, lat_range, lon_range, sog_range, heading_range)
            df = resample_interpolate_1min(df, freq="1min", rolling_window=5, max_gap_minutes=max_gap_minutes)
            df = filter_timestamps_min_vessels(df, min_vessels_per_timestamp)

            # Update streaming statistics (batch Welford's algorithm)
            for col in stats_cols:
                values = df[col].values
                n_new = len(values)
                if n_new == 0:
                    continue

                mean_new = float(values.mean())
                var_new  = float(values.var(ddof=1)) if n_new > 1 else 0.0
                M2_new   = var_new * (n_new - 1)

                acc   = streaming_stats[col]
                n_old = acc["count"]

                if n_old == 0:
                    acc["count"] = n_new
                    acc["mean"]  = mean_new
                    acc["M2"]    = M2_new
                else:
                    n_combined   = n_old + n_new
                    delta        = mean_new - acc["mean"]
                    acc["mean"]  = acc["mean"] + delta * n_new / n_combined
                    acc["M2"]    = acc["M2"] + M2_new + delta**2 * n_old * n_new / n_combined
                    acc["count"] = n_combined

            total_train_rows += len(df)
            total_train_vessels.update(df["MMSI"].unique())
            print(f"  Processed: {len(df):,} rows\n")

            del df
            del df_raw

        except Exception as e:
            print(f"ERROR processing {filename}: {e}\n")
            continue

    if streaming_stats["LON"]["count"] == 0:
        print("ERROR: No train data collected!")
        return

    print(f"\nTotal train data processed:")
    print(f"  Rows:    {total_train_rows:,}")
    print(f"  Vessels: {len(total_train_vessels)}")

    # Compute final statistics
    print(f"\nComputing global statistics (LON, LAT, SOG, Heading)...")
    global_stats = {}
    for col in stats_cols:
        count    = streaming_stats[col]["count"]
        mean     = streaming_stats[col]["mean"]
        variance = streaming_stats[col]["M2"] / (count - 1) if count > 1 else 0.0
        std      = np.sqrt(variance)
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        global_stats[col] = {"mean": float(mean), "std": float(std)}
        print(f"  {col}: μ={mean:.6f}, σ={std:.6f}")

    os.makedirs(dataset_base, exist_ok=True)
    with open(global_stats_path, "w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)
    print(f"\n✓ Saved global statistics: {global_stats_path}")

    del streaming_stats
    del total_train_vessels

    # =========================================================================
    # PASS 2: Process all files with global statistics
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"PASS 2: Processing all files with global statistics")
    print(f"{'='*80}\n")

    all_files = (
        [(csv_path, "train") for csv_path in train_files] +
        [(csv_path, "val")   for csv_path in val_files]   +
        [(csv_path, "test")  for csv_path in test_files]
    )

    for csv_path, split in all_files:
        filename = os.path.basename(csv_path)
        day_num  = get_day_from_filename(filename)
        date_str = get_date_str_from_filename(filename)

        if day_num is None or date_str is None:
            print(f"Skipping {filename} - couldn't parse date")
            continue

        out_dir   = os.path.join(dataset_base, split)
        out_csv   = os.path.join(out_dir, f"day_{date_str}.csv")
        out_stats = os.path.join(out_dir, f"day_{date_str}_stats.json")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[{split.upper()}] Processing day {day_num:02d}: {filename}")

        try:
            df_raw = load_noaa_csv(csv_path, nrows=nrows)
            print(f"  Loaded: {len(df_raw):,} rows")

            frames_df, _ = preprocess_noaa_to_frames(
                df_raw,
                lat_range=lat_range,
                lon_range=lon_range,
                sog_range=sog_range,
                heading_range=heading_range,
                min_vessels_per_timestamp=min_vessels_per_timestamp,
                max_gap_minutes=max_gap_minutes,
                do_zscore=True,
                zscore_stats=global_stats,
            )
            print(f"  Normalized: {len(frames_df):,} frame rows")

            save_frames_csv(frames_df, out_csv)
            print(f"  ✓ Saved: {out_csv}")

            with open(out_stats, "w", encoding="utf-8") as f:
                json.dump(global_stats, f, indent=2)
            print(f"  ✓ Saved stats: {out_stats}\n")

        except Exception as e:
            print(f"ERROR processing {filename}: {e}\n")
            continue

    print(f"{'='*80}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Dataset:      {dataset_name}")
    print(f"Location:     {dataset_base}/")
    print(f"Global stats: {global_stats_path}")
    print(f"\nTrain files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")
    print(f"\nNext: python train.py --dataset {dataset_name}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()