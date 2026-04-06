"""
utils.py

Includes:
1) NOAA AIS preprocessing (paper-aligned) -> frame-format CSV compatible with TrajectoryDataset:
   frame_id, vessel_id, LON, LAT, SOG, Heading

2) TrajectoryDataset: scene-based sliding window (aligned with Sekhon & Fleming 2020).
   Each window covers ALL vessels present in a scene at consecutive timestamps.
   This matches S&F data.py logic:
   - shift = sequence_length (obs_len) — no overlap between windows
   - sample = all vessels present in obs window
   - filters: consecutive timestamps, vessels moving, >= 3 valid vessels

NOTE: Status column removed — not present in 2017 NOAA AIS data.
"""

import os
import math
import zipfile

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from tqdm import tqdm


# ============================================================================
# AIS / NOAA PREPROCESSING (paper-aligned)
# Output: frame_id, vessel_id, LON, LAT, SOG, Heading
# ============================================================================

# Status removed: not available in 2017 NOAA AIS dataset
NOAA_REQUIRED_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading"]


def _valid_mmsi_9digits(series: pd.Series) -> pd.Series:
    s = series.astype("Int64").astype(str)
    return s.str.fullmatch(r"\d{9}")


def load_noaa_csv(csv_or_zip_path: str, inner_csv_name: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """
    Load NOAA AIS daily CSV from either a .csv or a .zip path.
    """
    if csv_or_zip_path.lower().endswith(".zip"):
        with zipfile.ZipFile(csv_or_zip_path) as z:
            if inner_csv_name is None:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise ValueError("Zip does not contain a .csv file.")
                inner_csv_name = names[0]
            with z.open(inner_csv_name) as f:
                return pd.read_csv(f, nrows=nrows)
    return pd.read_csv(csv_or_zip_path, nrows=nrows)


def clean_abnormal_data_noaa(
    df: pd.DataFrame,
    lat_range=(30.0, 35.0),
    lon_range=(-120.0, -115.0),
    sog_range=(1.0, 22.0),
    heading_range=(0.0, 360.0),
) -> pd.DataFrame:
    """
    Paper Step 1: Cleaning abnormal data.
    - 9-digit MMSI validation
    - Drop nulls in key fields
    - Filter by geographic/dynamic ranges (LAT, LON, SOG, Heading)

    NOTE: Status column excluded — not present in 2017 NOAA AIS data.
    """
    missing = [c for c in NOAA_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    initial_count = len(df)
    print(f"\n[Step 1/5] Cleaning abnormal data...")
    print(f"  Initial rows: {initial_count:,}")

    df = df.copy()
    df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"], errors="coerce", utc=True)

    # Drop nulls — Status excluded
    df = df.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading"])
    print(f"  After removing nulls: {len(df):,} rows")

    before = len(df)
    df = df[_valid_mmsi_9digits(df["MMSI"])]
    print(f"  After MMSI validation: {len(df):,} rows ({before - len(df):,} removed)")

    before = len(df)
    df = df[
        df["LAT"].between(lat_range[0], lat_range[1]) &
        df["LON"].between(lon_range[0], lon_range[1]) &
        df["SOG"].between(sog_range[0], sog_range[1]) &
        df["Heading"].between(heading_range[0], heading_range[1])
    ]
    print(f"  After range filtering: {len(df):,} rows ({before - len(df):,} removed)")
    print(f"  Final cleaned rows: {len(df):,} ({100*(1-len(df)/initial_count):.1f}% reduction)")

    return df


def resample_interpolate_1min(
    df: pd.DataFrame,
    freq: str = "1min",
    rolling_window: int = 5,
    max_gap_minutes: int = 10,
) -> pd.DataFrame:
    """
    Paper Step 2: Data interpolation (gap-aware).

    NOTE: Status column excluded — not present in 2017 NOAA AIS data.
    """
    initial_count = len(df)
    print(f"\n[Step 2/5] Data interpolation and resampling (max_gap={max_gap_minutes}min)...")
    print(f"  Input rows: {initial_count:,}")

    df = df.copy().sort_values(["MMSI", "BaseDateTime"])
    n_vessels = df["MMSI"].nunique()
    print(f"  Vessels to process: {n_vessels}")

    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    out_parts = []

    for mmsi, g in df.groupby("MMSI", sort=False):
        g = (
            g.drop_duplicates(subset=["BaseDateTime"], keep="last")
            .sort_values("BaseDateTime")
            .set_index("BaseDateTime")
        )
        if len(g) == 0:
            continue

        # Split into continuous segments at large gaps
        time_diffs = g.index.to_series().diff()
        gap_mask = time_diffs > max_gap
        segment_ids = gap_mask.cumsum()

        for _, seg in g.groupby(segment_ids):
            if len(seg) < 2:
                continue

            time_range = pd.date_range(
                start=seg.index.min(),
                end=seg.index.max(),
                freq=freq,
            )
            r = seg.reindex(time_range)
            r.index.name = "BaseDateTime"

            # LON/LAT: time-based linear interpolation
            r["LON"] = r["LON"].interpolate(method="time")
            r["LAT"] = r["LAT"].interpolate(method="time")

            # SOG/Heading: rolling average
            for c in ["SOG", "Heading"]:
                s = r[c]
                s = s.fillna(
                    s.rolling(window=rolling_window, min_periods=1, center=True).mean()
                )
                r[c] = s.ffill().bfill()

            # Status excluded
            r = r.dropna(subset=["LON", "LAT", "SOG", "Heading"])

            if len(r) == 0:
                continue

            r["MMSI"] = mmsi
            out_parts.append(r.reset_index())

    result = pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()
    print(f"  Output rows after resampling: {len(result):,}")
    print(f"  Vessels after resampling: {result['MMSI'].nunique()}")

    return result


def filter_timestamps_min_vessels(df: pd.DataFrame, min_vessels_per_timestamp: int = 3) -> pd.DataFrame:
    """
    Paper Step 1 (final): Keep only timestamps where >= min_vessels vessels exist.
    """
    initial_count = len(df)
    initial_timestamps = df["BaseDateTime"].nunique()

    print(f"\n[Step 3/5] Filtering timestamps by concurrent vessels...")
    print(f"  Initial timestamps: {initial_timestamps:,}")
    print(f"  Minimum vessels per timestamp: {min_vessels_per_timestamp}")

    df = df.copy()
    counts = df.groupby("BaseDateTime")["MMSI"].nunique()
    keep_times = counts[counts >= min_vessels_per_timestamp].index
    df = df[df["BaseDateTime"].isin(keep_times)]

    print(f"  Timestamps kept: {len(keep_times):,}")
    print(f"  Rows after filtering: {len(df):,} ({initial_count - len(df):,} removed)")
    print(f"  Vessels remaining: {df['MMSI'].nunique()}")

    return df


def zscore_normalize_global(df: pd.DataFrame, cols=("LON", "LAT", "SOG", "Heading"), stats: dict | None = None):
    """
    Paper Step 4: Z-score normalization using global dataset statistics.
    """
    print(f"\n[Step 4/5] Data standardization (z-score normalization)...")

    df = df.copy()
    if stats is None:
        stats = {}
        print(f"  Computing global statistics for {len(cols)} features:")
        for c in cols:
            mean = float(df[c].mean())
            std = float(df[c].std(ddof=1))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            stats[c] = {"mean": mean, "std": std}
            print(f"    {c}: μ={mean:.4f}, σ={std:.4f}")
    else:
        print(f"  Using provided statistics for normalization")

    for c in cols:
        df[c] = (df[c] - stats[c]["mean"]) / stats[c]["std"]

    print(f"  Normalization complete.")
    return df, stats


def to_frame_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Step 5: Convert to frame format.
    Output columns: frame_id, vessel_id, LON, LAT, SOG, Heading
    """
    print(f"\n[Step 5/5] Converting to frame format...")

    df = df.copy().sort_values("BaseDateTime")
    t0    = df["BaseDateTime"].min()
    t_end = df["BaseDateTime"].max()

    df["frame_id"] = ((df["BaseDateTime"] - t0).dt.total_seconds() / 60.0).round().astype(int)
    df = df.rename(columns={"MMSI": "vessel_id"})

    result = df[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]]

    print(f"  Time range: {t0} to {t_end}")
    print(f"  Total frames (minutes): {result['frame_id'].max() + 1}")
    print(f"  Total vessels: {result['vessel_id'].nunique()}")
    print(f"  Total data points: {len(result):,}")

    return result


def preprocess_noaa_to_frames(
    df_raw: pd.DataFrame,
    lat_range=(30.0, 35.0),
    lon_range=(-120.0, -115.0),
    sog_range=(1.0, 22.0),
    heading_range=(0.0, 360.0),
    min_vessels_per_timestamp: int = 3,
    max_gap_minutes: int = 10,
    do_zscore: bool = True,
    zscore_stats: dict | None = None,
):
    """
    Complete AIS preprocessing pipeline (paper-aligned).
    Steps 1-5 as described in the paper.
    """
    print("=" * 70)
    print("AIS DATA PREPROCESSING PIPELINE (Paper-Aligned)")
    print("=" * 70)
    print(f"Input data: {len(df_raw):,} rows, {df_raw['MMSI'].nunique()} vessels")

    df = clean_abnormal_data_noaa(df_raw, lat_range, lon_range, sog_range, heading_range)
    df = resample_interpolate_1min(df, freq="1min", rolling_window=5, max_gap_minutes=max_gap_minutes)
    df = filter_timestamps_min_vessels(df, min_vessels_per_timestamp=min_vessels_per_timestamp)

    stats = None
    if do_zscore:
        df, stats = zscore_normalize_global(df, cols=("LON", "LAT", "SOG", "Heading"), stats=zscore_stats)

    frames = to_frame_format(df)

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Output: {len(frames):,} data points across {frames['frame_id'].max()+1} frames")
    print(f"Vessels: {frames['vessel_id'].nunique()}")
    print("=" * 70 + "\n")

    return frames, stats


def save_frames_csv(frames_df: pd.DataFrame, out_csv_path: str):
    """Save frame-format CSV for TrajectoryDataset."""
    out_dir = os.path.dirname(out_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frames_df.to_csv(out_csv_path, index=False, header=True)


# ============================================================================
# ORIGINAL UTILITY FUNCTIONS
# ============================================================================

def anorm(p1, p2):
    norm = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if norm == 0:
        return 0
    return 1 / norm


def loc_pos(seq_):
    """
    Adds a positional index as an extra feature.
    Input:  seq_ shape (seq_len, N, F)
    Output: shape (seq_len, N, F+1)
    """
    seq_len   = seq_.shape[0]
    num_nodes = seq_.shape[1]
    pos_seq   = np.arange(1, seq_len + 1)[:, np.newaxis, np.newaxis]
    pos_seq   = pos_seq.repeat(num_nodes, axis=1)
    return np.concatenate((pos_seq, seq_), axis=-1)


def seq_to_graph(seq_, seq_rel, pos_enc=False):
    """
    Builds node feature tensor V from RELATIVE features (velocities).

    Args:
        seq_:    (N, 4, seq_len) — absolute positions
        seq_rel: (N, 4, seq_len) — velocities (differences between consecutive steps)
        pos_enc: if True, prepend positional index [1, 2, ..., seq_len]

    Returns:
        V: torch.FloatTensor of shape (seq_len, N, 5) if pos_enc else (seq_len, N, 4)
    """
    assert seq_rel.dim() == 3, f"Expected seq_rel (N, F, T), got {seq_rel.shape}"

    V = seq_rel.permute(2, 0, 1).contiguous()  # (seq_len, N, 4)

    if pos_enc:
        V_np = V.cpu().numpy()
        V_np = loc_pos(V_np)  # (seq_len, N, 5)
        return torch.from_numpy(V_np).float()

    return V.float()


def poly_fit(traj, traj_len, threshold):
    """
    Determines whether a trajectory is non-linear using a 2nd-order polynomial fit.
    Input: traj shape (C, traj_len), uses only first 2 channels (LON, LAT).
    """
    traj2 = traj[:2, :]
    t     = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj2[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj2[1, -traj_len:], 2, full=True)[1]
    return 1.0 if (res_x + res_y >= threshold) else 0.0


# ============================================================================
# TRAJECTORY DATASET — Scene-based sliding window
# Aligned with Sekhon & Fleming 2020 (data.py)
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Scene-based sliding window dataset aligned with Sekhon & Fleming 2020.

    For each day CSV:
      1. Get all unique timestamps (frames)
      2. Slide a window of seq_len=(obs_len+pred_len) frames
         with shift=obs_len (no overlap between windows)
      3. For each window:
         - Check timestamps are consecutive (max 1 min gap)
         - Find vessels present in ALL obs_len timestamps
         - Keep only vessels that are moving (LAT/LON diff > 1e-04)
         - Require >= 3 valid vessels (total_vessels > 3, i.e. >=4... 
           actually S&F uses total_vessels <= 3 to reject, so we need > 3)
      4. Build sequence tensors for all valid vessels

    This matches S&F data.py behavior:
      - shift = sequence_length (obs_len)
      - valid_vessels filter: moving + present in all obs timestamps
      - condition: total_vessels > 3
    """

    def __init__(
        self,
        data_dir,
        obs_len=5,
        pred_len=5,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim=",",
    ):
        super(TrajectoryDataset, self).__init__()

        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.seq_len  = obs_len + pred_len
        # S&F uses shift = sequence_length (obs_len) — no overlap
        self.shift    = obs_len
        self.max_peds_in_frame = 0

        all_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith('.csv')
        ])
        if not all_files:
            raise ValueError(f"No CSV files found in {data_dir}")

        num_peds_in_seq     = []
        seq_list            = []
        seq_list_rel        = []
        loss_mask_list      = []
        non_linear_ped_list = []

        for path in all_files:
            data    = pd.read_csv(path)
            data_np = data[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]].values.astype(np.float32)

            # All unique timestamps sorted
            timestamps = np.unique(data_np[:, 0])
            n_frames   = len(timestamps)

            vessel_ids_in_file = np.unique(data_np[:, 1])
            print(f"  {os.path.basename(path)}: {len(vessel_ids_in_file)} vessels, {n_frames} frames")

            j = 0
            while not (j + self.seq_len) > n_frames:
                # Current window timestamps
                frame_timestamps = timestamps[j:j + self.seq_len]

                # --- S&F condition_time: check consecutive (max 1 min gap) ---
                # timestamps are in minutes (frame_id), so diff should be 1
                diffs = np.diff(frame_timestamps)
                if np.any(diffs > 1):
                    j += self.shift
                    continue

                # Get all rows in this window
                mask  = np.isin(data_np[:, 0], frame_timestamps)
                frame = data_np[mask]

                # Obs timestamps only (for vessel filtering)
                obs_timestamps = frame_timestamps[:self.obs_len]
                obs_mask       = np.isin(frame[:, 0], obs_timestamps)
                obs_frame      = frame[obs_mask]

                total_vessels = len(np.unique(obs_frame[:, 1]))

                # --- S&F condition_vessels ---
                # valid_vessels: present in ALL obs timestamps AND moving
                valid_vessels = []
                for v in np.unique(obs_frame[:, 1]):
                    v_data = obs_frame[obs_frame[:, 1] == v]
                    # Must be present in all obs timesteps
                    if len(v_data) != self.obs_len:
                        continue
                    # Must be moving (LAT diff > 1e-04 OR LON diff > 1e-04)
                    lat_diff = np.abs(np.diff(v_data[:, 3])).max()  # LAT is col 3
                    lon_diff = np.abs(np.diff(v_data[:, 2])).max()  # LON is col 2
                    if lat_diff < 1e-04 and lon_diff < 1e-04:
                        continue
                    valid_vessels.append(v)

                # S&F rejects if valid_vessels < total_vessels OR total_vessels <= 3
                if len(valid_vessels) < total_vessels or total_vessels <= 3:
                    j += self.shift
                    continue

                # Build sequence for valid vessels
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(valid_vessels))

                curr_seq       = np.zeros((len(valid_vessels), 4, self.seq_len), dtype=np.float32)
                curr_seq_rel   = np.zeros((len(valid_vessels), 4, self.seq_len), dtype=np.float32)
                curr_loss_mask = np.zeros((len(valid_vessels), self.seq_len),    dtype=np.float32)

                num_peds_considered = 0
                _non_linear_ped     = []

                for v in valid_vessels:
                    v_data = frame[frame[:, 1] == v]

                    # Build full seq_len trajectory
                    # Map frame_id → position in window
                    traj = np.zeros((4, self.seq_len), dtype=np.float32)
                    mask_v = np.zeros(self.seq_len, dtype=np.float32)

                    for row in v_data:
                        t_idx = np.where(frame_timestamps == row[0])[0]
                        if len(t_idx) == 0:
                            continue
                        t_idx = t_idx[0]
                        traj[:, t_idx] = row[2:6]  # LON, LAT, SOG, Heading
                        mask_v[t_idx]  = 1.0

                    rel_traj = np.zeros_like(traj)
                    rel_traj[:, 1:] = traj[:, 1:] - traj[:, :-1]

                    curr_seq[num_peds_considered]       = traj
                    curr_seq_rel[num_peds_considered]   = rel_traj
                    curr_loss_mask[num_peds_considered] = mask_v

                    _non_linear_ped.append(poly_fit(traj, self.pred_len, threshold))
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped_list += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

                j += self.shift

        if not seq_list:
            raise ValueError(
                f"No valid sequences created from {data_dir}. "
                "Check preprocessing filters and obs_len/pred_len."
            )

        print(f"\nTotal sequences: {len(seq_list)}")
        self.num_seq = len(seq_list)

        seq_arr        = np.concatenate(seq_list,       axis=0)
        seq_rel_arr    = np.concatenate(seq_list_rel,   axis=0)
        loss_mask_arr  = np.concatenate(loss_mask_list, axis=0)
        non_linear_arr = np.asarray(non_linear_ped_list, dtype=np.float32)

        self.obs_traj      = torch.from_numpy(seq_arr[:, :, :self.obs_len]).float()
        self.pred_traj     = torch.from_numpy(seq_arr[:, :, self.obs_len:]).float()
        self.obs_traj_rel  = torch.from_numpy(seq_rel_arr[:, :, :self.obs_len]).float()
        self.pred_traj_rel = torch.from_numpy(seq_rel_arr[:, :, self.obs_len:]).float()
        self.loss_mask     = torch.from_numpy(loss_mask_arr).float()
        self.non_linear_ped = torch.from_numpy(non_linear_arr).float()

        cum_start_idx      = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(s, e) for s, e in zip(cum_start_idx, cum_start_idx[1:])]

        self.v_obs  = []
        self.v_pred = []
        print("Processing graph tensors...")

        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            v_ = seq_to_graph(self.obs_traj[start:end, :],  self.obs_traj_rel[start:end, :],  True)
            self.v_obs.append(v_.clone())
            v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        return [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.v_obs[index],
            self.v_pred[index],
        ]