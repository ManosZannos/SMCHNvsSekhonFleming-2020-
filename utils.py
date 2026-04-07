"""
utils.py — Aligned with Sekhon & Fleming 2020 data.py + geographic_utils.py

Exact replications from S&F:
  - obs_len=8, pred_len=12  (S&F main.py defaults: sequence_length=8, prediction_length=12)
  - shift = obs_len (= sequence_length in S&F)
  - _condition_time: max gap <= 1 min across all consecutive timestamps
  - _condition_vessels:
      * checked on OBS timestamps only (first obs_len timestamps)
      * ALL vessels must be present in all obs steps (len == obs_len)
      * ALL vessels must move: not lat_static AND not lon_static
        (static = max(abs(diff)) < 1e-04)
      * if ANY vessel fails → reject entire scene
      * total_vessels > 3
  - get_sequence: shape (N, obs+pred, 4), fillarr for missing pred steps
  - Normalization bounds: geographic_utils.py ACTIVE lines [32,35]x[-120,-117]
  - split: random 80/10/10 at sample level (S&F load_data)
"""

import os
import math
import random
import zipfile

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, random_split
from tqdm import tqdm


# ============================================================================
# SEED — exact replication of S&F utils.py seed_everything(seed=100)
# Called at import time to match S&F reproducibility
# ============================================================================

def seed_everything(seed=100):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


# ============================================================================
# geographic_utils.py constants — ACTIVE lines
# min_lat, max_lat, min_lon, max_lon = 32, 35, -120, -117 → radians
# Must match preprocess_sf.py normalization bounds exactly
# ============================================================================
NORM_LAT_MIN_DEG = 32.0
NORM_LAT_MAX_DEG = 35.0
NORM_LON_MIN_DEG = -120.0
NORM_LON_MAX_DEG = -117.0

MIN_LAT = (math.pi / 180) * NORM_LAT_MIN_DEG
MAX_LAT = (math.pi / 180) * NORM_LAT_MAX_DEG
MIN_LON = (math.pi / 180) * NORM_LON_MIN_DEG
MAX_LON = (math.pi / 180) * NORM_LON_MAX_DEG


# ============================================================================
# AIS LOADING HELPER
# ============================================================================

def load_noaa_csv(csv_or_zip_path: str, inner_csv_name: str | None = None,
                  nrows: int | None = None) -> pd.DataFrame:
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def anorm(p1, p2):
    norm = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if norm == 0:
        return 0
    return 1 / norm


def loc_pos(seq_):
    """Prepend positional index. Input (seq_len, N, F) → (seq_len, N, F+1)."""
    seq_len   = seq_.shape[0]
    num_nodes = seq_.shape[1]
    pos_seq   = np.arange(1, seq_len + 1)[:, np.newaxis, np.newaxis]
    pos_seq   = pos_seq.repeat(num_nodes, axis=1)
    return np.concatenate((pos_seq, seq_), axis=-1)


def seq_to_graph(seq_, seq_rel, pos_enc=False):
    """
    Build node feature tensor V from relative features.
    Args:
        seq_:    (N, F, seq_len) absolute
        seq_rel: (N, F, seq_len) relative
        pos_enc: prepend positional index if True
    Returns:
        V: FloatTensor (seq_len, N, F) or (seq_len, N, F+1)
    """
    assert seq_rel.dim() == 3, f"Expected (N, F, T), got {seq_rel.shape}"
    V = seq_rel.permute(2, 0, 1).contiguous()
    if pos_enc:
        V_np = V.cpu().numpy()
        V_np = loc_pos(V_np)
        return torch.from_numpy(V_np).float()
    return V.float()


def poly_fit(traj, traj_len, threshold):
    """Non-linearity check. traj: (F, traj_len), uses first 2 channels."""
    traj2 = traj[:2, :]
    t     = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj2[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj2[1, -traj_len:], 2, full=True)[1]
    return 1.0 if (res_x + res_y >= threshold) else 0.0


def get_features(ip, t):
    """Distance, bearing, heading matrices. ip: (N, seq_len, F)"""
    N       = ip.shape[0]
    seq_len = ip.shape[1]
    distance_matrix = torch.zeros(seq_len, N, N)
    bearing_matrix  = torch.zeros(seq_len, N, N)
    heading_matrix  = torch.zeros(seq_len, N, N)
    return distance_matrix, bearing_matrix, heading_matrix


# ============================================================================
# FILLARR — exact replication of S&F data.py fillarr()
# Forward-fill then backward-fill zero values in trajectory array
# ============================================================================

def fillarr(arr):
    """
    S&F data.py fillarr(): forward-fill then backward-fill zeros.
    arr: (seq_len, n_features) numpy array.
    """
    for i in range(arr.shape[1]):
        idx = np.arange(arr.shape[0])
        idx[arr[:, i] == 0] = 0
        np.maximum.accumulate(idx, axis=0, out=idx)
        arr[:, i] = arr[idx, i]
        if (arr[:, i] == 0).any():
            idx[arr[:, i] == 0] = 0
            np.minimum.accumulate(idx[::-1], axis=0)[::-1]
            arr[:, i] = arr[idx, i]
    return arr


# ============================================================================
# TRAJECTORY DATASET — exact replication of S&F data.py trajectory_dataset
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Exact replication of S&F data.py trajectory_dataset.

    Default obs_len=8, pred_len=12 matches S&F main.py:
      --sequence_length 8 --prediction_length 12

    Matched behaviours:
      - shift = obs_len (S&F: self.shift = self.sequence_length)
      - _condition_time: max gap across consecutive timestamps <= 1 min
      - _condition_vessels:
          * frame trimmed to OBS timestamps only
          * total_vessels = all vessels in obs window
          * valid = present in ALL obs_len steps
                    AND not lat_static AND not lon_static
                    (static: max|diff| < 1e-04)
          * reject if ANY vessel invalid OR total_vessels <= 3
      - get_sequence: (N, obs+pred, 4), fillarr fills missing pred-window zeros
      - Normalization already applied in preprocess_sf.py
    """

    def __init__(
        self,
        data_dir,
        obs_len=8,          # S&F default: sequence_length=8
        pred_len=12,        # S&F default: prediction_length=12
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim=",",
        feature_size=2,     # S&F default: feature_size=2 (LAT, LON only)
    ):
        super(TrajectoryDataset, self).__init__()

        self.obs_len      = obs_len
        self.pred_len     = pred_len
        self.seq_len      = obs_len + pred_len   # = 20
        self.feature_size = feature_size
        self.shift        = obs_len              # S&F: self.shift = self.sequence_length
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

        # Debug counters
        _dbg_total   = 0
        _dbg_time    = 0
        _dbg_few     = 0
        _dbg_invalid = 0
        _dbg_minped  = 0
        _dbg_ok      = 0

        for path in all_files:
            df = pd.read_csv(path)
            df.sort_values('frame_id', inplace=True)

            # Pre-build pivot tables for fast vectorized window checks
            pivot_lat = df.pivot_table(
                index='frame_id', columns='vessel_id', values='LAT', aggfunc='first'
            )
            pivot_lon = df.pivot_table(
                index='frame_id', columns='vessel_id', values='LON', aggfunc='first'
            )

            timestamps = np.unique(df['frame_id'].values)
            n_frames   = len(timestamps)

            j = 0
            while not (j + self.seq_len) > n_frames:
                _dbg_total += 1

                frame_timestamps = timestamps[j:j + self.seq_len]

                # _condition_time: no gap > 1 min
                diff_ts = np.diff(frame_timestamps).astype('float')
                if np.amax(diff_ts) > 1:
                    _dbg_time += 1
                    j += self.shift
                    continue

                obs_timestamps = frame_timestamps[:self.obs_len]

                # Vectorized _condition_vessels via pivot tables
                obs_lat = pivot_lat.reindex(obs_timestamps)  # (obs_len, n_vessels)
                obs_lon = pivot_lon.reindex(obs_timestamps)

                # Present vessels: no NaN in any obs timestep
                present_mask    = obs_lat.notna().all(axis=0)
                present_vessels = obs_lat.columns[present_mask].tolist()
                n_total         = len(present_vessels)

                if n_total <= 3:
                    _dbg_few += 1
                    j += self.shift
                    continue

                # Movement check: max|diff| >= 1e-04 in BOTH LAT and LON
                lat_vals = obs_lat[present_vessels].values  # (obs_len, n_present)
                lon_vals = obs_lon[present_vessels].values

                lat_diff = np.abs(np.diff(lat_vals, axis=0)).max(axis=0)
                lon_diff = np.abs(np.diff(lon_vals, axis=0)).max(axis=0)

                moving   = (lat_diff >= 1e-04) & (lon_diff >= 1e-04)
                valid_vessels = [v for v, m in zip(present_vessels, moving) if m]

                # S&F: ALL vessels must be valid
                if len(valid_vessels) < n_total:
                    _dbg_invalid += 1
                    j += self.shift
                    continue

                # Fetch actual rows only for accepted windows
                mask_window = np.isin(df['frame_id'].values, frame_timestamps)
                frame_df    = df[mask_window]
                obs_df      = frame_df[np.isin(frame_df['frame_id'].values, obs_timestamps)]

                # ── get_sequence ──────────────────────────────────────────────
                self.max_peds_in_frame = max(self.max_peds_in_frame, n_total)

                # Store as (N, 4, seq_len) for SMCHN compatibility
                curr_seq       = np.zeros((n_total, 4, self.seq_len), dtype=np.float32)
                curr_seq_rel   = np.zeros((n_total, 4, self.seq_len), dtype=np.float32)
                curr_loss_mask = np.zeros((n_total, self.seq_len),    dtype=np.float32)
                _non_linear_ped = []

                for vi, v in enumerate(total_vessels):
                    v_data = frame_df[frame_df['vessel_id'] == v]

                    # Build (seq_len, 4): [LON, LAT, SOG, Heading]
                    traj   = np.zeros((self.seq_len, 4), dtype=np.float32)
                    mask_v = np.zeros(self.seq_len,      dtype=np.float32)

                    for _, row in v_data.iterrows():
                        t_idx = np.where(frame_timestamps == row['frame_id'])[0]
                        if len(t_idx) == 0:
                            continue
                        t_idx = t_idx[0]
                        traj[t_idx, :] = [row['LON'], row['LAT'],
                                          row['SOG'], row['Heading']]
                        mask_v[t_idx]  = 1.0

                    # S&F fillarr: fill zeros for missing pred-window entries
                    if (traj == 0).any():
                        traj = fillarr(traj)

                    traj_T          = traj.T                          # (4, seq_len)
                    rel_traj        = np.zeros_like(traj_T)
                    rel_traj[:, 1:] = traj_T[:, 1:] - traj_T[:, :-1]

                    curr_seq[vi]       = traj_T
                    curr_seq_rel[vi]   = rel_traj
                    curr_loss_mask[vi] = mask_v

                    _non_linear_ped.append(
                        poly_fit(traj_T, self.pred_len, threshold)
                    )

                if n_total > min_ped:
                    _dbg_ok += 1
                    non_linear_ped_list += _non_linear_ped
                    num_peds_in_seq.append(n_total)
                    loss_mask_list.append(curr_loss_mask)
                    seq_list.append(curr_seq)
                    seq_list_rel.append(curr_seq_rel)
                else:
                    _dbg_minped += 1

                j += self.shift

        if not seq_list:
            raise ValueError(
                f"No valid sequences found in {data_dir}. "
                f"Check preprocessing and obs_len={obs_len}/pred_len={pred_len}."
            )

        print(f"\n  Windows checked:      {_dbg_total}")
        print(f"  Rejected (time gap):  {_dbg_time}")
        print(f"  Rejected (<=3 vessels): {_dbg_few}")
        print(f"  Rejected (invalid vessels): {_dbg_invalid}")
        print(f"  Rejected (min_ped):   {_dbg_minped}")
        print(f"  Accepted:             {_dbg_ok}")
        print(f"\nTotal sequences: {len(seq_list)}")
        self.num_seq = len(seq_list)

        seq_arr        = np.concatenate(seq_list,       axis=0)
        seq_rel_arr    = np.concatenate(seq_list_rel,   axis=0)
        loss_mask_arr  = np.concatenate(loss_mask_list, axis=0)
        non_linear_arr = np.asarray(non_linear_ped_list, dtype=np.float32)

        self.obs_traj       = torch.from_numpy(seq_arr[:, :, :self.obs_len]).float()
        self.pred_traj      = torch.from_numpy(seq_arr[:, :, self.obs_len:]).float()
        self.obs_traj_rel   = torch.from_numpy(seq_rel_arr[:, :, :self.obs_len]).float()
        self.pred_traj_rel  = torch.from_numpy(seq_rel_arr[:, :, self.obs_len:]).float()
        self.loss_mask      = torch.from_numpy(loss_mask_arr).float()
        self.non_linear_ped = torch.from_numpy(non_linear_arr).float()

        cum_start_idx      = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (s, e) for s, e in zip(cum_start_idx, cum_start_idx[1:])
        ]

        self.v_obs  = []
        self.v_pred = []
        print("Processing graph tensors...")

        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]

            obs_fs  = self.obs_traj[start:end, :self.feature_size, :]
            obs_rel = self.obs_traj_rel[start:end, :self.feature_size, :]
            prd_fs  = self.pred_traj[start:end, :self.feature_size, :]
            prd_rel = self.pred_traj_rel[start:end, :self.feature_size, :]

            v_ = seq_to_graph(obs_fs, obs_rel, True)
            self.v_obs.append(v_.clone())
            v_ = seq_to_graph(prd_fs, prd_rel, False)
            self.v_pred.append(v_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        obs_traj      = self.obs_traj[start:end, :self.feature_size, :]
        pred_traj     = self.pred_traj[start:end, :self.feature_size, :]
        obs_traj_rel  = self.obs_traj_rel[start:end, :self.feature_size, :]
        pred_traj_rel = self.pred_traj_rel[start:end, :self.feature_size, :]

        return [
            obs_traj,
            pred_traj,
            obs_traj_rel,
            pred_traj_rel,
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
            self.v_obs[index],
            self.v_pred[index],
        ]


# ============================================================================
# LOAD DATA — Random 80/10/10 split matching S&F data.py load_data()
# ============================================================================

def load_data(data_dir, args):
    """
    Replicates S&F data.py load_data():
    - Builds TrajectoryDataset from processed/ CSVs
    - Random 80/10/10 split at sample level
    - Saves/loads splits as .pt files

    args expected attributes:
      obs_len, pred_len, feature_size (default 2), split_data (default False)
    """
    processed_dir = os.path.join(data_dir, "processed")
    train_dir     = os.path.join(data_dir, "train")
    val_dir       = os.path.join(data_dir, "val")
    test_dir      = os.path.join(data_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    train_pt = os.path.join(train_dir, f"{args.obs_len:02d}_{args.pred_len:02d}.pt")
    val_pt   = os.path.join(val_dir,   f"{args.obs_len:02d}_{args.pred_len:02d}.pt")
    test_pt  = os.path.join(test_dir,  f"{args.obs_len:02d}_{args.pred_len:02d}.pt")

    if getattr(args, 'split_data', False) or not os.path.exists(train_pt):
        print(f"Building dataset from {processed_dir}...")
        print(f"obs_len={args.obs_len}, pred_len={args.pred_len} "
              f"(S&F defaults: 8, 12)")

        data = TrajectoryDataset(
            processed_dir,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            feature_size=getattr(args, 'feature_size', 2),
        )

        data_size  = len(data)
        val_size   = int(np.floor(0.1 * data_size))
        test_size  = val_size
        train_size = data_size - val_size - test_size

        print(f"Total samples: {data_size}")
        print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")
        print(f"Target: ~8676")

        traindataset, validdataset, testdataset = random_split(
            data, [train_size, val_size, test_size]
        )

        torch.save(traindataset, train_pt)
        torch.save(validdataset, val_pt)
        torch.save(testdataset,  test_pt)
        print(f"Saved splits to {data_dir}")

    else:
        print(f"Loading existing splits from {data_dir}...")
        traindataset = torch.load(train_pt)
        validdataset = torch.load(val_pt)
        testdataset  = torch.load(test_pt)

    return traindataset, validdataset, testdataset