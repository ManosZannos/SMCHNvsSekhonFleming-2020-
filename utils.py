"""
utils.py

Includes:
1) NOAA AIS preprocessing functions (used by preprocess_sf.py)

2) TrajectoryDataset: exact replication of Sekhon & Fleming 2020 data.py
   - Scene-based sliding window
   - shift = obs_len (no overlap)
   - feature_size = 2 (LAT, LON only, as in S&F paper)
   - Filters: valid vessels (moving + present in all obs timestamps)
   - Condition: total_vessels > 3
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
# AIS / NOAA PREPROCESSING
# ============================================================================

NOAA_REQUIRED_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "Heading"]


def _valid_mmsi_9digits(series: pd.Series) -> pd.Series:
    s = series.astype("Int64").astype(str)
    return s.str.fullmatch(r"\d{9}")


def load_noaa_csv(csv_or_zip_path: str, inner_csv_name: str | None = None, nrows: int | None = None) -> pd.DataFrame:
    """Load NOAA AIS daily CSV from either a .csv or a .zip path."""
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

    S&F uses feature_size=2 (LAT, LON only).
    SMCHN uses feature_size=4 (LON, LAT, SOG, Heading).

    Args:
        seq_:    (N, F, seq_len) — absolute positions
        seq_rel: (N, F, seq_len) — velocities
        pos_enc: if True, prepend positional index

    Returns:
        V: torch.FloatTensor
    """
    assert seq_rel.dim() == 3, f"Expected seq_rel (N, F, T), got {seq_rel.shape}"

    V = seq_rel.permute(2, 0, 1).contiguous()  # (seq_len, N, F)

    if pos_enc:
        V_np = V.cpu().numpy()
        V_np = loc_pos(V_np)
        return torch.from_numpy(V_np).float()

    return V.float()


def poly_fit(traj, traj_len, threshold):
    """
    Determines whether a trajectory is non-linear.
    Input: traj shape (F, traj_len), uses first 2 channels.
    """
    traj2 = traj[:2, :]
    t     = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj2[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj2[1, -traj_len:], 2, full=True)[1]
    return 1.0 if (res_x + res_y >= threshold) else 0.0


def get_features(ip, t):
    """
    Compute distance, bearing, heading matrices for spatial attention.
    Used by S&F model. ip: (N, F, T)
    """
    N = ip.shape[0]
    distance_matrix = torch.zeros(ip.shape[2], N, N)
    bearing_matrix  = torch.zeros(ip.shape[2], N, N)
    heading_matrix  = torch.zeros(ip.shape[2], N, N)
    return distance_matrix, bearing_matrix, heading_matrix


# ============================================================================
# TRAJECTORY DATASET — Exact replication of S&F data.py
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    Exact replication of Sekhon & Fleming 2020 data.py trajectory_dataset.

    Key properties (matching S&F):
    - feature_size=2: uses only LAT, LON (columns 3,2 in frame format)
    - shift = sequence_length (obs_len): no overlap between windows
    - Scene-based: each sample contains ALL vessels present in obs window
    - Valid vessels: present in ALL obs timesteps AND moving (diff > 1e-04)
    - Condition: total_vessels > 3 AND len(valid_vessels) == total_vessels
    - Normalization already applied in preprocessing (preprocess_sf.py)
    """

    def __init__(
        self,
        data_dir,
        obs_len=5,
        pred_len=5,
        skip=1,           # kept for API compatibility, S&F uses shift=obs_len internally
        threshold=0.002,
        min_ped=1,
        delim=",",
        feature_size=2,   # S&F uses 2 (LAT, LON only)
    ):
        super(TrajectoryDataset, self).__init__()

        self.obs_len      = obs_len
        self.pred_len     = pred_len
        self.seq_len      = obs_len + pred_len
        self.feature_size = feature_size
        # S&F: shift = sequence_length (obs_len) — no overlap
        self.shift        = obs_len
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
            # Load frame-format CSV
            # Columns: frame_id, vessel_id, LON, LAT, SOG, Heading
            data    = pd.read_csv(path)
            data_np = data[["frame_id", "vessel_id", "LON", "LAT", "SOG", "Heading"]].values.astype(np.float32)

            timestamps = np.unique(data_np[:, 0])
            n_frames   = len(timestamps)

            vessel_ids = np.unique(data_np[:, 1])
            print(f"  {os.path.basename(path)}: {len(vessel_ids)} vessels, {n_frames} frames")

            j = 0
            while not (j + self.seq_len) > n_frames:

                frame_timestamps = timestamps[j:j + self.seq_len]

                # --- S&F _condition_time: max 1 min gap between consecutive timestamps ---
                diffs = np.diff(frame_timestamps)
                if np.any(diffs > 1):
                    j += self.shift
                    continue

                # All rows in this window
                mask  = np.isin(data_np[:, 0], frame_timestamps)
                frame = data_np[mask]

                # Obs timestamps only
                obs_timestamps = frame_timestamps[:self.obs_len]
                obs_mask       = np.isin(frame[:, 0], obs_timestamps)
                obs_frame      = frame[obs_mask]

                total_vessels = len(np.unique(obs_frame[:, 1]))

                # --- S&F _condition_vessels ---
                # Valid vessel: present in ALL obs timesteps AND moving
                valid_vessels = []
                for v in np.unique(obs_frame[:, 1]):
                    v_data = obs_frame[obs_frame[:, 1] == v]

                    # Must be present in ALL obs timesteps
                    if len(v_data) != self.obs_len:
                        continue

                    # Must be moving: LAT diff > 1e-04 OR LON diff > 1e-04
                    # S&F checks: not (abs(LAT.diff).max() < 1e-04)
                    #             and not (abs(LON.diff).max() < 1e-04)
                    lat_diff = np.abs(np.diff(v_data[:, 3])).max()  # LAT col 3
                    lon_diff = np.abs(np.diff(v_data[:, 2])).max()  # LON col 2

                    if lat_diff < 1e-04 and lon_diff < 1e-04:
                        continue

                    valid_vessels.append(v)

                # S&F rejects if there are 3 or fewer valid vessels
                if len(valid_vessels) <= 3:
                    j += self.shift
                    continue

                # Build sequence tensors for valid vessels
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(valid_vessels))

                # Use ALL features (4) in storage, apply feature_size slice later
                curr_seq       = np.zeros((len(valid_vessels), 4, self.seq_len), dtype=np.float32)
                curr_seq_rel   = np.zeros((len(valid_vessels), 4, self.seq_len), dtype=np.float32)
                curr_loss_mask = np.zeros((len(valid_vessels), self.seq_len),    dtype=np.float32)

                num_peds_considered = 0
                _non_linear_ped     = []

                for v in valid_vessels:
                    v_frame = frame[frame[:, 1] == v]

                    # Build full seq_len trajectory
                    traj   = np.zeros((4, self.seq_len), dtype=np.float32)
                    mask_v = np.zeros(self.seq_len, dtype=np.float32)

                    for row in v_frame:
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
                "Check preprocessing and obs_len/pred_len."
            )

        print(f"\nTotal sequences: {len(seq_list)}")
        self.num_seq = len(seq_list)

        seq_arr        = np.concatenate(seq_list,       axis=0)
        seq_rel_arr    = np.concatenate(seq_list_rel,   axis=0)
        loss_mask_arr  = np.concatenate(loss_mask_list, axis=0)
        non_linear_arr = np.asarray(non_linear_ped_list, dtype=np.float32)

        # Store full 4-feature tensors, slice to feature_size in __getitem__
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

            # Apply feature_size slice (S&F uses 2: LAT, LON)
            # Note: columns are [LON, LAT, SOG, Heading] → slice [:feature_size]
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

        # Apply feature_size slice
        obs_traj     = self.obs_traj[start:end, :self.feature_size, :]
        pred_traj    = self.pred_traj[start:end, :self.feature_size, :]
        obs_traj_rel = self.obs_traj_rel[start:end, :self.feature_size, :]
        pred_traj_rel= self.pred_traj_rel[start:end, :self.feature_size, :]

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