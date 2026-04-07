import math
import torch
import numpy as np

# ============================================================================
# S&F geographic constants (geographic_utils.py active lines)
# Used for inverse normalization and equirectangular distance in NM
# ============================================================================
_RADIUS_EARTH_NM = 3440.1  # nautical miles

_MIN_LAT = (math.pi / 180) * 32.0
_MAX_LAT = (math.pi / 180) * 35.0
_MIN_LON = (math.pi / 180) * -120.0
_MAX_LON = (math.pi / 180) * -117.0


def _scale_values(lat_norm, lon_norm):
    """
    Inverse of data.py normalize() → back to radians.
    Matches S&F geographic_utils.py scale_values().
    """
    lat = (_MAX_LAT - _MIN_LAT) * lat_norm + _MIN_LAT
    lon = (_MAX_LON - _MIN_LON) * lon_norm + _MIN_LON
    return lat, lon


def equirectangular_distance_nm(lat1_n, lon1_n, lat2_n, lon2_n):
    """
    Equirectangular distance in nautical miles.
    Exact replication of S&F geographic_utils.py equirectangular_distance().
    Inputs are normalized [0,1] LAT/LON tensors.
    """
    lat1, lon1 = _scale_values(lat1_n, lon1_n)
    lat2, lon2 = _scale_values(lat2_n, lon2_n)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    dist = dlat ** 2 + (dlon * torch.cos((lat1 + lat2) / 2)) ** 2
    return _RADIUS_EARTH_NM * torch.sqrt(dist + 1e-24)


# ============================================================================
# Original SMCHN metrics (normalized space, L2)
# ============================================================================

def ade(predAll, targetAll, count_):
    """ADE in normalized space. predAll: list of [pred_len, N, 2]"""
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred   = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += math.sqrt(
                    (pred[i, t, 0] - target[i, t, 0]) ** 2 +
                    (pred[i, t, 1] - target[i, t, 1]) ** 2
                )
        sum_all += sum_ / (N * T)
    return sum_all / All


def fde(predAll, targetAll, count_):
    """FDE in normalized space. predAll: list of [pred_len, N, 2]"""
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred   = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T - 1, T):
                sum_ += math.sqrt(
                    (pred[i, t, 0] - target[i, t, 0]) ** 2 +
                    (pred[i, t, 1] - target[i, t, 1]) ** 2
                )
        sum_all += sum_ / N
    return sum_all / All


# ============================================================================
# S&F-compatible metrics in nautical miles
# pred/target shape: (pred_len, N, 2), channel 0=LON_norm, 1=LAT_norm
# ============================================================================

def ade_nm(pred, target):
    """
    ADE in nautical miles — matches S&F mean_displacement_error().
    pred/target: (pred_len, N, 2) normalized, channel 0=LON, 1=LAT
    """
    dist = equirectangular_distance_nm(
        pred[:, :, 1], pred[:, :, 0],      # lat_norm, lon_norm
        target[:, :, 1], target[:, :, 0]
    )  # (pred_len, N)
    return dist.mean().item()


def fde_nm(pred, target):
    """
    FDE in nautical miles — matches S&F final_displacement_error().
    pred/target: (pred_len, N, 2) normalized, channel 0=LON, 1=LAT
    """
    dist = equirectangular_distance_nm(
        pred[-1:, :, 1], pred[-1:, :, 0],
        target[-1:, :, 1], target[-1:, :, 0]
    )  # (1, N)
    return dist.mean().item()


# ============================================================================
# Utility functions
# ============================================================================

def seq_to_nodes(seq_, max_nodes=88):
    seq_ = seq_.squeeze()
    seq_ = seq_[:, :2]
    seq_len   = seq_.shape[2]
    max_nodes = seq_.shape[0]
    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_[h]
    return V.squeeze()


def nodes_rel_to_nodes_abs(nodes, init_node):
    nodes     = nodes[:, :, :2]
    init_node = init_node[:, :2]
    nodes_    = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]
    return nodes_.squeeze()


def closer_to_zero(current, new_v):
    dec = min([(abs(current), current), (abs(new_v), new_v)])[1]
    return dec != current


# ============================================================================
# Bivariate Gaussian loss
# ============================================================================

def bivariate_loss(V_pred, V_trgt):
    normx = V_trgt[:, :, 0] - V_pred[:, :, 0]
    normy = V_trgt[:, :, 1] - V_pred[:, :, 1]

    sx   = torch.exp(V_pred[:, :, 2])
    sy   = torch.exp(V_pred[:, :, 3])
    corr = torch.tanh(V_pred[:, :, 4])

    sxsy = torch.clamp(sx * sy, min=1e-6)
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = torch.clamp(1 - corr ** 2, min=1e-6)

    result = torch.exp(-z / (2 * negRho))
    denom  = 2 * torch.pi * (sxsy * torch.sqrt(negRho))
    result = result / denom
    result = -torch.log(torch.clamp(result, min=1e-20))
    return torch.mean(result)


# ============================================================================
# Best-of-K Sampling Evaluation (Paper-Aligned)
# ============================================================================

def sample_bivariate_gaussian(V_pred, num_samples=20):
    """
    Sample trajectories from bivariate Gaussian.
    V_pred: [pred_len, N, 5] = [μx, μy, log(σx), log(σy), ρ]
    Returns: [num_samples, pred_len, N, 2]
    """
    pred_len, N, _ = V_pred.shape
    device = V_pred.device

    mux  = V_pred[:, :, 0]
    muy  = V_pred[:, :, 1]
    sx   = torch.exp(V_pred[:, :, 2])
    sy   = torch.exp(V_pred[:, :, 3])
    corr = torch.tanh(V_pred[:, :, 4])

    z = torch.randn(num_samples, pred_len, N, 2, device=device)

    samples    = torch.zeros(num_samples, pred_len, N, 2, device=device)
    sqrt_term  = torch.sqrt(torch.clamp(1 - corr ** 2, min=1e-6))

    samples[:, :, :, 0] = mux.unsqueeze(0) + sx.unsqueeze(0) * z[:, :, :, 0]
    samples[:, :, :, 1] = (
        muy.unsqueeze(0) +
        corr.unsqueeze(0) * sy.unsqueeze(0) * z[:, :, :, 0] +
        sy.unsqueeze(0) * sqrt_term.unsqueeze(0) * z[:, :, :, 1]
    )
    return samples


def evaluate_best_of_k(V_pred, V_target, num_samples=20):
    """
    Best-of-K evaluation — paper aligned.

    Returns dict:
        minADE_norm:  min ADE across K samples (normalized space)
        FDE_norm:     FDE of best sample (normalized space)
        minADE_nm:    min ADE in nautical miles (S&F comparable)
        FDE_nm:       FDE of best sample in nautical miles
        mean_ADE_norm: ADE of mean prediction (normalized, fair vs S&F)
        mean_FDE_norm: FDE of mean prediction (normalized)
        mean_ADE_nm:  ADE of mean prediction in nautical miles
        mean_FDE_nm:  FDE of mean prediction in nautical miles
        best_sample:  selected trajectory [pred_len, N, 2]
    """
    samples = sample_bivariate_gaussian(V_pred, num_samples)  # [K, pred_len, N, 2]
    K, pred_len, N, _ = samples.shape

    target_exp = V_target.unsqueeze(0)  # [1, pred_len, N, 2]

    # L2 displacement for all samples: [K, pred_len, N]
    displacements = torch.sqrt(
        (samples[:, :, :, 0] - target_exp[:, :, :, 0]) ** 2 +
        (samples[:, :, :, 1] - target_exp[:, :, :, 1]) ** 2
    )

    ade_per_sample = displacements.mean(dim=[1, 2])   # [K]
    min_ade_idx    = torch.argmin(ade_per_sample)
    best_sample    = samples[min_ade_idx]             # [pred_len, N, 2]

    min_ade_norm  = ade_per_sample[min_ade_idx].item()
    fde_norm      = displacements[min_ade_idx, -1, :].mean().item()

    # NM metrics for best sample
    min_ade_nm = ade_nm(best_sample, V_target)
    fde_nm_val = fde_nm(best_sample, V_target)

    # Mean prediction metrics
    mean_pred     = V_pred[:, :, :2]                  # [pred_len, N, 2]
    mean_disp     = torch.sqrt(
        (mean_pred[:, :, 0] - V_target[:, :, 0]) ** 2 +
        (mean_pred[:, :, 1] - V_target[:, :, 1]) ** 2
    )
    mean_ade_norm = mean_disp.mean().item()
    mean_fde_norm = mean_disp[-1].mean().item()
    mean_ade_nm   = ade_nm(mean_pred, V_target)
    mean_fde_nm   = fde_nm(mean_pred, V_target)

    return {
        'minADE_norm':   min_ade_norm,
        'FDE_norm':      fde_norm,
        'minADE_nm':     min_ade_nm,
        'FDE_nm':        fde_nm_val,
        'mean_ADE_norm': mean_ade_norm,
        'mean_FDE_norm': mean_fde_norm,
        'mean_ADE_nm':   mean_ade_nm,
        'mean_FDE_nm':   mean_fde_nm,
        'best_sample':   best_sample.detach(),
    }


def best_of_k_ade(V_pred, V_target, num_samples=20):
    """minADE with best-of-K sampling (normalized space)."""
    results = evaluate_best_of_k(V_pred, V_target, num_samples)
    return results['minADE_norm'], results['best_sample']


def best_of_k_fde(V_pred, V_target, num_samples=20):
    """FDE of best sample selected by ADE (normalized space)."""
    results = evaluate_best_of_k(V_pred, V_target, num_samples)
    return results['FDE_norm'], results['best_sample']