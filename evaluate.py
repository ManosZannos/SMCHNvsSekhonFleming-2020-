"""
Evaluation Script - Deterministic ADE/FDE (aligned with Sekhon & Fleming 2020)

Uses only the mean (μx, μy) of the predicted Gaussian distribution as the
deterministic prediction — no sampling. This allows direct comparison with
Sekhon & Fleming 2020 which uses deterministic predictions.

Metrics reported in:
  - Degrees (°)         — SMCHN paper units
  - Nautical miles (nm) — Sekhon & Fleming 2020 units (for direct comparison)

Conversion: 1° ≈ 60 nautical miles

Usage:
  python evaluate.py --dataset noaa_jan2017 \
                     --checkpoint checkpoints/SMCHN_5_5/noaa_jan2017/val_best.pth \
                     --split test
"""

import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TrajectoryModel
from utils import TrajectoryDataset

# 1 degree ≈ 60 nautical miles
NM_PER_DEG = 60.0


def setup_args():
    parser = argparse.ArgumentParser(description='Trajectory Prediction Evaluation')

    parser.add_argument('--dataset', type=str, default='noaa_jan2017',
                        help='Dataset name (folder under ./dataset/)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--obs_len', type=int, default=5,
                        help='Observation sequence length (5 min, S&F aligned)')
    parser.add_argument('--pred_len', type=int, default=5,
                        help='Prediction sequence length (5 min, S&F aligned)')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--num_gcn_layers', type=int, default=1)
    parser.add_argument('--embedding_dims', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)

    return parser.parse_args()


def denormalize_predictions(V_pred, global_stats):
    """
    Denormalize μx (LON) and μy (LAT) from z-score to real-world coordinates.
    """
    V_pred_denorm = V_pred.clone()

    lon_mean = global_stats['LON']['mean']
    lon_std  = global_stats['LON']['std']
    lat_mean = global_stats['LAT']['mean']
    lat_std  = global_stats['LAT']['std']

    V_pred_denorm[:, :, 0] = V_pred[:, :, 0] * lon_std + lon_mean  # μx (LON)
    V_pred_denorm[:, :, 1] = V_pred[:, :, 1] * lat_std + lat_mean  # μy (LAT)

    return V_pred_denorm


def denormalize_coordinates(V_target, global_stats):
    """
    Denormalize ground truth from z-score to real-world coordinates (degrees).
    """
    V_target_denorm = V_target.clone()

    V_target_denorm[:, :, 0] = (V_target[:, :, 0] * global_stats['LON']['std']
                                 + global_stats['LON']['mean'])
    V_target_denorm[:, :, 1] = (V_target[:, :, 1] * global_stats['LAT']['std']
                                 + global_stats['LAT']['mean'])

    return V_target_denorm


def evaluate_model(model, loader, device, global_stats):
    """
    Deterministic evaluation using only the mean (μx, μy) of the Gaussian.

    Aligned with Sekhon & Fleming 2020:
    - ADE: average Euclidean displacement over all prediction timesteps and vessels
    - FDE: Euclidean displacement at the final prediction timestep, averaged over vessels
    """
    model.eval()

    all_ade = []
    all_fde = []
    total_sequences = 0

    print(f"\nEvaluating (deterministic — Gaussian mean only)...")
    print(f"Total batches: {len(loader)}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):

            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
                non_linear_ped, loss_mask, V_obs, V_tr = batch

            T = V_obs.shape[1]
            N = V_obs.shape[2]

            identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
            identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
            identity = [identity_spatial, identity_temporal]

            # Forward pass → Gaussian parameters [pred_len, N, 5]
            V_pred = model(V_obs, identity)
            V_pred = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred

            # --- DETERMINISTIC: use only μx, μy (Gaussian mean) ---
            mu_vel = V_pred[:, :, :2]  # [pred_len, N, 2] predicted velocity mean

            # Convert velocities → absolute positions
            last_obs = obs_traj.squeeze(0)[:, :2, -1]          # [N, 2]
            mu_abs   = torch.cumsum(mu_vel, dim=0) + last_obs.unsqueeze(0)  # [pred_len, N, 2]

            # Ground truth absolute positions [pred_len, N, 2]
            V_target = pred_traj_gt.squeeze(0).permute(2, 0, 1)[:, :, :2]

            # Denormalize to real-world coordinates (degrees)
            V_pred_abs = V_pred.clone()
            V_pred_abs[:, :, :2] = mu_abs
            pred_denorm   = denormalize_predictions(V_pred_abs, global_stats)[:, :, :2]
            target_denorm = denormalize_coordinates(V_target, global_stats)

            # Euclidean displacement at each timestep: [pred_len, N]
            displacements = torch.sqrt(
                (pred_denorm[:, :, 0] - target_denorm[:, :, 0])**2 +
                (pred_denorm[:, :, 1] - target_denorm[:, :, 1])**2
            )

            # ADE: mean over all timesteps and vessels (S&F definition)
            ade = displacements.mean().item()

            # FDE: mean over vessels at final timestep only (S&F definition)
            fde = displacements[-1, :].mean().item()

            all_ade.append(ade)
            all_fde.append(fde)
            total_sequences += 1

    avg_ade = np.mean(all_ade)
    avg_fde = np.mean(all_fde)

    return {
        'ADE_deg': avg_ade,
        'FDE_deg': avg_fde,
        'ADE_nm':  avg_ade * NM_PER_DEG,
        'FDE_nm':  avg_fde * NM_PER_DEG,
        'total_sequences': total_sequences,
    }


def main():
    args   = setup_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    data_path = os.path.join('./dataset', args.dataset, args.split)
    print(f"\nLoading dataset from: {data_path}")

    dataset = TrajectoryDataset(
        data_path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=1
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset loaded: {len(dataset)} sequences")

    # Load global statistics
    global_stats_path = os.path.join('./dataset', args.dataset, 'global_stats.json')
    print(f"\nLoading global statistics from: {global_stats_path}")

    if not os.path.exists(global_stats_path):
        raise FileNotFoundError(
            f"Global statistics file not found: {global_stats_path}\n"
            f"Run preprocessing first: python preprocess_ais.py"
        )

    with open(global_stats_path, 'r') as f:
        global_stats = json.load(f)

    print(f"Global stats loaded:")
    for feat in ['LON', 'LAT', 'SOG', 'Heading']:
        if feat in global_stats:
            print(f"  {feat}: mean={global_stats[feat]['mean']:.4f}, "
                  f"std={global_stats[feat]['std']:.4f}")

    # Initialize model
    print(f"\nInitializing model...")
    model = TrajectoryModel(
        number_asymmetric_conv_layer=2,
        embedding_dims=args.embedding_dims,
        number_gcn_layers=args.num_gcn_layers,
        dropout=0.0,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        out_dims=5,
        num_heads=args.num_heads
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!")

    # Evaluate
    results = evaluate_model(model, loader, device, global_stats)

    # Print results
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EVALUATION RESULTS — SMCHN vs Sekhon & Fleming 2020")
    print(f"{sep}")
    print(f"Dataset:    {args.dataset} ({args.split} split)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sequences:  {results['total_sequences']}")
    print(f"Obs/Pred:   {args.obs_len} min / {args.pred_len} min")
    print(f"Method:     Deterministic (Gaussian mean only)")
    print(f"\n--- Metrics in DEGREES (°) ---")
    print(f"  ADE: {results['ADE_deg']:.6f}°")
    print(f"  FDE: {results['FDE_deg']:.6f}°")
    print(f"\n--- Metrics in NAUTICAL MILES (nm) — S&F 2020 units ---")
    print(f"  ADE: {results['ADE_nm']:.6f} nm")
    print(f"  FDE: {results['FDE_nm']:.6f} nm")
    print(f"\n--- Sekhon & Fleming 2020 (reference) ---")
    print(f"  ADE (LSTM+Spatial+Temporal): 0.03314 nm")
    print(f"  FDE (LSTM+Spatial+Temporal): 0.03840 nm")
    print(f"{sep}")

    # Save results
    output_dir   = os.path.dirname(args.checkpoint)
    results_file = os.path.join(
        output_dir, f'eval_results_{args.split}_deterministic.txt'
    )

    with open(results_file, 'w') as f:
        f.write(f"EVALUATION RESULTS — SMCHN vs Sekhon & Fleming 2020\n")
        f.write(f"{'='*70}\n")
        f.write(f"Dataset:    {args.dataset} ({args.split} split)\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Sequences:  {results['total_sequences']}\n")
        f.write(f"Obs/Pred:   {args.obs_len} min / {args.pred_len} min\n")
        f.write(f"Method:     Deterministic (Gaussian mean only)\n\n")
        f.write(f"--- Metrics in DEGREES ---\n")
        f.write(f"ADE: {results['ADE_deg']:.6f}°\n")
        f.write(f"FDE: {results['FDE_deg']:.6f}°\n\n")
        f.write(f"--- Metrics in NAUTICAL MILES (S&F 2020 units) ---\n")
        f.write(f"ADE: {results['ADE_nm']:.6f} nm\n")
        f.write(f"FDE: {results['FDE_nm']:.6f} nm\n\n")
        f.write(f"--- Sekhon & Fleming 2020 (reference) ---\n")
        f.write(f"ADE (LSTM+Spatial+Temporal): 0.03314 nm\n")
        f.write(f"FDE (LSTM+Spatial+Temporal): 0.03840 nm\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()