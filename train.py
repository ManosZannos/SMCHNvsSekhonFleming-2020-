import os
import sys
import time
import argparse
import pickle
import numpy as np

# Parse GPU selection before importing torch
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', default="0", type=str, help='GPU device number')
parser.add_argument('--obs_len', type=int, default=8,
                    help='Observation sequence length — S&F default: 8 min')
parser.add_argument('--pred_len', type=int, default=12,
                    help='Prediction sequence length — S&F default: 12 min')
parser.add_argument('--dataset', default='noaa_jan2017_sf',
                    help='Dataset name (folder under ./dataset/)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size (used for gradient accumulation)')
parser.add_argument('--num_epochs', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
parser.add_argument('--milestones', type=int, default=[0, 100],
                    help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='SMCHN_8_12',
                    help='personal tag for the model')
parser.add_argument('--feature_size', type=int, default=2,
                    help='Input feature size (2=LAT/LON only, S&F aligned)')
parser.add_argument('--split_data', action="store_true", default=False,
                    help='Force re-split of dataset')
parser.add_argument('--test_only', action="store_true", default=False,
                    help='Skip training, only run test evaluation')

# Parse early to set CUDA device before importing torch
args_early, _ = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args_early.gpu_num
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from metrics import *
from model import *
from utils import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

print("Training initiating....")
print(args)


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


def make_identity(T, N, device):
    identity_spatial  = torch.ones((T, N, N), device=device) * torch.eye(N, device=device)
    identity_temporal = torch.ones((N, T, T), device=device) * torch.eye(T, device=device)
    return [identity_spatial, identity_temporal]


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {
    'min_val_epoch': -1, 'min_val_loss': 9999999999999999,
    'min_train_epoch': -1, 'min_train_loss': 9999999999999999
}


def train(epoch, model, optimizer, checkpoint_dir, loader_train):
    global metrics, constant_metrics
    model.train()

    loss_batch  = 0
    batch_count = 0
    is_fst_loss = True
    loader_len  = len(loader_train)
    turn_point  = int(loader_len / args.batch_size) * args.batch_size + \
                  loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch

        T = V_obs.shape[1]
        N = V_obs.shape[2]
        identity = make_identity(T, N, device)

        V_pred = model(V_obs, identity)
        V_pred = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred

        V_target = V_tr.squeeze(0)

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_target)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss / args.batch_size
            is_fst_loss = True

            optimizer.zero_grad()
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / max(1, batch_count))

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')


def vald(epoch, model, checkpoint_dir, loader_val):
    global metrics, constant_metrics
    model.eval()

    loss_batch  = 0
    batch_count = 0
    is_fst_loss = True
    loader_len  = len(loader_val)
    turn_point  = int(loader_len / args.batch_size) * args.batch_size + \
                  loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
            loss_mask, V_obs, V_tr = batch

        with torch.no_grad():
            T = V_obs.shape[1]
            N = V_obs.shape[2]
            identity = make_identity(T, N, device)

            V_pred = model(V_obs, identity)
            V_pred = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred

            V_target = V_tr.squeeze(0)

            if batch_count % args.batch_size != 0 and cnt != turn_point:
                l = graph_loss(V_pred, V_target)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                loss = loss / args.batch_size
                is_fst_loss = True
                loss_batch += loss.item()
                print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    avg_loss = loss_batch / max(1, batch_count)
    metrics['val_loss'].append(avg_loss)
    print('VALD:', '\t Epoch:', epoch, '\t Loss:', avg_loss)

    if avg_loss < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = avg_loss
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')


def test(model, checkpoint_dir, loader_test, num_samples=20):
    """
    Test evaluation matching SMCHN paper:
    - Loads best validation checkpoint
    - Samples num_samples=20 from bivariate Gaussian (SMCHN inference)
    - Reports ADE and FDE (best of 20 samples, matching SMCHN paper)
    - Also reports mean prediction ADE/FDE for fair comparison with S&F
      (S&F is deterministic, SMCHN best-of-20 is inherently advantaged)
    """
    print("\n" + "="*50)
    print("TEST EVALUATION")
    print("="*50)

    checkpoint_path = checkpoint_dir + 'val_best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ade_outer, fde_outer = [], []
    ade_mean_outer, fde_mean_outer = [], []

    with torch.no_grad():
        for batch in loader_test:
            batch = [tensor.to(device) for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, \
                non_linear_ped, loss_mask, V_obs, V_tr = batch

            T = V_obs.shape[1]
            N = V_obs.shape[2]
            identity = make_identity(T, N, device)

            # Ground truth: (pred_len, N, 2)
            V_tr_sq = V_tr.squeeze(0)

            # ── Best-of-20 sampling (SMCHN paper method) ──────────────────
            ade_samples, fde_samples = [], []
            mean_pred = None

            for s in range(num_samples):
                V_pred = model(V_obs, identity)
                V_pred = V_pred.squeeze(0) if V_pred.dim() == 4 else V_pred

                # Sample from bivariate Gaussian
                V_pred_pos = sample_bivariate(V_pred)  # (pred_len, N, 2)

                if s == 0:
                    # Mean prediction for fair comparison with deterministic S&F
                    mean_pred = get_mean_prediction(V_pred)  # (pred_len, N, 2)

                ade_s = ade(V_pred_pos, V_tr_sq[:, :, :2])
                fde_s = fde(V_pred_pos, V_tr_sq[:, :, :2])
                ade_samples.append(ade_s)
                fde_samples.append(fde_s)

            # Best of 20 (SMCHN paper)
            ade_outer.append(min(ade_samples))
            fde_outer.append(fde_samples[np.argmin(ade_samples)])

            # Mean prediction (fair comparison with S&F)
            ade_m = ade(mean_pred, V_tr_sq[:, :, :2])
            fde_m = fde(mean_pred, V_tr_sq[:, :, :2])
            ade_mean_outer.append(ade_m)
            fde_mean_outer.append(fde_m)

    ade_best20 = sum(ade_outer) / len(ade_outer)
    fde_best20 = sum(fde_outer) / len(fde_outer)
    ade_mean   = sum(ade_mean_outer) / len(ade_mean_outer)
    fde_mean   = sum(fde_mean_outer) / len(fde_mean_outer)

    print(f"\nSMCHN Results (obs={args.obs_len}, pred={args.pred_len}):")
    print(f"  Best-of-20 (paper method):  ADE={ade_best20:.4f}  FDE={fde_best20:.4f}")
    print(f"  Mean prediction (vs S&F):   ADE={ade_mean:.4f}   FDE={fde_mean:.4f}")
    print(f"\nNote: Best-of-20 vs S&F deterministic is not directly comparable.")
    print(f"Use mean prediction metrics for fair comparison with S&F.")
    print("="*50)

    return ade_best20, fde_best20, ade_mean, fde_mean


def sample_bivariate(V_pred):
    """
    Sample (x,y) from bivariate Gaussian parameters output by SMCHN.
    V_pred: (pred_len, N, 5) — [mu_x, mu_y, sigma_x, sigma_y, rho]
    Returns: (pred_len, N, 2)
    """
    pred_len, N, _ = V_pred.shape
    mu_x    = V_pred[:, :, 0]
    mu_y    = V_pred[:, :, 1]
    sigma_x = torch.exp(V_pred[:, :, 2])
    sigma_y = torch.exp(V_pred[:, :, 3])
    rho     = torch.tanh(V_pred[:, :, 4])

    samples = torch.zeros(pred_len, N, 2, device=V_pred.device)
    for t in range(pred_len):
        for n in range(N):
            cov = torch.tensor([
                [sigma_x[t, n]**2, rho[t, n]*sigma_x[t, n]*sigma_y[t, n]],
                [rho[t, n]*sigma_x[t, n]*sigma_y[t, n], sigma_y[t, n]**2]
            ], device=V_pred.device)
            mean = torch.stack([mu_x[t, n], mu_y[t, n]])
            try:
                dist = torch.distributions.MultivariateNormal(mean, cov)
                samples[t, n] = dist.sample()
            except Exception:
                samples[t, n] = mean
    return samples


def get_mean_prediction(V_pred):
    """
    Return mean (mu_x, mu_y) from bivariate Gaussian — for fair comparison
    with deterministic S&F model.
    V_pred: (pred_len, N, 5)
    Returns: (pred_len, N, 2)
    """
    return V_pred[:, :, :2]


def ade(pred, target):
    """Average Displacement Error. pred/target: (pred_len, N, 2)"""
    error = torch.norm(pred - target, dim=-1)   # (pred_len, N)
    return error.mean().item()


def fde(pred, target):
    """Final Displacement Error. pred/target: (pred_len, N, 2)"""
    error = torch.norm(pred[-1] - target[-1], dim=-1)   # (N,)
    return error.mean().item()


def main(args):
    data_dir = os.path.join('./dataset', args.dataset)

    traindataset, validdataset, testdataset = load_data(data_dir, args)

    loader_train = DataLoader(traindataset, batch_size=1, shuffle=True,  num_workers=0)
    loader_val   = DataLoader(validdataset, batch_size=1, shuffle=False, num_workers=0)
    loader_test  = DataLoader(testdataset,  batch_size=1, shuffle=False, num_workers=0)

    print('Training started ...')
    print(f'Using device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'obs_len={args.obs_len} min, pred_len={args.pred_len} min (S&F aligned)')
    print(f'feature_size={args.feature_size}')

    writer = SummaryWriter(
        f"runs/{args.tag}_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}"
    )

    model = TrajectoryModel(
        number_asymmetric_conv_layer=2,
        embedding_dims=64,
        number_gcn_layers=1,
        dropout=0,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        out_dims=5,
        num_heads=4
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=0.1
        )

    checkpoint_dir = os.path.join('./checkpoints', args.tag, args.dataset) + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    if not args.test_only:
        for epoch in range(args.num_epochs):
            train(epoch, model, optimizer, checkpoint_dir, loader_train)
            vald(epoch, model, checkpoint_dir, loader_val)

            writer.add_scalar('trainloss', np.array(metrics['train_loss'])[epoch], epoch)
            writer.add_scalar('valloss',   np.array(metrics['val_loss'])[epoch],   epoch)

            if args.use_lrschd:
                scheduler.step()

            print('*' * 30)
            print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
            for k, v in metrics.items():
                if len(v) > 0:
                    print(k, v[-1])
            print(constant_metrics)
            print('*' * 30)

            with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
                pickle.dump(constant_metrics, fp)

    # Always run test evaluation at the end
    test(model, checkpoint_dir, loader_test, num_samples=20)

    writer.close()
    print('Done!')


if __name__ == '__main__':
    log_path = './Logs_train/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = (log_path + 'log-' +
                     time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log')
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)

    args = parser.parse_args()
    main(args)