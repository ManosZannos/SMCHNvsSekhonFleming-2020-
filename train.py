import os
import sys
import time
import argparse
import pickle
import numpy as np

# Parse GPU selection before importing torch
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', default="0", type=str, help='GPU device number')
parser.add_argument('--obs_len', type=int, default=5,
                    help='Observation sequence length (5 min, S&F aligned)')
parser.add_argument('--pred_len', type=int, default=5,
                    help='Prediction sequence length (5 min, S&F aligned)')
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
parser.add_argument('--tag', default='SMCHN_5_5',
                    help='personal tag for the model')
parser.add_argument('--feature_size', type=int, default=2,
                    help='Input feature size (2=LAT/LON only, S&F aligned)')
parser.add_argument('--split_data', action="store_true", default=False,
                    help='Force re-split of dataset')

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


def main(args):
    data_dir = os.path.join('./dataset', args.dataset)

    # Load data with random 80/10/10 split (S&F aligned)
    traindataset, validdataset, testdataset = load_data(data_dir, args)

    loader_train = DataLoader(traindataset, batch_size=1, shuffle=True,  num_workers=0)
    loader_val   = DataLoader(validdataset, batch_size=1, shuffle=False, num_workers=0)

    print('Training started ...')
    print(f'Using device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'obs_len={args.obs_len} min, pred_len={args.pred_len} min (S&F aligned)')
    print(f'feature_size={args.feature_size} ({"LAT/LON only" if args.feature_size==2 else "LAT/LON/SOG/Heading"})')

    writer = SummaryWriter(f"runs/{args.tag}_{args.dataset}_{time.strftime('%Y%m%d-%H%M%S')}")

    # SMCHN model: input features = feature_size + 1 (pos_enc)
    # SparseWeightedAdjacency expects spa_in_dims = feature_size (after slicing pos_enc)
    # tem_in_dims = feature_size + 1 (with pos_enc)
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
            optimizer, milestones=[0, 100], gamma=0.1
        )

    checkpoint_dir = os.path.join('./checkpoints', args.tag, args.dataset) + '/'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

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

    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    log_path = './Logs_train/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file_name = log_path + 'log-' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.log'
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)

    args = parser.parse_args()
    main(args)