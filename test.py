from __future__ import print_function
import numpy as np
import numpy.ma as ma
import json
import time
import sys
from datetime import datetime
import pathlib
import shutil
import yaml
from argparse import ArgumentParser
import os
from functools import partial
from sklearn import metrics
from tqdm import tqdm, trange
import torchvision.models as models

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import dataloader
from models.CBIT import CBIT
from models.CBIT_V3 import CBIT_V3
from models.SUNet18 import SUNet18
from models.MTBIT1 import MTBIT
from dataloader import Dataset
from augmentations import get_validation_augmentations, get_training_augmentations
from losses import choose_criterion3d, choose_criterion2d
from optim import set_optimizer, set_scheduler
from cp import pretrain_strategy
from torchvision import utils

import warnings

warnings.filterwarnings('ignore')


def make_numpy_grid(tensor_data, pad_value=0, padding=0):
    tensor_data = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
    vis = np.array(vis.cpu()).transpose((1, 2, 0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_args():
    parser = ArgumentParser(description="Hyperparameters", add_help=True)
    parser.add_argument('-c', '--config-name', type=str, help='YAML Config name', dest='CONFIG', default='config')
    parser.add_argument('-nw', '--num-workers', type=str, help='Number of workers', dest='num_workers', default=2)
    parser.add_argument('-v', '--verbose', type=bool, help='Verbose validation metrics', dest='verbose', default=False)
    return parser.parse_args()


# to calculate rmse
def metric_mse(inputs, targets, mask, exclude_zeros=False):
    if exclude_zeros:
        mask_ = mask.copy()
        indices_one = mask_ == 1
        indices_zero = mask_ == 0
        mask_[indices_one] = 0  # replacing 1s with 0s
        mask_[indices_zero] = 1  # replacing 0s with 1s
        inputs = ma.masked_array(inputs, mask=mask_)
        targets = ma.masked_array(targets, mask=mask_)
        loss = (inputs - targets) ** 2
        n_pixels = np.count_nonzero(targets)
        return np.sum(loss) / n_pixels
    else:
        loss = (inputs - targets) ** 2
        return np.mean(loss)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


args = get_args()

device = 'cuda'
cuda = True
num_GPU = 1
torch.cuda.set_device(1)
manual_seed = 18
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)

config_name = args.CONFIG
config_path = './config/' + config_name
default_dst_dir = "./results/"
out_file = default_dst_dir + config_name + '/'
os.makedirs(out_file, exist_ok=True)

# Load the configuration params of the experiment
full_config_path = config_path + ".yaml"
print(f"Loading experiment {full_config_path}")
with open(full_config_path, "r") as f:
    exp_config = yaml.load(f, Loader=yaml.SafeLoader)

print(f"Logs and/or checkpoints will be stored on {out_file}")
shutil.copyfile(full_config_path, out_file + 'config.yaml')
print("Config file correctly saved!")

stats_file = open(out_file + 'stats.txt', 'a', buffering=1)
print(' '.join(sys.argv), file=stats_file)
print(' '.join(sys.argv))

print(exp_config)
print(exp_config, file=stats_file)

x_train_dir = exp_config['data']['train']['path']
x_valid_dir = exp_config['data']['val']['path']
x_test_dir = exp_config['data']['test']['path']

batch_size = exp_config['data']['train']['batch_size']

lweight2d, lweight3d = exp_config['model']['loss_weights']
weights2d = exp_config['model']['2d_loss_weights']

augmentation = exp_config['data']['augmentations']
min_scale = exp_config['data']['min_value']
max_scale = exp_config['data']['max_value']

mean = exp_config['data']['mean']
std = exp_config['data']['std']

if augmentation:
    train_transform = get_training_augmentations(m=mean, s=std)
else:
    train_transform = get_validation_augmentations(m=mean, s=std)

valid_transform = get_validation_augmentations(m=mean, s=std)

train_dataset = Dataset(x_train_dir,
                        augmentation=train_transform)

valid_dataset = Dataset(x_valid_dir,
                        augmentation=valid_transform)

test_dataset = Dataset(x_test_dir,
                       augmentation=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

name_3dloss = exp_config['model']['3d_loss']
exclude_zeros = exp_config['model']['exclude_zeros']
criterion3d = choose_criterion3d(name=name_3dloss)

class_weights2d = torch.FloatTensor(weights2d).to(device)
name_2dloss = exp_config['model']['2d_loss']
criterion2d = choose_criterion2d(name_2dloss, class_weights2d)  # , class_ignored)

nepochs = exp_config['optim']['num_epochs']
lr = exp_config['optim']['lr']

model = exp_config['model']['model']
classes = exp_config['model']['num_classes']

pretrain = exp_config['model']['pretraining_strategy']
arch = exp_config['model']['feature_extractor_arch']
CHECKPOINTS = exp_config['model']['checkpoints_path']

encoder, pretrained, _ = pretrain_strategy(pretrain, CHECKPOINTS, arch)

if model == "SUNet18":
    net = SUNet18(3, 2, resnet=encoder).to(device)
elif model == 'mtbit_resnet18':
    net = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
                enc_depth=1, dec_depth=8, decoder_dim_head=16).to(device)
elif model == 'mtbit_resnet50':
    net = MTBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, if_upsample_2x=True,
                enc_depth=1, dec_depth=8, decoder_dim_head=16, backbone='resnet50').to(device)
elif model == 'CBIT':
    net = CBIT(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=5, if_upsample_2x=True,
               enc_depth=1, dec_depth=8, decoder_dim_head=16).to(device)
else:
    print('Model not implemented yet')

print('Model selected: ', model)

optimizer = set_optimizer(exp_config['optim'], net)
print('Optimizer selected: ', exp_config['optim']['optim_type'])
lr_adjust = set_scheduler(exp_config['optim'], optimizer)
print('Scheduler selected: ', exp_config['optim']['lr_schedule_type'])

res_cp = exp_config['model']['restore_checkpoints']
if os.path.exists(out_file + f'{res_cp}bestnet.pth'):
    net.load_state_dict(torch.load(out_file + f'{res_cp}bestnet.pth'))
    print('Checkpoints successfully loaded!')
else:
    print('No checkpoints founded')

tr_par, tot_par = count_parameters(net)
print(f'Trainable parameters: {tr_par}, total parameters {tot_par}')
print(f'Trainable parameters: {tr_par}, total parameters {tot_par}', file=stats_file)

net.eval()

TN = 0
FP = 0
FN = 0
TP = 0
mean_mae = 0
rmse1 = 0
rmse2 = 0
flag = 0

for t1, t2, mask2d, mask3d in tqdm(test_loader):

    t1 = t1.to(device)
    t2 = t2.to(device)

    out2d, out3d = net(t1, t2)

    out2d = out2d.detach().argmax(dim=1)

    out2d = out2d.cpu().numpy()  # （1，400，400）
    out3d = out3d.detach().cpu().numpy()
    out3d = (out3d + 1) * (max_scale - min_scale) / 2 + min_scale

    try:
        tn, fp, fn, tp = metrics.confusion_matrix(mask2d.ravel(), out2d.ravel()).ravel()
    except:
        tn, fp, fn, tp = [0, 0, 0, 0]
        print('Only 0 mask')

    mean_ae = metrics.mean_absolute_error(mask3d.ravel(), out3d.ravel())
    s_rmse1 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros=False)
    s_rmse2 = metric_mse(out3d.ravel(), mask3d.cpu().numpy().ravel(), mask2d.cpu().numpy().ravel(), exclude_zeros=True)
    max_error = metrics.max_error(mask3d.ravel(), out3d.ravel())
    mask_max = np.abs(out3d).max()
    mask_min = out3d.min()
    print(f'max: {mask_max}  min:{mask_min}')

    mean_mae += mean_ae
    rmse1 += s_rmse1
    rmse2 += s_rmse2
    TN += tn
    FP += fp
    FN += fn
    TP += tp

mean_mae = mean_mae / len(test_loader)
mean_f1 = 2 * TP / (2 * TP + FP + FN)
mIoU = TP / (TP + FN + FP)
RMSE1 = np.sqrt(rmse1 / len(test_loader))
RMSE2 = np.sqrt(rmse2 / len(test_loader))

print(
    f'Test metrics - 2D: F1 Score -> {mean_f1 * 100} %; mIoU -> {mIoU * 100} %; 3D: MAE -> {mean_mae} m; RMSE -> {RMSE1} m; cRMSE -> {RMSE2} m')
stats = dict(epoch='Test', MeanAbsoluteError=mean_mae, RMSE=RMSE1, cRMSE=RMSE2, F1Score=mean_f1 * 100, mIoU=mIoU * 100)
print(json.dumps(stats), file=stats_file)
