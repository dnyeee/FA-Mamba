import os
import numpy as np
import random
from PIL import Image
from openpyxl import load_workbook, Workbook
from thop import profile
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
from scipy import io
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import time
import utils
from torch.nn import LayerNorm
from torch.fft import fft2, ifft2

import time
import math
from functools import partial
from typing import Callable
from block import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
    ):
        super().__init__()
        self.d_model = d_model  
        self.d_state = d_state  
        self.d_conv = d_conv  
        self.expand = expand  
        self.d_inner = int(self.expand * self.d_model)  
        self.dt_rank = math.ceil(self.d_model / 16)  

        self.visualize = True  
        self.visualize_counter = 0
        self.save_dir = "feature_maps"  
        os.makedirs(self.save_dir, exist_ok=True)  

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner, 
            out_channels=self.d_inner, 
            kernel_size=d_conv,  
            padding=(d_conv - 1) // 2,  
            bias=conv_bias,
            groups=self.d_inner,  
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )

        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) 
        self.forward_core = self.forward_corev0
        self.fft=Conditioning(128)
        

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) 
       
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
       
        xs = xs.float().view(B, -1, L)  
        dts = dts.contiguous().float().view(B, -1, L)  
        Bs = Bs.float().view(B, K, -1, L) 
        Cs = Cs.float().view(B, K, -1, L) 
        Ds = self.Ds.float().view(-1)  
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y=self.fft(y)
        y = self.out_norm(y)
        y = y * F.silu(z)  
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0.2,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path)
       

    def forward(self, input: torch.Tensor):
        x_norm = self.ln_1(input)
        x_perm = x_norm.permute(0, 2, 3, 1)
        attn_out = self.self_attention(x_perm)
        attn_out = attn_out.permute(0, 3, 1, 2)
        return input + self.drop_path(attn_out)


DataPath1 = 'Houston2013/HSI.mat'
DataPath2 = 'Houston2013/LiDAR.mat'
TRPath = 'Houston2013/TRLabel.mat'
TSPath = 'Houston2013/TSLabel.mat'
GTPath = 'Houston2013/gt.mat'

patchsize1 = 17
patchsize2 = 17
batchsize = 64
EPOCH = 10
LR = 0.001
NC = 20 


TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)
TrLabel = TrLabel['TRLabel']
TsLabel = TsLabel['TSLabel']
print('TrLabel',TrLabel.shape)
print('TsLabel',TsLabel.shape)

ground_truth = io.loadmat(GTPath)
ground_truth = ground_truth['gt']
print('ground_truth',ground_truth.shape)

Data = io.loadmat(DataPath1)
Data = Data['HSI']
Data = Data.astype(np.float32)
print('Data:HSI',Data.shape)

Data2 = io.loadmat(DataPath2)
Data2 = Data2['LiDAR']
Data2 = Data2.astype(np.float32)
print('Data2:LiDAR',Data2.shape)

[m, n, l] = Data.shape
for i in range(l):
    minimal = Data[:, :, i].min()
    maximal = Data[:, :, i].max()
    Data[:, :, i] = (Data[:, :, i] - minimal)/(maximal - minimal)

minimal = Data2.min()
maximal = Data2.max()
Data2 = (Data2 - minimal)/(maximal - minimal)


PC = np.reshape(Data, (m*n, l))
pca = PCA(n_components=NC, copy=True, whiten=False)
PC = pca.fit_transform(PC)
PC = np.reshape(PC, (m, n, NC))

temp = PC[:,:,0]
pad_width = np.floor(patchsize1/2)
pad_width = int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape
x = np.empty((m2,n2,NC),dtype='float32')

for i in range(NC):
    temp = PC[:,:,i]
    pad_width = np.floor(patchsize1/2)
    pad_width = int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x[:,:,i] = temp2

x2 = Data2
pad_width2 = np.floor(patchsize2/2)
pad_width2 = int(pad_width2)
temp2 = np.pad(x2, pad_width2, 'symmetric')
x2 = temp2

[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
TrainLabel = np.empty(TrainNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (NC, patchsize1, patchsize1))
    TrainPatch[i, :, :, :] = patch
    patchlabel = TrLabel[ind1[i], ind2[i]]
    TrainLabel[i] = patchlabel

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
TestLabel = np.empty(TestNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (NC, patchsize1, patchsize1))
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel

print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)

TsLabel_h = np.copy(TsLabel)

TsLabel_h[TsLabel_h == 0] = 1
[ind1, ind2] = np.where(TsLabel_h != 0)
TestNum_h = len(ind1)
TestPatch_h = np.empty((TestNum_h, NC, patchsize1, patchsize1), dtype='float32')
TestLabel_h = np.empty(TestNum_h)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (NC, patchsize1, patchsize1)) 
    TestPatch_h[i, :, :, :] = patch
    patchlabel_h = TsLabel[ind1[i], ind2[i]]
    TestLabel_h[i] = patchlabel_h

print(' testing size of HSI_drawing are:', TestPatch_h.shape)

[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
TrainLabel2 = np.empty(TrainNum)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TrainPatch2[i, :, :, :] = patch
    patchlabel2 = TrLabel[ind1[i], ind2[i]]
    TrainLabel2[i] = patchlabel2

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
TestLabel2 = np.empty(TestNum)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TestPatch2[i, :, :, :] = patch
    patchlabel2 = TsLabel[ind1[i], ind2[i]]
    TestLabel2[i] = patchlabel2

print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)

TsLabel_l = np.copy(TsLabel)

TsLabel_l[TsLabel_l == 0] = 1
[ind1, ind2] = np.where(TsLabel_l != 0)
TestNum= len(ind1)
TestPatch2_l = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
TestLabel2_l = np.empty(TestNum)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TestPatch2_l[i, :, :, :] = patch
    patchlabel2_l= TsLabel[ind1[i], ind2[i]]
    TestLabel2_l[i] = patchlabel2_l

print(' testing size of LiDAR are:' ,TestPatch2_l.shape)

TrainPatch1 = torch.from_numpy(TrainPatch)
TrainLabel1 = torch.from_numpy(TrainLabel)-1
TrainLabel1 = TrainLabel1.long()
TestPatch1 = torch.from_numpy(TestPatch)
TestLabel1 = torch.from_numpy(TestLabel)-1
TestLabel1 = TestLabel1.long()
TestPatch1_h = torch.from_numpy(TestPatch_h)
TestLabel1_h = torch.from_numpy(TestLabel_h)-1
TestLabel1_h = TestLabel1_h.long()
Classes = len(np.unique(TrainLabel))
TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainLabel2 = torch.from_numpy(TrainLabel2)-1
TrainLabel2 = TrainLabel2.long()
TestPatch2 = torch.from_numpy(TestPatch2)
TestLabel2 = torch.from_numpy(TestLabel2)-1
TestLabel2 = TestLabel1.long()
TestPatch2_l = torch.from_numpy(TestPatch2_l)
TestLabel2_l = torch.from_numpy(TestLabel2_l)-1
TestLabel2_l = TestLabel2_l.long()

dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)
trdraw_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=False)
dataset = dataf.TensorDataset(TestPatch1, TestPatch2, TestLabel2)
test_loader= dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual  
        return out


class Conditioning(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ff_parser_attn_map = nn.Conv2d(dim, 64, 1)
        self.norm_input = LayerNorm(64, elementwise_affine=True)
        self.norm_condition = LayerNorm(64, elementwise_affine=True)
        self.block = ResnetBlock(dim, dim)
        self.conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        dtype = x.dtype
        x = fft2(x)
        # x = x * w        
        x = ifft2(x).real
        x = x.type(dtype)
        return x
FM = 32    
class FA(nn.Module):
    def __init__(self):
        super(FA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = NC, 
                out_channels = FM,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM*2, FM*4, 3, 1, 1),
            nn.BatchNorm2d(FM*4),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Dropout(0.5),
        )
        self.out1 = nn.Sequential(
            nn.Linear(FM*4, Classes),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=FM,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(FM),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out2 = nn.Sequential(
            nn.Linear(FM*4, Classes),
        )
        self.out3 = nn.Sequential(
            nn.Linear(384, Classes),
        )
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1/3]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1/3]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1/3]))
        self.vss1 = VSSBlock(128, 0.2, norm_layer=nn.BatchNorm2d)
        self.vss3 = VSSBlock(128, 0.2, norm_layer=nn.BatchNorm2d)


    def forward(self, x1, x2, epoch=None):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.vss1(x1)
        x2 = self.conv4(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.vss1(x2)
        x_1=x1
        x1 = x1.view(x1.size(0), -1)
        x_2=x2
        x2 = x2.view(x2.size(0), -1)
        x = x_1 + x_2
        concatenated = torch.cat([x_1, x_2], axis=1)
        x=torch.cat([x,concatenated],axis=1)
        x = x.view(x.size(0), -1)
        out3 = self.out3(x)
       
        return  out3
model = FA()
print('The structure of the designed network', model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
model.to(device)  

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  

BestAcc = 0

torch.cuda.synchronize()
start = time.time()
val_acc = []
print("start training")
tic = time.time()
for epoch in range(EPOCH):

    model.train()
    train_acc, train_obj, tar_t, pre_t = utils.train_epoch(model, train_loader, loss_func , optimizer,device)
    OA1, AA1, Kappa1, CA1 = utils.output_metric(tar_t, pre_t)
    print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
          .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
    get_ts_result = False
    if (epoch % 10 == 0) | (epoch == EPOCH - 1):
        data = []
        model.eval()
        tar_v, pre_v = utils.valid_epoch(model, test_loader, loss_func,device ,get_ts_result)
        OA2, AA2, Kappa2, CA2 = utils.output_metric(tar_v, pre_v)

        val_acc.append(OA2)
        print("Every 10 epochs' records:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
        print(CA2)

        if OA2 > BestAcc:
            torch.save(model.state_dict(), 'models/houston_1.pkl')
            BestAcc = OA2









