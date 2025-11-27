#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NeuroAI Fast R-CNN with SNN Integration
Converted from Jupyter notebook to Python script
"""

import os
import time
import json
import math
import signal
import subprocess
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import snntorch as snn
from snntorch import surrogate
from torch.cuda.amp import GradScaler
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.ops import boxes as box_ops
from collections import defaultdict
from torchvision.models import resnet50
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import pandas as pd

print("torch", torch.__version__, "torchvision", torchvision.__version__, "snnTorch", snn.__version__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Paths / outputs
OUT_DIR = "/mnt/data"
os.makedirs(OUT_DIR, exist_ok=True)
PPT_PATH = "/mnt/data/neuro AI ppt.pdf"

# Experiment knobs (small for Colab)
SNN_TSTEPS = 4
SNN_ENCODING = 'latency'
SNN_DEPTH = 2
BATCH_SIZE = 4
NUM_WORKERS = 2
EPOCHS_BASELINE = 30
EPOCHS_HYBRID = 60


# ==================== Dataset ====================
class TinyDetDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, img_size=128, transforms=None):
        self.n = n
        self.img_size = img_size
        self.transforms = transforms or Compose([Resize((img_size, img_size)), ToTensor()])
        self.base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        img, _ = self.base[idx % len(self.base)]
        img = img.resize((self.img_size, self.img_size))
        w, h = self.img_size, self.img_size
        x1 = np.random.randint(0, w//2)
        y1 = np.random.randint(0, h//2)
        x2 = np.random.randint(w//2, w)
        y2 = np.random.randint(h//2, h)
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        img_t = self.transforms(img)
        target = {"boxes": boxes, "labels": labels}
        return img_t, target


train_ds = TinyDetDataset(n=400, img_size=128)
val_ds = TinyDetDataset(n=100, img_size=128)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                        collate_fn=collate_fn, num_workers=NUM_WORKERS)
print("TinyDet ready:", len(train_ds), "train,", len(val_ds), "val")


# ==================== Evaluation ====================
def evaluate_mean_iou(model, loader, device):
    model.eval()
    total_iou = 0.0
    count = 0
    total_dets = 0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval"):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                boxes_pred = out.get("boxes", torch.empty((0, 4))).cpu()
                boxes_gt = tgt["boxes"].cpu()
                total_dets += boxes_pred.shape[0]
                if boxes_pred.shape[0] > 0 and boxes_gt.shape[0] > 0:
                    ious = box_ops.box_iou(boxes_pred, boxes_gt)
                    max_iou_per_gt, _ = ious.max(dim=0)
                    total_iou += max_iou_per_gt.sum().item()
                    count += boxes_gt.shape[0]
    mean_iou = (total_iou / count) if count > 0 else 0.0
    avg_dets = total_dets / len(loader.dataset)
    return {"mean_iou": mean_iou, "avg_dets_per_image": avg_dets}


# ==================== SNN Front ====================
class SNNFrontDet(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, tsteps=4, beta=0.95, encode='latency'):
        super().__init__()
        self.t = int(tsteps)
        self.beta = float(beta)
        self.encode = encode
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def poisson_encode(self, x):
        T = self.t
        rnd = torch.rand((T,) + x.shape, device=x.device)
        xr = x.unsqueeze(0).expand(T, -1, -1, -1, -1)
        return (rnd < xr).float()

    def latency_encode(self, x):
        T = self.t
        B, C, H, W = x.shape
        times = torch.clamp(((1.0 - x) * (T - 1)).to(torch.long), 0, T-1)
        times_flat = times.view(B, -1)
        N = C * H * W
        pos_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        seq = torch.zeros((T, B, C, H, W), dtype=torch.float32, device=x.device)
        t_idx = times_flat
        b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N)
        ch_idx = (pos_idx // (H * W)) % C
        rem = pos_idx % (H * W)
        h_idx = rem // W
        w_idx = rem % W
        seq[t_idx, b_idx, ch_idx, h_idx, w_idx] = 1.0
        return seq

    def forward(self, x_raw):
        if self.encode == 'poisson':
            seq = self.poisson_encode(x_raw)
        else:
            seq = self.latency_encode(x_raw)
        T, B, C, H, W = seq.shape
        spike_acc = torch.zeros((B, self.conv2.out_channels, H//2, W//2), device=x_raw.device)
        total_spikes = torch.zeros(B, device=x_raw.device)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(T):
            xt = seq[t]
            z1 = self.conv1(xt)
            z1 = self.bn1(z1)
            sp1, mem1 = self.lif1(z1, mem1)
            z2 = self.conv2(sp1)
            z2 = self.bn2(z2)
            sp2, mem2 = self.lif2(z2, mem2)
            spike_acc += sp2
            total_spikes += sp2.view(B, -1).sum(dim=1)
        feat = spike_acc / float(T)
        avg_spikes = total_spikes / float(T)
        return feat, avg_spikes


# ==================== Backbone ====================
class ResNet50_Backbone_SNN(nn.Module):
    def __init__(self, snn_tsteps=4, pretrained_backbone=True, encode='latency'):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        self.resnet = resnet50(weights=weights)
        self.resnet.conv1 = nn.Identity()
        self.bn1 = self.resnet.bn1
        self.relu = self.resnet.relu
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
        self.snn_front = SNNFrontDet(in_ch=3, out_ch=64, tsteps=snn_tsteps, encode=encode)
        in_channels_list = [512, 1024, 2048]
        out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
        self._out_channels = out_channels

    def forward(self, x_norm):
        x_raw = x_norm
        feat0, _ = self.snn_front(x_raw)
        x = self.bn1(feat0)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        features = {"0": c2, "1": c3, "2": c4}
        fpn_out = self.fpn(features)
        return fpn_out

    @property
    def out_channels(self):
        return self._out_channels


# ==================== Create Models ====================
print("Creating baseline model...")
baseline_model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = baseline_model.roi_heads.box_predictor.cls_score.in_features
baseline_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
baseline_model.to(device)

print("Creating hybrid model...")
backbone = ResNet50_Backbone_SNN(snn_tsteps=SNN_TSTEPS, pretrained_backbone=True, encode=SNN_ENCODING)

anchor_sizes = ((32, 64, 128), (128, 256, 512), (512, 1024, 2048))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

roi_pooler = MultiScaleRoIAlign(featmap_names=['0', '1', '2'], output_size=7, sampling_ratio=2)

hybrid_model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
hybrid_model.to(device)

print("Baseline and Hybrid models instantiated.")


# ==================== Training Functions ====================
scaler_base = GradScaler()

def train_one_epoch_det(model, optimizer, loader, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc=f"Train E{epoch}"):
        imgs = [img.to(device) for img in imgs]
        targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss_dict = model(imgs, targs)
                loss = sum(loss_dict.values())
        else:
            loss_dict = model(imgs, targs)
            loss = sum(loss_dict.values())
        scaler_base.scale(loss).backward()
        scaler_base.step(optimizer)
        scaler_base.update()
        running_loss += loss.item()
    return running_loss / len(loader)


# ==================== Train Baseline ====================
print("\n=== Training Baseline (1 epoch quick) ===")
opt_b = optim.SGD(baseline_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
train_one_epoch_det(baseline_model, opt_b, train_loader, device, epoch=30)
b_metrics = evaluate_mean_iou(baseline_model, val_loader, device)
print("Baseline metrics (proxy):", b_metrics)
torch.save({'model_state': baseline_model.state_dict()}, "baseline_quick.pth")
open('det_exp_results.json', 'w').write(json.dumps({'baseline_metrics': b_metrics}, indent=2))


# ==================== Train Hybrid ====================
print("\n=== Training Hybrid ===")
scaler_h = GradScaler()
opt_h = optim.SGD(hybrid_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

def train_one_epoch_surrogate(model, optimizer, loader, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc=f"Surrogate Train E{epoch}"):
        imgs = [img.to(device) for img in imgs]
        targs = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss_dict = model(imgs, targs)
                loss = sum(loss_dict.values())
        else:
            loss_dict = model(imgs, targs)
            loss = sum(loss_dict.values())
        scaler_h.scale(loss).backward()
        scaler_h.step(optimizer)
        scaler_h.update()
        running_loss += loss.item()
    return running_loss / len(loader)

best = {'epoch': 0, 'metric': 0.0}
for e in range(1, EPOCHS_HYBRID + 1):
    tr_loss = train_one_epoch_surrogate(hybrid_model, opt_h, train_loader, device, e)
    h_metrics = evaluate_mean_iou(hybrid_model, val_loader, device)
    print(f"Epoch {e} loss {tr_loss:.4f} eval {h_metrics}")
    torch.save({'epoch': e, 'model_state': hybrid_model.state_dict(), 'opt': opt_h.state_dict()}, 
               f"hybrid_epoch{e}.pth")
    try:
        prev = json.load(open('det_exp_results.json'))
    except:
        prev = {}
    prev['hybrid_metrics'] = h_metrics
    open('det_exp_results.json', 'w').write(json.dumps(prev, indent=2))
    if h_metrics['mean_iou'] > best['metric']:
        best = {'epoch': e, 'metric': h_metrics['mean_iou']}

print("Hybrid best:", best)


# ==================== Measurement Function ====================
def measure_model_inference(model, dataset, n_images=100, sample_interval=0.2, power_log='power.csv'):
    cmd = f"nvidia-smi --query-gpu=power.draw --format=csv -l {sample_interval}"
    try:
        logger = subprocess.Popen(cmd.split(), stdout=open(power_log, 'w'), 
                                 stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        time.sleep(0.5)
    except Exception as e:
        logger = None
    
    if hasattr(model, "eval"):
        model.eval()
    
    count = 0
    t0 = time.time()
    spikes_total = 0.0
    
    with torch.no_grad():
        for img, _ in dataset:
            imgs = [img.to(device)]
            _ = model(imgs)
            try:
                x = img.unsqueeze(0).to(device)
                feat, avg_spikes = model.backbone.snn_front(x)
                spikes_total += avg_spikes.mean().item()
            except Exception:
                pass
            count += 1
            if count >= n_images:
                break
    
    t1 = time.time()
    total_time = t1 - t0
    
    powers = []
    if logger is not None:
        try:
            os.killpg(os.getpgid(logger.pid), signal.SIGTERM)
        except Exception:
            try:
                logger.terminate()
            except:
                pass
        time.sleep(0.1)
        try:
            with open(power_log, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if nums:
                        try:
                            powers.append(float(nums[-1]))
                        except:
                            pass
        except Exception:
            powers = []
    
    avg_power = float(np.mean(powers)) if len(powers) > 0 else float('nan')
    energy_total = avg_power * total_time if not math.isnan(avg_power) else float('nan')
    energy_per_image = energy_total / count if count > 0 and not math.isnan(energy_total) else float('nan')
    throughput = count / total_time if total_time > 0 else float('nan')
    latency_ms = 1000.0 * total_time / count if count > 0 else float('nan')
    avg_spikes_per_img = spikes_total / count if count > 0 else float('nan')
    
    summary = {
        'n_images': count,
        'total_time_s': total_time,
        'throughput_img_s': throughput,
        'latency_ms': latency_ms,
        'avg_power_w': avg_power,
        'energy_per_image_j': energy_per_image,
        'avg_spikes_per_img': avg_spikes_per_img,
        'raw_power_samples': len(powers)
    }
    return summary


# ==================== Measure Performance ====================
print("\n=== Measuring baseline ===")
baseline_meas = measure_model_inference(baseline_model, val_ds, n_images=50, 
                                       sample_interval=0.2, power_log='p_base.csv')
print("Baseline measurement:", baseline_meas)

print("\n=== Measuring hybrid ===")
hybrid_meas = measure_model_inference(hybrid_model, val_ds, n_images=50, 
                                     sample_interval=0.2, power_log='p_hybrid.csv')
print("Hybrid measurement:", hybrid_meas)

res_comp = {'baseline': baseline_meas, 'hybrid': hybrid_meas}
try:
    det_prev = json.load(open('det_exp_results.json'))
except:
    det_prev = {}
det_prev.update({
    'baseline_metrics': b_metrics if 'b_metrics' in globals() else det_prev.get('baseline_metrics'),
    'hybrid_metrics': h_metrics if 'h_metrics' in globals() else det_prev.get('hybrid_metrics')
})
open('det_exp_results.json', 'w').write(json.dumps(det_prev, indent=2))
open('comparison_energy.json', 'w').write(json.dumps(res_comp, indent=2))
print("Saved det_exp_results.json and comparison_energy.json")


# ==================== Plotting ====================
print("\n=== Creating plots ===")
det_res = {}
eng_res = {}
if os.path.exists('det_exp_results.json'):
    try:
        det_res = json.load(open('det_exp_results.json'))
    except:
        det_res = {}
if os.path.exists('comparison_energy.json'):
    try:
        eng_res = json.load(open('comparison_energy.json'))
    except:
        eng_res = {}

acc_base = det_res.get('baseline_metrics', {}).get('mean_iou', float('nan'))
acc_hyb = det_res.get('hybrid_metrics', {}).get('mean_iou', float('nan'))
e_base = eng_res.get('baseline', {}).get('energy_per_image_j', float('nan'))
e_hyb = eng_res.get('hybrid', {}).get('energy_per_image_j', float('nan'))

df = pd.DataFrame([
    {'model': 'baseline', 'accuracy': acc_base, 'energy_j': e_base},
    {'model': 'hybrid', 'accuracy': acc_hyb, 'energy_j': e_hyb}
]).set_index('model')

def pct(a, b):
    try:
        if math.isnan(a) or math.isnan(b):
            return None
        return 100.0 * (b - a) / a if a != 0 else None
    except:
        return None

acc_pct = pct(acc_base, acc_hyb)
eng_pct = pct(e_base, e_hyb)

print("Baseline mean_iou:", acc_base, "Hybrid mean_iou:", acc_hyb, "Change %:", acc_pct)
print("Baseline energy J/img:", e_base, "Hybrid energy J/img:", e_hyb, "Change %:", eng_pct)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(df.index, df['accuracy'].values)
axes[0].set_title("Mean IoU (proxy)")
axes[0].set_ylabel("mean_iou")
axes[1].bar(df.index, df['energy_j'].values)
axes[1].set_title("Energy per image (J)")
axes[1].set_ylabel("J/image")

for i, col in enumerate(['accuracy', 'energy_j']):
    for j, v in enumerate(df[col].values):
        lab = f"{v:.3f}" if not (isinstance(v, float) and math.isnan(v)) else "n/a"
        axes[i].text(j, (v if not math.isnan(v) else 0.01), lab, ha='center')

plt.tight_layout()
png_out = os.path.join(OUT_DIR, "acc_energy_comparison.png")
try:
    fig.savefig(png_out, dpi=150, bbox_inches='tight')
    print("Saved comparison plot to", png_out)
except Exception as e:
    print("Warning: failed to save PNG:", e)
plt.show()

csv_out = os.path.join(OUT_DIR, "acc_energy_summary.csv")
df.reset_index().to_csv(csv_out, index=False)
print("Saved CSV summary:", csv_out)
print("PPT path for slide reference:", PPT_PATH)

print("\n=== Script completed ===")