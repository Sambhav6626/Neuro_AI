
!pip install --quiet snntorch timm matplotlib pandas tqdm


import os, time, json, math
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import snntorch as snn
from snntorch import surrogate
from torch.cuda.amp import GradScaler
print("torch", torch.__version__, "torchvision", torchvision.__version__, "snnTorch", snn.__version__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Paths / outputs
OUT_DIR = "/mnt/data"
os.makedirs(OUT_DIR, exist_ok=True)
PPT_PATH = "/mnt/data/neuro AI ppt.pdf" 


SNN_TSTEPS = 4       
SNN_ENCODING = 'latency'  
SNN_DEPTH = 2        
BATCH_SIZE = 4
NUM_WORKERS = 2
EPOCHS_BASELINE = 30   
EPOCHS_HYBRID = 60  


from PIL import Image
import numpy as np, torch
from torchvision.transforms import Compose, Resize, ToTensor

class TinyDetDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, img_size=128, transforms=None):
        self.n = n
        self.img_size = img_size
        self.transforms = transforms or Compose([Resize((img_size,img_size)), ToTensor()])
        
        self.base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img, _ = self.base[idx % len(self.base)]
        img = img.resize((self.img_size, self.img_size))
        # one random box per image
        w,h = self.img_size, self.img_size
        x1 = np.random.randint(0, w//2)
        y1 = np.random.randint(0, h//2)
        x2 = np.random.randint(w//2, w)
        y2 = np.random.randint(h//2, h)
        boxes = torch.tensor([[x1,y1,x2,y2]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  
        img_t = self.transforms(img)
        target = {"boxes": boxes, "labels": labels}
        return img_t, target

# Build loaders
train_ds = TinyDetDataset(n=400, img_size=128)
val_ds   = TinyDetDataset(n=100, img_size=128)
def collate_fn(batch): return tuple(zip(*batch))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
print("TinyDet ready:", len(train_ds), "train,", len(val_ds), "val")


from PIL import Image
import numpy as np, torch
from torchvision.transforms import Compose, Resize, ToTensor

class TinyDetDataset(torch.utils.data.Dataset):
    def __init__(self, n=1000, img_size=128, transforms=None):
        self.n = n
        self.img_size = img_size
        self.transforms = transforms or Compose([Resize((img_size,img_size)), ToTensor()])
       
        self.base = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img, _ = self.base[idx % len(self.base)]
        img = img.resize((self.img_size, self.img_size))
        # one random box per image
        w,h = self.img_size, self.img_size
        x1 = np.random.randint(0, w//2)
        y1 = np.random.randint(0, h//2)
        x2 = np.random.randint(w//2, w)
        y2 = np.random.randint(h//2, h)
        boxes = torch.tensor([[x1,y1,x2,y2]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)  
        img_t = self.transforms(img)
        target = {"boxes": boxes, "labels": labels}
        return img_t, target

# Build loaders
train_ds = TinyDetDataset(n=400, img_size=128)
val_ds   = TinyDetDataset(n=100, img_size=128)
def collate_fn(batch): return tuple(zip(*batch))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
print("TinyDet ready:", len(train_ds), "train,", len(val_ds), "val")


from torchvision.ops import boxes as box_ops
from collections import defaultdict

def evaluate_mean_iou(model, loader, device):
    # simple mean IoU proxy across validation set
    model.eval()
    total_iou = 0.0; count = 0; total_dets = 0
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval"):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            for out, tgt in zip(outputs, targets):
                boxes_pred = out.get("boxes", torch.empty((0,4))).cpu()
                boxes_gt = tgt["boxes"].cpu()
                total_dets += boxes_pred.shape[0]
                if boxes_pred.shape[0] > 0 and boxes_gt.shape[0] > 0:
                    ious = box_ops.box_iou(boxes_pred, boxes_gt)
                    max_iou_per_gt, _ = ious.max(dim=0)
                    total_iou += max_iou_per_gt.sum().item()
                    count += boxes_gt.shape[0]
    mean_iou = (total_iou / count) if count>0 else 0.0
    avg_dets = total_dets / len(loader.dataset)
    return {"mean_iou": mean_iou, "avg_dets_per_image": avg_dets}

# SNN front
class SNNFrontDet(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, tsteps=4, beta=0.95, encode='latency'):
        super().__init__()
        self.t = int(tsteps)
        self.beta = float(beta)
        self.encode = encode
        # early downsample using stride=2 to reduce SNN compute
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.conv2 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def poisson_encode(self, x):
        # x: [B,C,H,W] in [0,1]
        T = self.t
        rnd = torch.rand((T,)+x.shape, device=x.device)
        xr = x.unsqueeze(0).expand(T, -1, -1, -1, -1)
        return (rnd < xr).float()

    def latency_encode(self, x):
        
        T = self.t
        B,C,H,W = x.shape
        times = torch.clamp(((1.0 - x) * (T - 1)).to(torch.long), 0, T-1)
        times_flat = times.view(B, -1)  # [B, N]
        N = C*H*W
        pos_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # [B,N]
        seq = torch.zeros((T, B, C, H, W), dtype=torch.float32, device=x.device)
        t_idx = times_flat; b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, N)
        ch_idx = (pos_idx // (H*W)) % C
        rem = pos_idx % (H*W)
        h_idx = rem // W; w_idx = rem % W
        seq[t_idx, b_idx, ch_idx, h_idx, w_idx] = 1.0
        return seq

    def forward(self, x_raw):
        
        if self.encode == 'poisson':
            seq = self.poisson_encode(x_raw)
        else:
            seq = self.latency_encode(x_raw)
        T,B,C,H,W = seq.shape
        spike_acc = torch.zeros((B, self.conv2.out_channels, H//2, W//2), device=x_raw.device)
        total_spikes = torch.zeros(B, device=x_raw.device)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(T):
            xt = seq[t]
            z1 = self.conv1(xt); z1 = self.bn1(z1)
            sp1, mem1 = self.lif1(z1, mem1)
            z2 = self.conv2(sp1); z2 = self.bn2(z2)
            sp2, mem2 = self.lif2(z2, mem2)
            spike_acc += sp2
            total_spikes += sp2.view(B, -1).sum(dim=1)
        feat = spike_acc / float(T)   # temporal pooling -> dense feature map
        avg_spikes = total_spikes / float(T)
        return feat, avg_spikes

#  backbone wrapper and FasterRCNN creation 
from torchvision.models import resnet50
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN

class ResNet50_Backbone_SNN(nn.Module):
    def __init__(self, snn_tsteps=4, pretrained_backbone=True, encode='latency'):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        self.resnet = resnet50(weights=weights)
        # Remove conv1 (SNN front replaces it)
        self.resnet.conv1 = nn.Identity()
        self.bn1 = self.resnet.bn1; self.relu = self.resnet.relu; self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1; self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3; self.layer4 = self.resnet.layer4
        # SNN front
        self.snn_front = SNNFrontDet(in_ch=3, out_ch=64, tsteps=snn_tsteps, encode=encode)
        # FPN
        in_channels_list = [512, 1024, 2048]  # c2,c3,c4 channels
        out_channels = 256
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
        self._out_channels = out_channels

    def forward(self, x_norm):
        # Here we assume input x_norm in [0,1] already. If normalized differently we will adjust accordingly.
        x_raw = x_norm
        feat0, _ = self.snn_front(x_raw)   # [B,64,H/2,W/2]
        x = self.bn1(feat0); x = self.relu(x); x = self.maxpool(x)  # now feed into resnet layers
        c1 = self.layer1(x); c2 = self.layer2(c1); c3 = self.layer3(c2); c4 = self.layer4(c3)
        features = {"0": c2, "1": c3, "2": c4}
        fpn_out = self.fpn(features)
        return fpn_out

    @property
    def out_channels(self): return self._out_channels

# baseline standard Faster R-CNN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
baseline_model = fasterrcnn_resnet50_fpn(pretrained=True)
# classifier for single-class demo
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
in_features = baseline_model.roi_heads.box_predictor.cls_score.in_features
baseline_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
baseline_model.to(device)

#  hybrid custom backbone with SNN front
backbone = ResNet50_Backbone_SNN(snn_tsteps=SNN_TSTEPS, pretrained_backbone=True, encode=SNN_ENCODING)

# Anchor sizes: one tuple per FPN feature (we use 3 feature maps)
anchor_sizes = ((32, 64, 128), (128, 256, 512), (512, 1024, 2048))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

# ROI pooler using feature names "0","1","2"
roi_pooler = MultiScaleRoIAlign(featmap_names=['0','1','2'], output_size=7, sampling_ratio=2)

# Construct FasterRCNN with custom backbone
hybrid_model = FasterRCNN(backbone,
                          num_classes=2,
                          rpn_anchor_generator=anchor_generator,
                          box_roi_pool=roi_pooler)
hybrid_model.to(device)

print("Baseline and Hybrid models instantiated.")

# training helpers (baseline)
from torch.cuda.amp import GradScaler, autocast
scaler_base = GradScaler()
def train_one_epoch_det(model, optimizer, loader, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc=f"Train E{epoch}"):
        imgs = [img.to(device) for img in imgs]
        targs = [{k:v.to(device) for k,v in t.items()} for t in targets]
        optimizer.zero_grad()
        # detection model returns loss dict when in training mode
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

@torch.no_grad()
def evaluate_det_proxy(model, loader, device):
    return evaluate_mean_iou(model, loader, device)

# quick baseline train & eval (1 epoch small)
opt_b = optim.SGD(baseline_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
print("Training baseline (1 epoch quick)...")
train_one_epoch_det(baseline_model, opt_b, train_loader, device, epoch=30)
b_metrics = evaluate_det_proxy(baseline_model, val_loader, device)
print("Baseline metrics (proxy):", b_metrics)
# Save baseline checkpoint & metrics
torch.save({'model_state': baseline_model.state_dict()}, "baseline_quick.pth")
open('det_exp_results.json','w').write(json.dumps({'baseline_metrics': b_metrics}, indent=2))

#  surrogate fine-tune for hybrid
scaler_h = GradScaler()
opt_h = optim.SGD(hybrid_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

def train_one_epoch_surrogate(model, optimizer, loader, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc=f"Surrogate Train E{epoch}"):
        imgs = [img.to(device) for img in imgs]
        targs = [{k:v.to(device) for k,v in t.items()} for t in targets]
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

#  hybrid fine-tuning for a few epochs
best = {'epoch':0, 'metric':0.0}
for e in range(1, EPOCHS_HYBRID+1):
    tr_loss = train_one_epoch_surrogate(hybrid_model, opt_h, train_loader, device, e)
    h_metrics = evaluate_det_proxy(hybrid_model, val_loader, device)
    print(f"Epoch {e} loss {tr_loss:.4f} eval {h_metrics}")
    torch.save({'epoch':e, 'model_state': hybrid_model.state_dict(), 'opt': opt_h.state_dict()}, f"hybrid_epoch{e}.pth")
    # save metrics incrementally
    try:
        prev = json.load(open('det_exp_results.json'))
    except:
        prev = {}
    prev['hybrid_metrics'] = h_metrics
    open('det_exp_results.json','w').write(json.dumps(prev, indent=2))
    if h_metrics['mean_iou'] > best['metric']:
        best = {'epoch': e, 'metric': h_metrics['mean_iou']}
print("Hybrid best:", best)

#  measurement helper
import subprocess, signal

def measure_model_inference(model, dataset, n_images=100, sample_interval=0.2, power_log='power.csv'):
   
    cmd = f"nvidia-smi --query-gpu=power.draw --format=csv -l {sample_interval}"
    try:
        logger = subprocess.Popen(cmd.split(), stdout=open(power_log,'w'), stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        time.sleep(0.5)
    except Exception as e:
        logger = None
    if hasattr(model, "eval"): model.eval()
    count = 0; t0 = time.time(); spikes_total = 0.0
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
            if count >= n_images: break
    t1 = time.time(); total_time = t1 - t0
    # stop logger
    powers = []
    if logger is not None:
        try:
            os.killpg(os.getpgid(logger.pid), signal.SIGTERM)
        except Exception:
            try: logger.terminate()
            except: pass
        time.sleep(0.1)
        try:
            with open(power_log,'r') as f:
                for line in f:
                    line=line.strip()
                    if not line: continue
                    import re
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if nums:
                        try: powers.append(float(nums[-1]))
                        except: pass
        except Exception:
            powers = []
    avg_power = float(np.mean(powers)) if len(powers)>0 else float('nan')
    energy_total = avg_power * total_time if not math.isnan(avg_power) else float('nan')
    energy_per_image = energy_total / count if count>0 and not math.isnan(energy_total) else float('nan')
    throughput = count / total_time if total_time>0 else float('nan')
    latency_ms = 1000.0 * total_time / count if count>0 else float('nan')
    avg_spikes_per_img = spikes_total / count if count>0 else float('nan')
    summary = {'n_images': count, 'total_time_s': total_time, 'throughput_img_s': throughput,
               'latency_ms': latency_ms, 'avg_power_w': avg_power, 'energy_per_image_j': energy_per_image,
               'avg_spikes_per_img': avg_spikes_per_img, 'raw_power_samples': len(powers)}
    return summary

#  measure baseline & hybrid 

baseline_meas = measure_model_inference(baseline_model, val_ds, n_images=50, sample_interval=0.2, power_log='p_base.csv')
print("Baseline measurement:", baseline_meas)
hybrid_meas   = measure_model_inference(hybrid_model, val_ds, n_images=50, sample_interval=0.2, power_log='p_hybrid.csv')
print("Hybrid measurement:", hybrid_meas)

# Save energy + det metrics into single JSON for comparison
res_comp = {'baseline': baseline_meas, 'hybrid': hybrid_meas}
# merge detection metrics if present
try:
    det_prev = json.load(open('det_exp_results.json'))
except:
    det_prev = {}
det_prev.update({'baseline_metrics': b_metrics if 'b_metrics' in globals() else det_prev.get('baseline_metrics'),
                 'hybrid_metrics': h_metrics if 'h_metrics' in globals() else det_prev.get('hybrid_metrics')})
open('det_exp_results.json','w').write(json.dumps(det_prev, indent=2))
open('comparison_energy.json','w').write(json.dumps(res_comp, indent=2))
print("Saved det_exp_results.json and comparison_energy.json")

# plot & CSV summary
import matplotlib.pyplot as plt
import pandas as pd

# Load saved files
det_res = {}
eng_res = {}
if os.path.exists('det_exp_results.json'):
    try: det_res = json.load(open('det_exp_results.json'))
    except: det_res = {}
if os.path.exists('comparison_energy.json'):
    try: eng_res = json.load(open('comparison_energy.json'))
    except: eng_res = {}

acc_base = det_res.get('baseline_metrics', {}).get('mean_iou', float('nan'))
acc_hyb  = det_res.get('hybrid_metrics', {}).get('mean_iou', float('nan'))
e_base = eng_res.get('baseline', {}).get('energy_per_image_j', float('nan'))
e_hyb  = eng_res.get('hybrid', {}).get('energy_per_image_j', float('nan'))

df = pd.DataFrame([{'model':'baseline','accuracy':acc_base,'energy_j':e_base},
                   {'model':'hybrid','accuracy':acc_hyb,'energy_j':e_hyb}]).set_index('model')

# percent changes
def pct(a,b):
    try:
        if math.isnan(a) or math.isnan(b): return None
        return 100.0*(b-a)/a if a!=0 else None
    except:
        return None

acc_pct = pct(acc_base, acc_hyb)
eng_pct = pct(e_base, e_hyb)

print("Baseline mean_iou:", acc_base, "Hybrid mean_iou:", acc_hyb, "Change %:", acc_pct)
print("Baseline energy J/img:", e_base, "Hybrid energy J/img:", e_hyb, "Change %:", eng_pct)

# Plot
fig, axes = plt.subplots(1,2, figsize=(12,4))
axes[0].bar(df.index, df['accuracy'].values); axes[0].set_title("Mean IoU (proxy)"); axes[0].set_ylabel("mean_iou")
axes[1].bar(df.index, df['energy_j'].values); axes[1].set_title("Energy per image (J)"); axes[1].set_ylabel("J/image")
for i, col in enumerate(['accuracy','energy_j']):
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

# Save CSV
csv_out = os.path.join(OUT_DIR, "acc_energy_summary.csv")
df.reset_index().to_csv(csv_out, index=False)
print("Saved CSV summary:", csv_out)
print("PPT path for slide reference:", PPT_PATH)


ablation_results = []
for T in [1,2,4]:
    for enc in ['latency','poisson']:
        print("Ablation T",T,"enc",enc)
        
        tmp_backbone = ResNet50_Backbone_SNN(snn_tsteps=T, pretrained_backbone=True, encode=enc)
        tmp_model = FasterRCNN(tmp_backbone, num_classes=2,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler).to(device)
        res = measure_model_inference(tmp_model, val_ds, n_images=20, sample_interval=0.2, power_log=f'p_T{T}_{enc}.csv')
        res.update({'T':T, 'encode':enc})
        ablation_results.append(res)
open('ablation_results.json','w').write(json.dumps(ablation_results, indent=2))
print("Saved ablation_results.json")


import json, torch
from collections import defaultdict


class DummyLIF(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def init_leaky(self):

        return None
    def forward(self, x, mem=None):

        return x, mem


tests = []
for T in [1,2,4]:
    for depth in [1,2]:
        print(f"\nTest => T {T} depth {depth}")
        # instantiate a fresh backbone
        tmp_backbone = ResNet50_Backbone_SNN(snn_tsteps=T, pretrained_backbone=True, encode='latency')

        if depth == 1:
            
            tmp_backbone.snn_front.lif2 = DummyLIF()

        else:

            pass


        tmp_model = FasterRCNN(tmp_backbone, num_classes=2,
                               rpn_anchor_generator=anchor_generator,
                               box_roi_pool=roi_pooler).to(device)

        try:
            res = measure_model_inference(tmp_model, val_ds, n_images=30, sample_interval=0.2,
                                          power_log=f'p_T{T}_d{depth}.csv')
        except Exception as e:
 
            res = {'error': str(e)}
            print("Measurement error:", e)

        res.update({'T': T, 'depth': depth})
        tests.append(res)

# Save results
open('focused_ablation_fixed.json','w').write(json.dumps(tests, indent=2))
print("\nSaved focused_ablation_fixed.json")
print("Results preview:")
for r in tests:
    print(r)


import json, time, torch

# Set config
T_best = 1
ENC = 'latency'
DEPTH = 2
print("Setting SNN config T=", T_best, "encode=", ENC, "depth=", DEPTH)

# Create fresh backbone & hybrid model (ensures lif modules are real snn.Leaky)
backbone_best = ResNet50_Backbone_SNN(snn_tsteps=T_best, pretrained_backbone=True, encode=ENC)
hybrid_best = FasterRCNN(backbone_best, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).to(device)


if DEPTH == 1:
   
    backbone_best.snn_front.lif2 = DummyLIF()
else:
   
    try:
        _ = backbone_best.snn_front.lif2
    except Exception as e:
        
        backbone_best.snn_front.lif2 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid())

# Fine-tune hybrid (surrogate training)
opt = torch.optim.SGD(hybrid_best.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
scaler_local = torch.cuda.amp.GradScaler()
EPOCHS_FT = 3   

def train_one_epoch_surrogate_local(model, optimizer, loader, device, epoch):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc=f"FT E{epoch}"):
        imgs = [img.to(device) for img in imgs]
        targs = [{k:v.to(device) for k,v in t.items()} for t in targets]
        optimizer.zero_grad()
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss_dict = model(imgs, targs)
                loss = sum(loss_dict.values())
        else:
            loss_dict = model(imgs, targs)
            loss = sum(loss_dict.values())
        scaler_local.scale(loss).backward()
        scaler_local.step(optimizer)
        scaler_local.update()
        running_loss += loss.item()
    return running_loss / len(loader)

# Run FT
best_metric = -1.0
best_epoch = 0
for e in range(1, EPOCHS_FT+1):
    tr_loss = train_one_epoch_surrogate_local(hybrid_best, opt, train_loader, device, e)
    metrics = evaluate_mean_iou(hybrid_best, val_loader, device)
    print(f"Epoch {e} loss {tr_loss:.4f} eval mean_iou {metrics['mean_iou']:.4f}")
    # save checkpoint
    torch.save({'epoch':e, 'model_state':hybrid_best.state_dict(), 'opt':opt.state_dict()}, f"hybrid_best_T{T_best}_E{e}.pth")
    if metrics['mean_iou'] > best_metric:
        best_metric = metrics['mean_iou']; best_epoch = e

print("Fine-tune best mean_iou:", best_metric, "epoch:", best_epoch)


hybrid_meas_best = measure_model_inference(hybrid_best, val_ds, n_images=200, sample_interval=0.2, power_log='hybrid_best_power.csv')
print("Hybrid best measurement:", hybrid_meas_best)

# Also measure baseline again robustly (n_images=200)
print("Measuring baseline inference (n_images=200)...")
baseline_meas_best = measure_model_inference(baseline_model, val_ds, n_images=200, sample_interval=0.2, power_log='baseline_best_power.csv')
print("Baseline measurement:", baseline_meas_best)


final_det_metrics = {}


if 'b_metrics' in globals() and isinstance(b_metrics, dict):
    final_det_metrics['baseline_metrics'] = b_metrics
else:
    try:

        prev_det = json.load(open('det_exp_results.json'))
        final_det_metrics['baseline_metrics'] = prev_det.get('baseline_metrics', {})
    except:
        final_det_metrics['baseline_metrics'] = {}


final_det_metrics['hybrid_metrics'] = {'mean_iou': best_metric}

open('det_exp_results.json','w').write(json.dumps(final_det_metrics, indent=2))

comp = {'baseline': baseline_meas_best, 'hybrid': hybrid_meas_best}
open('comparison_energy.json','w').write(json.dumps(comp, indent=2))
print("Saved det_exp_results.json and comparison_energy.json for T=1,depth=2")

## wit some change and fixes

import os, time, json, math, signal, subprocess, re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.transforms import ToTensor
import yaml

# Config
OUT_DIR = Path("/mnt/data/yolo_v8_cmp_fixed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IM_DIR = OUT_DIR / "images"
LAB_DIR = OUT_DIR / "labels"
TRAIN_DIR = IM_DIR / "train"
VAL_DIR = IM_DIR / "val"
(TRAIN_DIR).mkdir(parents=True, exist_ok=True)
(VAL_DIR).mkdir(parents=True, exist_ok=True)
(LAB_DIR / "train").mkdir(parents=True, exist_ok=True)
(LAB_DIR / "val").mkdir(parents=True, exist_ok=True)

DATA_YAML = OUT_DIR / "data.yaml"
YOLO_MODEL = "yolov8s.pt"      
YOLO_IMG_SIZE = 128            
EPOCHS = 5                      
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_MEASURE_IMAGES = 200
POWER_LOG = OUT_DIR / "yolo_power.csv"
RESULTS_JSON = OUT_DIR / "yolo_results.json"


MEASURE_TYPE = "full" 

print("Device:", DEVICE)

#  Ensure ultralytics
try:
    from ultralytics import YOLO
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

# changing tiny_dataset to yolo type
def build_yolo_dataset(n_train=400, n_val=100, img_size=128, out_dir=OUT_DIR):
    base = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    def to_yolo(box, W, H):
        x1,y1,x2,y2 = box
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w = (x2 - x1) / float(W)
        h = (y2 - y1) / float(H)
        return [0, cx, cy, w, h]  

    def save_split(n, img_folder, lab_folder, offset=0):
        for i in range(n):
            img, _ = base[(offset + i) % len(base)]
            img = img.resize((img_size, img_size))
            img_path = Path(img_folder) / f"{i:05d}.jpg"
            img.save(img_path)
            w,h = img_size, img_size
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = np.random.randint(w//2, w)
            y2 = np.random.randint(h//2, h)
            yolo_box = to_yolo([x1,y1,x2,y2], w, h)
            label_path = Path(lab_folder) / f"{i:05d}.txt"
            with open(label_path, "w") as f:
                f.write(" ".join(map(str, yolo_box)) + "\n")

    save_split(n_train, str(TRAIN_DIR), str(LAB_DIR / "train"), offset=0)
    save_split(n_val, str(VAL_DIR), str(LAB_DIR / "val"), offset=n_train)

    data = {"train": str(TRAIN_DIR), "val": str(VAL_DIR), "nc": 1, "names": ["obj"]}
    with open(DATA_YAML, "w") as f:
        yaml.safe_dump(data, f)
    print("Wrote YOLO dataset to:", OUT_DIR)
    return DATA_YAML

data_yaml = build_yolo_dataset(n_train=400, n_val=100, img_size=YOLO_IMG_SIZE)

# Train YOLOv8 
print("Loading YOLO model:", YOLO_MODEL)
model = YOLO(YOLO_MODEL)  # loads pretrained weights

print(f"Training YOLO (epochs={EPOCHS}, imgsz={YOLO_IMG_SIZE}, batch={BATCH_SIZE}) ...")
model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=YOLO_IMG_SIZE, batch=BATCH_SIZE, device=DEVICE, workers=4)
print("Training finished. Best weights saved in ultralytics run folder.")

# Validate via ultralytics 

val_res = model.val(data=str(DATA_YAML), imgsz=YOLO_IMG_SIZE, batch=1, device=DEVICE)
try:
    mAP50_95 = float(val_res.box.map)
    mAP50 = float(val_res.box.map50)
except Exception:
    mAP50_95 = None; mAP50 = None
print("mAP50-95:", mAP50_95, "mAP50:", mAP50)

# Compute mean IoU  
def compute_mean_iou_yolo(model, val_images_dir, labels_dir, imgsz=YOLO_IMG_SIZE, conf=0.001):
    val_paths = sorted(Path(val_images_dir).glob("*.jpg"))
    total_iou = 0.0
    total_images = len(val_paths)
    images_with_preds = 0

    for p in tqdm(val_paths, desc="YOLO IoU Eval"):
        im = np.array(Image.open(p).convert("RGB"))
        
        try:
            res = model.predict(source=str(p), imgsz=imgsz, device=DEVICE, conf=conf, verbose=False)
            boxes_pred = []
            if len(res) > 0 and hasattr(res[0], "boxes") and len(res[0].boxes) > 0:
                try:
                    xyxy = res[0].boxes.xyxy.cpu().numpy()
                except Exception:
                    xyxy = np.array([])
                if getattr(xyxy, "size", 0) > 0:
                    for row in xyxy:
                        x1,y1,x2,y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                        boxes_pred.append([x1,y1,x2,y2])
        except Exception:
            boxes_pred = []

        # load GT (YOLO format)
        lab_file = Path(labels_dir) / (p.stem + ".txt")
        if not lab_file.exists():
            
            total_images -= 1
            continue
        with open(lab_file) as f:
            parts = f.readline().strip().split()
            if len(parts) < 5:
                gt = None
            else:
                cx, cy, w, h = map(float, parts[1:5])
                W, H = im.shape[1], im.shape[0]
                x1 = (cx - w/2.0) * W; y1 = (cy - h/2.0) * H
                x2 = (cx + w/2.0) * W; y2 = (cy + h/2.0) * H
                gt = [x1, y1, x2, y2]
        if gt is None:
            total_images -= 1
            continue

        best_iou = 0.0
        for bp in boxes_pred:
            xA = max(bp[0], gt[0]); yA = max(bp[1], gt[1])
            xB = min(bp[2], gt[2]); yB = min(bp[3], gt[3])
            interW = max(0.0, xB - xA)
            interH = max(0.0, yB - yA)
            inter = interW * interH
            areaA = max(0.0, (bp[2]-bp[0])) * max(0.0, (bp[3]-bp[1]))
            areaB = max(0.0, (gt[2]-gt[0])) * max(0.0, (gt[3]-gt[1]))
            union = areaA + areaB - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou

        if len(boxes_pred) > 0:
            images_with_preds += 1
        total_iou += best_iou  

    mean_iou = (total_iou / total_images) if total_images > 0 else 0.0
    return {"mean_iou": mean_iou, "images_with_any_pred": images_with_preds, "total_images": total_images}

yolo_iou_metrics = compute_mean_iou_yolo(model, str(VAL_DIR), str(LAB_DIR / "val"), imgsz=YOLO_IMG_SIZE, conf=0.001)
print("YOLO mean IoU (proxy):", yolo_iou_metrics)

# ---------- Inference measurement (timing + nvidia-smi power) ----------
def measure_yolo_inference(model, val_images_dir, n_images=200, imgsz=YOLO_IMG_SIZE, sample_interval=0.2, power_log=POWER_LOG, measure_type="full", conf=0.001):
    # spawn nvidia-smi logger
    cmd = f"nvidia-smi --query-gpu=power.draw --format=csv -l {sample_interval}"
    logger = None
    try:
        logger = subprocess.Popen(cmd.split(), stdout=open(power_log, "w"), stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        time.sleep(0.5)
    except Exception:
        logger = None

    val_paths = sorted(Path(val_images_dir).glob("*.jpg"))[:n_images]

    # Warm-up
    for p in val_paths[:10]:
        try:
            if measure_type == "full":
                _ = model.predict(source=str(p), imgsz=imgsz, device=DEVICE, conf=conf, verbose=False)
            else:
                # raw forward: load tensor and call model.model
                im = Image.open(p).convert("RGB").resize((imgsz, imgsz))
                t = ToTensor()(im).unsqueeze(0).to(DEVICE)  # (1,3,H,W) in 0..1
                try:
                    with torch.no_grad():
                        _ = model.model(t)
                except Exception:
                    pass
        except Exception:
            pass

    # Measurement loop
    times = []
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.time()
    for p in val_paths:
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        s = time.time()
        try:
            if measure_type == "full":
                _ = model.predict(source=str(p), imgsz=imgsz, device=DEVICE, conf=conf, verbose=False)
            else:
                im = Image.open(p).convert("RGB").resize((imgsz, imgsz))
                t = ToTensor()(im).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    _ = model.model(t)
        except Exception:
            pass
        if DEVICE.startswith("cuda"):
            torch.cuda.synchronize()
        e = time.time()
        times.append(e - s)
    t1 = time.time()
    total_time = t1 - t0

    # stop logger
    if logger is not None:
        try:
            os.killpg(os.getpgid(logger.pid), signal.SIGTERM)
        except Exception:
            try:
                logger.terminate()
            except Exception:
                pass
        time.sleep(0.1)

    # parse power log
    powers = []
    if Path(power_log).exists():
        try:
            with open(power_log, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    if nums:
                        try:
                            powers.append(float(nums[-1]))
                        except Exception:
                            pass
        except Exception:
            powers = []

    avg_power = float(np.mean(powers)) if len(powers) > 0 else float("nan")
    energy_total = avg_power * total_time if not math.isnan(avg_power) else float("nan")
    energy_per_image = energy_total / len(times) if len(times) > 0 and not math.isnan(energy_total) else float("nan")
    throughput = len(times) / total_time if total_time > 0 else float("nan")
    latency_ms = 1000.0 * np.mean(times) if len(times) > 0 else float("nan")

    return {'n_images': len(times), 'total_time_s': total_time, 'throughput_img_s': throughput,
            'latency_ms': latency_ms, 'avg_power_w': avg_power, 'energy_per_image_j': energy_per_image,
            'raw_power_samples': len(powers)}

print(f"Measuring YOLO inference ({MEASURE_TYPE}) n_images={N_MEASURE_IMAGES} ...")
yolo_meas = measure_yolo_inference(model, str(VAL_DIR), n_images=N_MEASURE_IMAGES, imgsz=YOLO_IMG_SIZE,
                                   sample_interval=0.2, power_log=str(POWER_LOG), measure_type=MEASURE_TYPE, conf=0.001)
print("YOLO measurement:", yolo_meas)


out = {
    "mAP50_95": mAP50_95,
    "mAP50": mAP50,
    "mean_iou_proxy": yolo_iou_metrics,
    "measurement": yolo_meas,
    "dataset_folder": str(OUT_DIR),
    "measure_type": MEASURE_TYPE
}
with open(RESULTS_JSON, "w") as f:
    json.dump(out, f, indent=2)
print("Saved YOLO results to:", RESULTS_JSON)

# ---------- Quick print summary ----------
print("\n--- YOLOv8 Baseline Summary ---")
print("mAP50-95:", mAP50_95, "mAP50:", mAP50)
print("mean_iou (proxy):", yolo_iou_metrics)
print("Latency (ms):", yolo_meas.get("latency_ms"), "Throughput img/s:", yolo_meas.get("throughput_img_s"),
      "Energy J/img:", yolo_meas.get("energy_per_image_j"))
print("Dataset / outputs:", OUT_DIR)

# Plot Accuracy & Energy Comparison for Baseline vs Hybrid 
import json
import matplotlib.pyplot as plt
import numpy as np

# 
det = json.load(open("det_exp_results.json"))
comp = json.load(open("comparison_energy.json"))

# Extract metrics
baseline_acc = det.get("baseline_metrics", {}).get("mean_iou", np.nan)
hybrid_acc   = det.get("hybrid_metrics", {}).get("mean_iou", np.nan)

baseline_energy = comp.get("baseline", {}).get("energy_per_image_j", np.nan)
hybrid_energy   = comp.get("hybrid", {}).get("energy_per_image_j", np.nan)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# ---- ACCURACY ----
axs[0].bar(["baseline", "hybrid"], [baseline_acc, hybrid_acc])
axs[0].set_title("Mean IoU (Accuracy)")
axs[0].set_ylabel("Mean IoU")
axs[0].set_ylim(0, max(baseline_acc, hybrid_acc)*1.3)
for i, v in enumerate([baseline_acc, hybrid_acc]):
    axs[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

# ENERGY 
axs[1].bar(["baseline", "hybrid"], [baseline_energy, hybrid_energy], color=["tab:gray","tab:blue"])
axs[1].set_title("Energy per Image (J)")
axs[1].set_ylabel("Joules / image")
axs[1].set_ylim(0, max(baseline_energy, hybrid_energy)*1.3)
for i, v in enumerate([baseline_energy, hybrid_energy]):
    axs[1].text(i, v + 0.02, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig("/mnt/data/final_acc_energy_comparison.png", dpi=150)
plt.show()



import json
import numpy as np

# Load results
det = json.load(open("det_exp_results.json"))
comp = json.load(open("comparison_energy.json"))

# Extract metrics
acc_b = det["baseline_metrics"]["mean_iou"]
acc_h = det["hybrid_metrics"]["mean_iou"]

eng_b = comp["baseline"]["energy_per_image_j"]
eng_h = comp["hybrid"]["energy_per_image_j"]

lat_b = comp["baseline"]["latency_ms"]
lat_h = comp["hybrid"]["latency_ms"]

thr_b = comp["baseline"]["throughput_img_s"]
thr_h = comp["hybrid"]["throughput_img_s"]

pow_b = comp["baseline"]["avg_power_w"]
pow_h = comp["hybrid"]["avg_power_w"]

# Percent change helper
pct = lambda b, h: 100*(h - b)/b

print("=== Percentage Change (Hybrid vs Baseline) ===")
print(f"Accuracy change     : {pct(acc_b, acc_h):+.2f}%")
print(f"Energy change       : {pct(eng_b, eng_h):+.2f}%")
print(f"Latency change      : {pct(lat_b, lat_h):+.2f}%")
print(f"Throughput change   : {pct(thr_b, thr_h):+.2f}%")
print(f"Avg Power change    : {pct(pow_b, pow_h):+.2f}%")


import json
import matplotlib.pyplot as plt
import numpy as np


comp = json.load(open("comparison_energy.json"))


baseline_latency = comp.get("baseline", {}).get("latency_ms", np.nan)
hybrid_latency   = comp.get("hybrid", {}).get("latency_ms", np.nan)

# Plot
fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(["baseline", "hybrid"], [baseline_latency, hybrid_latency], color=["tab:red", "tab:green"])
ax.set_title("Latency per Image")
ax.set_ylabel("Latency (ms)")
ax.set_ylim(0, max(baseline_latency, hybrid_latency)*1.3)
for i, v in enumerate([baseline_latency, hybrid_latency]):
    ax.text(i, v + 1, f"{v:.2f}", ha="center")

plt.tight_layout()
plt.savefig("/mnt/data/latency_comparison.png", dpi=150)
plt.show()

print("Saved latency plot:", "/mnt/data/latency_comparison.png")