#!/usr/bin/env python3
import sys
sys.path.insert(0, "/root/RC-FedBiT")
import torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from src.compression.rank1_compress import Rank1GradientCompressor
from src.channel.channel_adaptive import ChannelAdaptiveSelector, RayleighChannelSimulator
from src.data.partition import dirichlet_partition
from src.federated.client import FedBiTClient

DEVICE = "cuda"
transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_ds = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform)
client_datasets = dirichlet_partition(train_ds, 5, alpha=0.5)
client_loaders = [DataLoader(ds, batch_size=32, shuffle=True, num_workers=2) for ds in client_datasets]

def make_model(device):
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    return m.to(device)

config = {
    "device": DEVICE, "lr": 0.01, "local_lr": 0.01, "local_epochs": 1,
    "gamma_high": 15.0, "gamma_low": 5.0, "total_rounds": 3, "mean_snr_db": 10.0
}
ch_sim = RayleighChannelSimulator(mean_snr_db=10.0)
global_model = make_model(DEVICE)
global_weights = {k:v.cpu().clone() for k,v in global_model.state_dict().items()}
snr_vals = ch_sim.sample_snr(5)

payloads, weights = [], []
for k in range(5):
    c_model = make_model(DEVICE)
    c = FedBiTClient(k, c_model, client_loaders[k], config)
    payload, wt, stats = c.train(global_weights, snr_vals[k], 0)
    payloads.append(payload)
    weights.append(wt)

# Now simulate aggregation
layer_updates = defaultdict(list)
for payload, weight in zip(payloads, weights):
    for name, data in payload.items():
        layer_updates[name].append((data, weight))

# Check each layer
problem_layers = []
for name, updates in layer_updates.items():
    first_data = updates[0][0]
    is_fp16 = first_data.get("type") == "fp16"
    for i, (data, w) in enumerate(updates):
        has_B = "B" in data
        data_type = data.get("type", data.get("mode", "unknown"))
        if (not is_fp16) and (not has_B):
            problem_layers.append((name, i, data_type, list(data.keys())))

if problem_layers:
    print("PROBLEM LAYERS (rank-1 section but no B):")
    for name, client_idx, dtype, keys in problem_layers:
        print(f"  Layer={name}, Client={client_idx}, type={dtype}, keys={keys}")
else:
    print("No problem layers found - all B keys present")

# Also check state_dict types  
m = make_model(DEVICE)
sd = m.state_dict()
print(f"\nModel has {len(sd)} state_dict entries")
non_param_keys = []
for n, v in sd.items():
    if not v.is_floating_point():
        non_param_keys.append((n, v.dtype, v.shape))
if non_param_keys:
    print("Non-float state dict entries:")
    for n, dt, sh in non_param_keys:
        print(f"  {n}: {dt} {sh}")
else:
    print("All state_dict entries are float")
