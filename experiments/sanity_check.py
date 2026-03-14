#!/usr/bin/env python3
"""RC-FedBiT Sanity Check - 5 clients, 3 rounds, CIFAR-10, ViT-Tiny"""
import sys, os
sys.path.insert(0, "/root/RC-FedBiT")
import torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.compression.rank1_compress import Rank1GradientCompressor
from src.channel.channel_adaptive import ChannelAdaptiveSelector, RayleighChannelSimulator
from src.data.partition import dirichlet_partition
from src.federated.server import FedBiTServer
from src.federated.client import FedBiTClient

def make_model(device):
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    return m.to(device)

def eval_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
    return 100*correct/total

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform)
    test_ds  = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    print(f"Data loaded: train={len(train_ds)}, test={len(test_ds)}", flush=True)

    N_CLIENTS = 5
    client_datasets = dirichlet_partition(train_ds, N_CLIENTS, alpha=0.5)
    client_loaders  = [DataLoader(ds, batch_size=32, shuffle=True, num_workers=2) for ds in client_datasets]
    print(f"FL partition: {N_CLIENTS} clients, sizes={[len(ds) for ds in client_datasets]}", flush=True)

    global_model = make_model(DEVICE)
    config = {
        "device": DEVICE, "lr": 0.01, "local_lr": 0.01, "local_epochs": 1,
        "gamma_high": 15.0, "gamma_low": 5.0, "total_rounds": 3, "mean_snr_db": 10.0
    }
    server   = FedBiTServer(global_model, config)
    ch_sim   = RayleighChannelSimulator(mean_snr_db=10.0)
    compressor = Rank1GradientCompressor()

    print("\n=== RC-FedBiT Sanity Check ===", flush=True)
    for rnd in range(3):
        global_weights = {k:v.cpu().clone() for k,v in global_model.state_dict().items()}
        snr_vals = ch_sim.sample_snr(N_CLIENTS)
        payloads, weights = [], []

        for k in range(N_CLIENTS):
            c_model = make_model(DEVICE)
            client  = FedBiTClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = client.train(global_weights, snr_vals[k], rnd)
            payloads.append(payload); weights.append(wt)
            mode = list(payload.values())[0].get("mode","?")
            print(f"  Client {k}: SNR={snr_vals[k]:.1f}dB | {mode} | cr={stats['cr']:.1f}x | loss={stats['loss']:.3f}", flush=True)

        server.aggregate(payloads, weights)
        acc = eval_model(global_model, test_loader, DEVICE)
        print(f"Round {rnd+1}/3 | Test Acc: {acc:.2f}%\n", flush=True)

    print("=== Sanity check PASSED ===", flush=True)

if __name__ == "__main__":
    main()
