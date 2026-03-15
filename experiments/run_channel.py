#!/usr/bin/env python3
"""
Channel Adaptive Analysis: mean_snr_db sweep with multiple methods.
Methods: rc_fedbit, signsgd, qsgd, all
"""
import sys, os, json, time, argparse, copy
sys.path.insert(0, "/root/RC-FedBiT")

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import timm
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.data.partition import dirichlet_partition
from src.channel.channel_adaptive import RayleighChannelSimulator
from src.federated.server import FedBiTServer
from src.federated.client import FedBiTClient
from src.baselines.signsgd import SignSGDClient, SignSGDServer
from src.baselines.qsgd import QSGDClient, QSGDServer


def make_model(device, nc=10):
    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=nc,
        img_size=32,
        patch_size=4,
    ).to(device)


def eval_model(model, loader, device):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            c += (pred == y).sum().item()
            t += y.size(0)
    return 100.0 * c / t if t > 0 else 0.0


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_rc_fedbit(args, global_model, client_loaders, test_loader, device, rng):
    tag = f"rc_fedbit_snr{int(args.snr)}dB"
    config = {
        "device": device,
        "lr": args.lr,
        "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
        "gamma_high": 15.0,
        "gamma_low": 5.0,
        "total_rounds": args.rounds,
        "mean_snr_db": args.snr,
    }
    server = FedBiTServer(global_model, config)
    ch_sim = RayleighChannelSimulator(mean_snr_db=args.snr, seed=args.seed)
    results = []
    cum_bits = 0

    for rnd in range(args.rounds):
        gw = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        snr_vals = ch_sim.sample_snr(n_part)
        payloads, wts = [], []
        for i, k in enumerate(selected):
            c_model = copy.deepcopy(global_model)
            c = FedBiTClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(gw, snr_vals[i], rnd)
            payloads.append(payload)
            wts.append(len(client_loaders[k].dataset))
            for nm, dat in payload.items():
                mode = dat.get("mode", dat.get("type", ""))
                if mode == "rank1_fp32":
                    pp = global_model.state_dict()[nm]
                    W = pp.view(pp.shape[0], -1)
                    m, n = W.shape
                    cum_bits += m * n + 32 * (m + n)
                elif mode == "rank1_int8":
                    pp = global_model.state_dict()[nm]
                    W = pp.view(pp.shape[0], -1)
                    m, n = W.shape
                    cum_bits += m * n + 8 * (m + n)
                elif mode == "binary_only":
                    cum_bits += global_model.state_dict()[nm].numel()
                else:
                    cum_bits += global_model.state_dict()[nm].numel() * 16
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({
            "method": tag,
            "snr_db": args.snr,
            "round": rnd + 1,
            "acc": acc,
            "comm_bits": cum_bits,
        })
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}%", flush=True)

    return results


def run_signsgd(args, global_model, client_loaders, test_loader, device, rng):
    tag = f"signsgd_snr{int(args.snr)}dB"
    config = {
        "device": device,
        "lr": args.lr,
        "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
    }
    server = SignSGDServer(global_model, config)
    n_params = count_params(global_model)
    results = []
    cum_bits = 0

    for rnd in range(args.rounds):
        gw = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for i, k in enumerate(selected):
            c_model = copy.deepcopy(global_model)
            c = SignSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(gw)
            payloads.append(payload)
            wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params  # 1 bit per param
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({
            "method": tag,
            "snr_db": args.snr,
            "round": rnd + 1,
            "acc": acc,
            "comm_bits": cum_bits,
        })
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}%", flush=True)

    return results


def run_qsgd(args, global_model, client_loaders, test_loader, device, rng):
    tag = f"qsgd_snr{int(args.snr)}dB"
    config = {
        "device": device,
        "lr": args.lr,
        "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
        "qsgd_bits": 4,
    }
    server = QSGDServer(global_model, config)
    n_params = count_params(global_model)
    results = []
    cum_bits = 0

    for rnd in range(args.rounds):
        gw = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for i, k in enumerate(selected):
            c_model = copy.deepcopy(global_model)
            c = QSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(gw)
            payloads.append(payload)
            wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params * 4  # 4 bits per param
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({
            "method": tag,
            "snr_db": args.snr,
            "round": rnd + 1,
            "acc": acc,
            "comm_bits": cum_bits,
        })
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}%", flush=True)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snr", type=float, required=True)
    p.add_argument("--method", type=str, default="rc_fedbit",
                   choices=["rc_fedbit", "signsgd", "qsgd", "all"])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--participation", type=float, default=0.1)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Channel analysis: mean_SNR={args.snr}dB, method={args.method}", flush=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    train_ds = datasets.CIFAR10(
        "/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform
    )
    test_ds = datasets.CIFAR10(
        "/root/RC-FedBiT/data/cifar10", train=False, download=False, transform=transform
    )
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    client_datasets = dirichlet_partition(
        train_ds, args.n_clients, alpha=0.5, seed=args.seed
    )
    client_loaders = [
        DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]

    methods_to_run = (
        ["rc_fedbit", "signsgd", "qsgd"] if args.method == "all" else [args.method]
    )

    all_new_results = []
    for method in methods_to_run:
        rng = np.random.RandomState(args.seed)
        global_model = make_model(DEVICE)
        print(f"\n=== Running {method} at SNR={args.snr}dB ===", flush=True)
        if method == "rc_fedbit":
            res = run_rc_fedbit(args, global_model, client_loaders, test_loader, DEVICE, rng)
        elif method == "signsgd":
            res = run_signsgd(args, global_model, client_loaders, test_loader, DEVICE, rng)
        elif method == "qsgd":
            res = run_qsgd(args, global_model, client_loaders, test_loader, DEVICE, rng)
        all_new_results.extend(res)
        print(f"[{method}] Final acc: {res[-1]['acc']:.2f}%", flush=True)

    # Save results
    out_file = "/root/RC-FedBiT/results/channel_analysis.json"
    os.makedirs("/root/RC-FedBiT/results", exist_ok=True)
    if os.path.exists(out_file):
        with open(out_file) as f:
            existing = json.load(f)
    else:
        existing = []

    # Replace entries for methods we just ran at this SNR
    new_method_tags = {r["method"] for r in all_new_results}
    existing = [r for r in existing if r.get("method") not in new_method_tags]
    existing.extend(all_new_results)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
