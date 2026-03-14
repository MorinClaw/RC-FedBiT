#!/usr/bin/env python3
"""
RC-FedBiT Main Experiment Runner
Phase 2: CIFAR-10, n_clients=100, participation=0.1, rounds=100, local_epochs=5, ViT-Tiny, alpha=0.5
Usage: python run_main.py --method <fedavg|signsgd|qsgd|powersgd|rc_fedbit> [--alpha 0.5] [--rounds 100]
"""
import sys, os, argparse, json, time
sys.path.insert(0, "/root/RC-FedBiT")

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.data.partition import dirichlet_partition, iid_partition

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["fedavg","signsgd","qsgd","powersgd","rc_fedbit"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--participation", type=float, default=0.1)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--mean_snr_db", type=float, default=10.0)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    # Ablation flags for rc_fedbit
    p.add_argument("--no_rank1", action="store_true", help="A0: pure binary (no rank1 comp)")
    p.add_argument("--no_ca_rs", action="store_true", help="A1: disable channel-adaptive rate selection")
    p.add_argument("--no_nia_cva", action="store_true", help="A2: disable NIA-CVA alignment")
    p.add_argument("--no_afc", action="store_true", help="A3: disable AFC calibration")
    p.add_argument("--ablation_tag", type=str, default=None)
    return p.parse_args()

def make_model(device, num_classes=10):
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    return m.to(device)

def eval_model(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def run_rc_fedbit(args, global_model, client_loaders, test_loader, device, rng):
    from src.compression.rank1_compress import Rank1GradientCompressor
    from src.channel.channel_adaptive import ChannelAdaptiveSelector, RayleighChannelSimulator
    from src.federated.server import FedBiTServer
    from src.federated.client import FedBiTClient
    import torch.nn as nn

    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
        "gamma_high": 15.0, "gamma_low": 5.0,
        "total_rounds": args.rounds, "mean_snr_db": args.mean_snr_db,
    }
    server = FedBiTServer(global_model, config)
    ch_sim = RayleighChannelSimulator(mean_snr_db=args.mean_snr_db, seed=args.seed)
    results = []
    cum_bits = 0

    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        # Sample participating clients
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        snr_vals = ch_sim.sample_snr(n_part)

        payloads, wts = [], []
        round_bits = 0
        for i, k in enumerate(selected):
            c_model = make_model(device)
            c = FedBiTClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights, snr_vals[i], rnd)
            payloads.append(payload)
            wts.append(wt * len(client_loaders[k].dataset))
            # Estimate comm bits for this client
            for name, data in payload.items():
                mode = data.get("mode", data.get("type", ""))
                if mode == "rank1_fp32":
                    p = global_model.state_dict()[name]
                    W = p.view(p.shape[0], -1)
                    m, n = W.shape
                    round_bits += m * n + 32 * (m + n)
                elif mode == "rank1_int8":
                    p = global_model.state_dict()[name]
                    W = p.view(p.shape[0], -1)
                    m, n = W.shape
                    round_bits += m * n + 8 * (m + n)
                elif mode == "binary_only":
                    p = global_model.state_dict()[name]
                    round_bits += p.numel()
                elif mode == "fp16":
                    p = global_model.state_dict()[name]
                    round_bits += p.numel() * 16

        cum_bits += round_bits
        server.aggregate(payloads, wts)

        acc = eval_model(global_model, test_loader, device)
        tag = args.ablation_tag or "rc_fedbit"
        results.append({"method": tag, "round": rnd + 1, "acc": acc, "comm_bits": cum_bits})
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)

    return results

def run_fedavg(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.fedavg import FedAvgClient, FedAvgServer
    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
    }
    server = FedAvgServer(global_model, config)
    results = []
    cum_bits = 0
    n_params = count_params(global_model)

    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        deltas, wts = [], []
        for k in selected:
            c_model = make_model(device)
            c = FedAvgClient(k, c_model, client_loaders[k], config)
            delta, wt, stats = c.train(global_weights)
            deltas.append(delta)
            wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params * 32  # FP32 per client

        server.aggregate(deltas, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "fedavg", "round": rnd + 1, "acc": acc, "comm_bits": cum_bits})
        print(f"[fedavg] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)

    return results

def run_signsgd(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.signsgd import SignSGDClient, SignSGDServer
    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
    }
    server = SignSGDServer(global_model, config)
    results = []
    cum_bits = 0
    n_params = count_params(global_model)

    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for k in selected:
            c_model = make_model(device)
            c = SignSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights)
            payloads.append(payload)
            wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params  # 1 bit per param

        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "signsgd", "round": rnd + 1, "acc": acc, "comm_bits": cum_bits})
        print(f"[signsgd] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)

    return results

def run_qsgd(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.qsgd import QSGDClient, QSGDServer
    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs, "qsgd_bits": 4,
    }
    server = QSGDServer(global_model, config)
    results = []
    cum_bits = 0
    n_params = count_params(global_model)

    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for k in selected:
            c_model = make_model(device)
            c = QSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights)
            payloads.append(payload)
            wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params * 4  # 4 bits per param

        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "qsgd", "round": rnd + 1, "acc": acc, "comm_bits": cum_bits})
        print(f"[qsgd] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)

    return results

def run_powersgd(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.powersgd import PowerSGDClient, PowerSGDServer
    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
    }
    server = PowerSGDServer(global_model, config)
    results = []
    cum_bits = 0

    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for k in selected:
            c_model = make_model(device)
            c = PowerSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights)
            payloads.append(payload)
            wts.append(len(client_loaders[k].dataset))
            cum_bits += stats["comm_bits"]

        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "powersgd", "round": rnd + 1, "acc": acc, "comm_bits": cum_bits})
        print(f"[powersgd] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)

    return results

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}, Method: {args.method}, Alpha: {args.alpha}, Rounds: {args.rounds}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform)
    test_ds  = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    if args.alpha >= 100:
        client_datasets = iid_partition(train_ds, args.n_clients, seed=args.seed)
        print(f"IID partition: {args.n_clients} clients", flush=True)
    else:
        client_datasets = dirichlet_partition(train_ds, args.n_clients, alpha=args.alpha, seed=args.seed)
        print(f"Dirichlet(alpha={args.alpha}) partition: {args.n_clients} clients", flush=True)

    # Create loaders lazily to avoid 200+ worker processes
    client_loaders = [
        DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]

    global_model = make_model(DEVICE)

    runners = {
        "fedavg": run_fedavg,
        "signsgd": run_signsgd,
        "qsgd": run_qsgd,
        "powersgd": run_powersgd,
        "rc_fedbit": run_rc_fedbit,
    }

    t0 = time.time()
    results = runners[args.method](args, global_model, client_loaders, test_loader, DEVICE, rng)
    elapsed = time.time() - t0

    print(f"\nTotal time: {elapsed/3600:.2f}h | Final acc: {results[-1]['acc']:.2f}%", flush=True)

    # Save results
    if args.output:
        out_file = args.output
    else:
        os.makedirs("/root/RC-FedBiT/results", exist_ok=True)
        out_file = f"/root/RC-FedBiT/results/cifar10_main.json"

    # Load existing or create new
    if os.path.exists(out_file):
        with open(out_file) as f:
            all_results = json.load(f)
    else:
        all_results = []

    # Remove old results for this method
    method_tag = args.ablation_tag or args.method
    all_results = [r for r in all_results if r["method"] != method_tag]
    all_results.extend(results)

    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
