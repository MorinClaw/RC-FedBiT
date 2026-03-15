#!/usr/bin/env python3
"""
RC-FedBiT Multi-Dataset Experiment Runner
Supports: cifar10, cifar100, cub200, imagenet100
Supports: variable n_clients, participation rate
"""
import copy, sys, os, argparse, json, time
sys.path.insert(0, "/root/RC-FedBiT")

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image

from src.data.partition import dirichlet_partition, iid_partition

# ─── Dataset configs ───────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10, "img_size": 32, "patch_size": 4,
        "model": "vit_tiny_patch16_224",
        "data_root": "/root/RC-FedBiT/data/cifar10",
        "mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616],
    },
    "cifar100": {
        "num_classes": 100, "img_size": 32, "patch_size": 4,
        "model": "vit_tiny_patch16_224",
        "data_root": "/root/RC-FedBiT/data/cifar100",
        "mean": [0.5071, 0.4867, 0.4408], "std": [0.2675, 0.2565, 0.2761],
    },
    "cub200": {
        "num_classes": 200, "img_size": 224, "patch_size": 16,
        "model": "vit_small_patch16_224",
        "data_root": "/root/RC-FedBiT/data/cub200/CUB_200_2011/images",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    },
    "imagenet100": {
        "num_classes": 100, "img_size": 224, "patch_size": 16,
        "model": "vit_small_patch16_224",
        "data_root": "/root/RC-FedBiT/data/imagenet100/imagenet100",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["fedavg", "signsgd", "qsgd", "powersgd", "rc_fedbit"])
    p.add_argument("--dataset", default="cifar10",
                   choices=list(DATASET_CONFIGS.keys()))
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
    p.add_argument("--no_rank1", action="store_true")
    p.add_argument("--no_ca_rs", action="store_true")
    p.add_argument("--no_nia_cva", action="store_true")
    p.add_argument("--no_afc", action="store_true")
    p.add_argument("--ablation_tag", type=str, default=None)
    return p.parse_args()


def make_model(device, cfg):
    m = timm.create_model(
        cfg["model"],
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        pretrained=False,
        num_classes=cfg["num_classes"],
    )
    return m.to(device)


def load_dataset(dataset_name, cfg):
    """Load train+test datasets for the given dataset."""
    mean, std = cfg["mean"], cfg["std"]
    img_size = cfg["img_size"]

    if img_size == 32:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    root = cfg["data_root"]

    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test_ds  = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    elif dataset_name == "cifar100":
        train_ds = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test_ds  = datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
    elif dataset_name in ("cub200", "imagenet100"):
        # Expect ImageFolder structure: root/train/, root/test/ (or root/val/)
        train_root = os.path.join(root, "train")
        test_root  = os.path.join(root, "test")
        if not os.path.isdir(test_root):
            test_root = os.path.join(root, "val")
        if not os.path.isdir(train_root):
            # Flat ImageFolder (all in root)
            full_ds = datasets.ImageFolder(root, transform=train_tf)
            n = len(full_ds)
            idx = list(range(n))
            np.random.seed(42)
            np.random.shuffle(idx)
            split = int(0.8 * n)
            train_ds = Subset(full_ds, idx[:split])
            full_ds_test = datasets.ImageFolder(root, transform=test_tf)
            test_ds = Subset(full_ds_test, idx[split:])
        else:
            train_ds = datasets.ImageFolder(train_root, transform=train_tf)
            test_ds  = datasets.ImageFolder(test_root,  transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_ds, test_ds


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


# ─── Runner helpers ────────────────────────────────────────────────────────

def run_rc_fedbit(args, global_model, client_loaders, test_loader, device, rng):
    from src.compression.rank1_compress import Rank1GradientCompressor
    from src.channel.channel_adaptive import ChannelAdaptiveSelector, RayleighChannelSimulator
    from src.federated.server import FedBiTServer
    from src.federated.client import FedBiTClient

    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs,
        "gamma_high": 15.0, "gamma_low": 5.0,
        "total_rounds": args.rounds, "mean_snr_db": args.mean_snr_db,
    }
    server = FedBiTServer(global_model, config)
    ch_sim = RayleighChannelSimulator(mean_snr_db=args.mean_snr_db, seed=args.seed)
    results, cum_bits = [], 0

    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        snr_vals = ch_sim.sample_snr(n_part)
        payloads, wts = [], []
        round_bits = 0
        for i, k in enumerate(selected):
            c_model = copy.deepcopy(global_model)
            c = FedBiTClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights, snr_vals[i], rnd)
            payloads.append(payload)
            wts.append(wt * len(client_loaders[k].dataset))
            for name, data in payload.items():
                mode = data.get("mode", data.get("type", ""))
                p = global_model.state_dict()[name]
                W = p.view(p.shape[0], -1); m, n = W.shape
                if mode == "rank1_fp32":   round_bits += m * n + 32 * (m + n)
                elif mode == "rank1_int8": round_bits += m * n + 8 * (m + n)
                elif mode == "binary_only": round_bits += p.numel()
                elif mode == "fp16":       round_bits += p.numel() * 16
        cum_bits += round_bits
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        tag = args.ablation_tag or "rc_fedbit"
        results.append({"method": tag, "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)
    return results


def run_fedavg(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.fedavg import FedAvgClient, FedAvgServer
    config = {"device": device, "lr": args.lr, "local_lr": args.local_lr, "local_epochs": args.local_epochs}
    server = FedAvgServer(global_model, config)
    results, cum_bits = [], 0
    n_params = count_params(global_model)
    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        deltas, wts = [], []
        for k in selected:
            c_model = copy.deepcopy(global_model)
            c = FedAvgClient(k, c_model, client_loaders[k], config)
            delta, wt, stats = c.train(global_weights)
            deltas.append(delta); wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params * 32
        server.aggregate(deltas, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "fedavg", "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[fedavg] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)
    return results


def run_signsgd(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.signsgd import SignSGDClient, SignSGDServer
    config = {"device": device, "lr": args.lr, "local_lr": args.local_lr, "local_epochs": args.local_epochs}
    server = SignSGDServer(global_model, config)
    results, cum_bits = [], 0
    n_params = count_params(global_model)
    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for k in selected:
            c_model = copy.deepcopy(global_model)
            c = SignSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights)
            payloads.append(payload); wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "signsgd", "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[signsgd] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)
    return results


def run_qsgd(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.qsgd import QSGDClient, QSGDServer
    config = {"device": device, "lr": args.lr, "local_lr": args.local_lr,
              "local_epochs": args.local_epochs, "qsgd_bits": 4}
    server = QSGDServer(global_model, config)
    results, cum_bits = [], 0
    n_params = count_params(global_model)
    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for k in selected:
            c_model = copy.deepcopy(global_model)
            c = QSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights)
            payloads.append(payload); wts.append(len(client_loaders[k].dataset))
            cum_bits += n_params * 4
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "qsgd", "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[qsgd] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)
    return results


def run_powersgd(args, global_model, client_loaders, test_loader, device, rng):
    from src.baselines.powersgd import PowerSGDClient, PowerSGDServer
    config = {"device": device, "lr": args.lr, "local_lr": args.local_lr, "local_epochs": args.local_epochs}
    server = PowerSGDServer(global_model, config)
    results, cum_bits = [], 0
    for rnd in range(args.rounds):
        global_weights = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        payloads, wts = [], []
        for k in selected:
            c_model = copy.deepcopy(global_model)
            c = PowerSGDClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(global_weights)
            payloads.append(payload); wts.append(len(client_loaders[k].dataset))
            cum_bits += stats["comm_bits"]
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": "powersgd", "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[powersgd] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)
    return results


RUNNERS = {
    "fedavg": run_fedavg, "signsgd": run_signsgd,
    "qsgd": run_qsgd, "powersgd": run_powersgd, "rc_fedbit": run_rc_fedbit,
}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DATASET_CONFIGS[args.dataset]
    tag = args.ablation_tag or args.method
    print(f"Device: {DEVICE}, Method: {tag}, Dataset: {args.dataset}, "
          f"n_clients: {args.n_clients}, participation: {args.participation}, "
          f"Alpha: {args.alpha}, Rounds: {args.rounds}", flush=True)

    train_ds, test_ds = load_dataset(args.dataset, cfg)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    if args.alpha >= 100:
        client_datasets = iid_partition(train_ds, args.n_clients, seed=args.seed)
        print(f"IID partition: {args.n_clients} clients", flush=True)
    else:
        client_datasets = dirichlet_partition(train_ds, args.n_clients, alpha=args.alpha, seed=args.seed)
        print(f"Dirichlet(alpha={args.alpha}) partition: {args.n_clients} clients", flush=True)

    client_loaders = [
        DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
        for ds in client_datasets
    ]

    print("Creating global model...", flush=True)
    global_model = make_model(DEVICE, cfg)
    print(f"Global model ready. Params: {count_params(global_model)/1e6:.2f}M", flush=True)

    t0 = time.time()
    results = RUNNERS[args.method](args, global_model, client_loaders, test_loader, DEVICE, rng)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/3600:.2f}h | Final acc: {results[-1]['acc']:.2f}%", flush=True)

    # Output file
    out_file = args.output or f"/root/RC-FedBiT/results/{args.dataset}_{args.n_clients}c_{int(args.participation*100)}p.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    existing = []
    if os.path.exists(out_file):
        with open(out_file) as f:
            try: existing = json.load(f)
            except: existing = []
    existing = [r for r in existing if r["method"] != tag]
    existing.extend(results)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
