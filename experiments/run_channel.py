#!/usr/bin/env python3
"""Channel Adaptive Analysis: mean_snr_db in {3, 7, 10, 15, 20} dB"""
import sys, os, json, time, argparse
sys.path.insert(0, "/root/RC-FedBiT")

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.partition import dirichlet_partition
from src.channel.channel_adaptive import RayleighChannelSimulator
from src.federated.server import FedBiTServer
from src.federated.client import FedBiTClient

def make_model(device, nc=10):
    return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=nc).to(device)

def eval_model(model, loader, device):
    model.eval(); c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device); pred=model(x).argmax(1)
            c+=(pred==y).sum().item(); t+=y.size(0)
    return 100.0*c/t if t>0 else 0.0

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snr", type=float, required=True)
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--participation", type=float, default=0.1)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tag = f"snr_{int(args.snr)}dB"
    print(f"Channel analysis: mean_SNR={args.snr}dB", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3,[0.5]*3)
    ])
    train_ds = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform)
    test_ds  = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    client_datasets = dirichlet_partition(train_ds, args.n_clients, alpha=0.5, seed=args.seed)
    client_loaders = [DataLoader(ds, batch_size=64, shuffle=True, num_workers=0) for ds in client_datasets]

    config = {"device": DEVICE, "lr": args.lr, "local_lr": args.local_lr, "local_epochs": args.local_epochs,
              "gamma_high": 15.0, "gamma_low": 5.0, "total_rounds": args.rounds, "mean_snr_db": args.snr}
    global_model = make_model(DEVICE)
    server = FedBiTServer(global_model, config)
    ch_sim = RayleighChannelSimulator(mean_snr_db=args.snr, seed=args.seed)
    results = []; cum_bits = 0

    for rnd in range(args.rounds):
        gw = {k: v.cpu().clone() for k,v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients*args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        snr_vals = ch_sim.sample_snr(n_part)
        payloads, wts = [], []
        for i,k in enumerate(selected):
            c_model = make_model(DEVICE)
            c = FedBiTClient(k, c_model, client_loaders[k], config)
            payload, wt, stats = c.train(gw, snr_vals[i], rnd)
            payloads.append(payload); wts.append(len(client_loaders[k].dataset))
            for nm, dat in payload.items():
                mode = dat.get("mode", dat.get("type",""))
                if mode == "rank1_fp32":
                    pp = global_model.state_dict()[nm]; W=pp.view(pp.shape[0],-1); m,n=W.shape
                    cum_bits += m*n+32*(m+n)
                elif mode == "rank1_int8":
                    pp = global_model.state_dict()[nm]; W=pp.view(pp.shape[0],-1); m,n=W.shape
                    cum_bits += m*n+8*(m+n)
                elif mode == "binary_only":
                    cum_bits += global_model.state_dict()[nm].numel()
                else:
                    cum_bits += global_model.state_dict()[nm].numel()*16
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device=DEVICE)
        results.append({"method": f"rc_fedbit_{tag}", "snr_db": args.snr, "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}%", flush=True)

    out_file = "/root/RC-FedBiT/results/channel_analysis.json"
    os.makedirs("/root/RC-FedBiT/results", exist_ok=True)
    if os.path.exists(out_file):
        with open(out_file) as f: all_results = json.load(f)
    else:
        all_results = []
    method_tag = results[0]["method"] if results else ""
    all_results = [r for r in all_results if r.get("method") != method_tag]
    all_results.extend(results)
    with open(out_file,"w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_file}", flush=True)

if __name__ == "__main__":
    main()
