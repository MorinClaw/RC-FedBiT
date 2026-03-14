#!/usr/bin/env python3
"""
RC-FedBiT Ablation Study
A0: Binary-only (no rank1, no CA-RS, no NIA-CVA, no AFC)
A1: + Rank1 compensation (no CA-RS: always fp32, no NIA-CVA, no AFC)
A2: + CA-RS (no NIA-CVA, no AFC)
A3: + NIA-CVA (no AFC)
A4: + AFC = Full RC-FedBiT
"""
import sys, os, json, time, argparse
sys.path.insert(0, "/root/RC-FedBiT")

import torch, timm
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import defaultdict

from src.data.partition import dirichlet_partition
from src.compression.rank1_compress import Rank1GradientCompressor
from src.channel.channel_adaptive import ChannelAdaptiveSelector, RayleighChannelSimulator

def make_model(device, num_classes=10):
    m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    return m.to(device)

def eval_model(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
    return 100.0*correct/total if total > 0 else 0.0

class AblationClient:
    def __init__(self, client_id, model, loader, config, ablation_level):
        self.client_id = client_id
        self.model = model
        self.loader = loader
        self.config = config
        self.level = ablation_level  # 0..4
        self.device = config["device"]
        self.compressor = Rank1GradientCompressor()
        self.ca = ChannelAdaptiveSelector(
            gamma_high=config.get("gamma_high",15.0),
            gamma_low=config.get("gamma_low",5.0),
            total_rounds=config.get("total_rounds",100))

    def train(self, global_weights, snr_db, t):
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        orig = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(),
                              lr=self.config.get("local_lr",0.01), momentum=0.9)
        crit = nn.CrossEntropyLoss()
        total_loss = 0
        for _ in range(self.config.get("local_epochs",5)):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(x), y)
                loss.backward(); opt.step()
                total_loss += loss.item()
        new_w = self.model.state_dict()
        delta = {k: orig[k]-new_w[k] for k in orig}

        payload = {}
        cum_bits = 0
        for name, d in delta.items():
            if d.dim() >= 2:
                W = d.view(d.shape[0], -1)
                m_dim, n_dim = W.shape
                if self.level == 0:
                    # A0: Binary only
                    B = torch.sign(W); B[B==0] = 1
                    payload[name] = {"B": B.to(torch.int8), "mode": "binary_only", "shape": d.shape}
                    cum_bits += m_dim * n_dim  # 1 bit per param
                else:
                    B, h1, h2, meta = self.compressor.compress(W)
                    if self.level == 1:
                        # A1: + Rank1, always fp32, no CA-RS
                        payload[name] = {"B": B, "h1": h1.float(), "h2": h2.float(), "mode": "rank1_fp32", "shape": d.shape}
                        cum_bits += m_dim*n_dim + 32*(m_dim+n_dim)
                    else:
                        # A2+: + CA-RS
                        lp, wt = self.ca.select_payload(snr_db, B, h1, h2, t)
                        lp["shape"] = d.shape
                        payload[name] = lp
                        mode = lp.get("mode","binary_only")
                        if mode == "rank1_fp32":
                            cum_bits += m_dim*n_dim + 32*(m_dim+n_dim)
                        elif mode == "rank1_int8":
                            cum_bits += m_dim*n_dim + 8*(m_dim+n_dim)
                        else:
                            cum_bits += m_dim*n_dim
            else:
                payload[name] = {"type": "fp16", "data": d.half(), "shape": d.shape}
                cum_bits += d.numel() * 16

        wt = 1.0
        return payload, wt, {"loss": total_loss, "comm_bits": cum_bits}


class AblationServer:
    def __init__(self, model, config, ablation_level):
        self.model = model
        self.config = config
        self.level = ablation_level  # 0..4
        self.prev_h1 = {}
        self.round = 0

    def aggregate(self, payloads, weights):
        layer_updates = defaultdict(list)
        for payload, w in zip(payloads, weights):
            for name, data in payload.items():
                layer_updates[name].append((data, w))

        global_delta = {}
        for name, updates in layer_updates.items():
            global_delta[name] = self._aggregate_layer(name, updates)

        # A4: AFC calibration
        if self.level >= 4:
            beta0, lam = 0.1, 0.05
            bt = beta0 * np.exp(-lam * self.round)
            if bt >= 0.01:
                global_delta = {k: (1-bt)*v for k,v in global_delta.items()}

        lr = self.config.get("lr", 0.01)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_delta:
                    param.data -= lr * global_delta[name].to(param.device).view(param.shape)
        self.round += 1

    def _aggregate_layer(self, name, updates):
        first = updates[0][0]
        if first.get("type") == "fp16":
            tw = sum(w for _, w in updates)
            return sum(d["data"].float()*w for d,w in updates) / max(tw, 1e-8)

        binary_sum = None
        h1s, h2s, ws = [], [], []
        for data, w in updates:
            B = data["B"].float()
            binary_sum = w*B if binary_sum is None else binary_sum + w*B
            if data.get("mode","binary_only") != "binary_only" and "h1" in data:
                h1 = data["h1"].float()
                h2 = data["h2"].float()
                if data.get("mode") == "rank1_int8":
                    h1 = h1 * data.get("h1_scale", torch.tensor(1.0)).float()
                    h2 = h2 * data.get("h2_scale", torch.tensor(1.0)).float()
                h1s.append(h1); h2s.append(h2); ws.append(w)

        B_agg = torch.sign(binary_sum); B_agg[B_agg==0] = 1

        if not h1s:
            return B_agg

        # A3+: NIA-CVA
        if self.level >= 3 and name in self.prev_h1:
            prev = self.prev_h1[name]
            aw = [abs(float(torch.dot(h, prev)/(h.norm()*prev.norm()+1e-8))) for h in h1s]
            fw = [w*a for w,a in zip(ws, aw)]
        else:
            fw = list(ws)

        tw = sum(fw) + 1e-8
        h1g = sum(fw[i]*h1s[i] for i in range(len(h1s))) / tw
        h2g = sum(fw[i]*h2s[i] for i in range(len(h2s))) / tw
        if self.level >= 3:
            self.prev_h1[name] = h1g.detach()

        return B_agg * torch.outer(h1g, h2g)


def run_ablation(level, tag, args, global_model, client_loaders, test_loader, device, rng):
    config = {
        "device": device, "lr": args.lr, "local_lr": args.local_lr,
        "local_epochs": args.local_epochs, "gamma_high": 15.0, "gamma_low": 5.0,
        "total_rounds": args.rounds, "mean_snr_db": args.mean_snr_db,
    }
    server = AblationServer(global_model, config, level)
    ch_sim = RayleighChannelSimulator(mean_snr_db=args.mean_snr_db, seed=args.seed)
    results = []; cum_bits = 0

    for rnd in range(args.rounds):
        gw = {k: v.cpu().clone() for k,v in global_model.state_dict().items()}
        n_part = max(1, int(args.n_clients * args.participation))
        selected = rng.choice(args.n_clients, n_part, replace=False)
        snr_vals = ch_sim.sample_snr(n_part)
        payloads, wts = [], []
        for i, k in enumerate(selected):
            c_model = make_model(device)
            c = AblationClient(k, c_model, client_loaders[k], config, level)
            payload, wt, stats = c.train(gw, snr_vals[i], rnd)
            payloads.append(payload); wts.append(len(client_loaders[k].dataset))
            cum_bits += stats["comm_bits"]
        server.aggregate(payloads, wts)
        acc = eval_model(global_model, test_loader, device)
        results.append({"method": tag, "round": rnd+1, "acc": acc, "comm_bits": cum_bits})
        print(f"[{tag}] Round {rnd+1}/{args.rounds} | Acc: {acc:.2f}% | CumBits: {cum_bits:.2e}", flush=True)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", type=int, required=True, choices=[0,1,2,3,4])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--n_clients", type=int, default=100)
    p.add_argument("--participation", type=float, default=0.1)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--mean_snr_db", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    tags = ["A0_binary", "A1_rank1", "A2_ca_rs", "A3_nia_cva", "A4_full"]
    tag = tags[args.level]
    print(f"Ablation {tag} (level={args.level}), Rounds={args.rounds}", flush=True)

    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform)
    test_ds  = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=False, download=False, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    client_datasets = dirichlet_partition(train_ds, args.n_clients, alpha=0.5, seed=args.seed)
    client_loaders = [DataLoader(ds, batch_size=64, shuffle=True, num_workers=2) for ds in client_datasets]

    global_model = make_model(DEVICE)
    t0 = time.time()
    results = run_ablation(args.level, tag, args, global_model, client_loaders, test_loader, DEVICE, rng)
    elapsed = time.time() - t0
    print(f"Done in {elapsed/3600:.2f}h | Final acc: {results[-1]['acc']:.2f}%", flush=True)

    os.makedirs("/root/RC-FedBiT/results", exist_ok=True)
    out_file = "/root/RC-FedBiT/results/ablation.json"
    if os.path.exists(out_file):
        with open(out_file) as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results = [r for r in all_results if r["method"] != tag]
    all_results.extend(results)
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_file}", flush=True)


if __name__ == "__main__":
    main()
