#!/usr/bin/env python3
"""
Sequential experiment launcher for multi-dataset runs.
Runs all method x dataset combinations in order.
Usage: python3 run_seq.py --server seeta|matpool
"""
import subprocess, os, sys, time

PYTHON = sys.argv[2] if len(sys.argv) > 2 else "python3"
SERVER = sys.argv[1] if len(sys.argv) > 1 else "seeta"

BASE = ["--rounds", "100", "--alpha", "0.5"]

# seeta: cifar100, cub200 (has data)
# matpool: client settings experiments (after non-IID done)
SEETA_JOBS = [
    # CIFAR-100
    *[(m, ["--dataset","cifar100"]+BASE) for m in ["fedavg","rc_fedbit","signsgd","qsgd","powersgd"]],
    # CUB-200
    *[(m, ["--dataset","cub200"]+BASE) for m in ["fedavg","rc_fedbit","signsgd","qsgd","powersgd"]],
    # ImageNet-100
    *[(m, ["--dataset","imagenet100"]+BASE) for m in ["fedavg","rc_fedbit","signsgd","qsgd","powersgd"]],
    # Client setting: 50 clients
    *[(m, ["--dataset","cifar10","--n_clients","50","--participation","0.1"]+BASE)
      for m in ["fedavg","rc_fedbit","signsgd"]],
    # Client setting: 200 clients
    *[(m, ["--dataset","cifar10","--n_clients","200","--participation","0.05"]+BASE)
      for m in ["fedavg","rc_fedbit","signsgd"]],
]

MATPOOL_JOBS = [
    # Client settings on cifar10
    *[(m, ["--dataset","cifar10","--n_clients","50","--participation","0.1"]+BASE)
      for m in ["fedavg","rc_fedbit","signsgd","qsgd","powersgd"]],
    *[(m, ["--dataset","cifar10","--n_clients","200","--participation","0.05"]+BASE)
      for m in ["fedavg","rc_fedbit","signsgd","qsgd","powersgd"]],
    # High participation
    *[(m, ["--dataset","cifar10","--n_clients","100","--participation","0.2"]+BASE)
      for m in ["fedavg","rc_fedbit","signsgd","qsgd","powersgd"]],
]

JOBS = SEETA_JOBS if SERVER == "seeta" else MATPOOL_JOBS

print(f"Server: {SERVER}, Total jobs: {len(JOBS)}")
for i, (method, extra_args) in enumerate(JOBS):
    dataset = next((extra_args[j+1] for j,a in enumerate(extra_args) if a=="--dataset"), "cifar10")
    n_c     = next((extra_args[j+1] for j,a in enumerate(extra_args) if a=="--n_clients"), "100")
    par     = next((extra_args[j+1] for j,a in enumerate(extra_args) if a=="--participation"), "0.1")
    log = f"/root/RC-FedBiT/logs/{dataset}_{n_c}c_{par}_{method}.log"
    cmd = [PYTHON, "/root/RC-FedBiT/experiments/run_multi.py",
           "--method", method] + extra_args
    print(f"\n[{i+1}/{len(JOBS)}] {method} | {dataset} | {n_c}c {par}p → {log}", flush=True)
    with open(log, "w") as lf:
        ret = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             cwd="/root/RC-FedBiT",
                             env={**os.environ, "HF_HUB_OFFLINE": "1"})
    print(f"  → exit code {ret.returncode}", flush=True)
    time.sleep(3)

print("\n=== ALL DONE ===")
