#!/usr/bin/env python3
"""Auto-sequencer for RC-FedBiT experiments on seetacloud."""
import os, subprocess, time

PROJ = "/root/RC-FedBiT"
PYTHON = "/root/miniconda3/bin/python"
LOGS = f"{PROJ}/logs"
MAX_PARALLEL = 2  # Max concurrent GPU experiments

def count_rounds(log_file):
    if not os.path.exists(log_file):
        return 0
    with open(log_file) as f:
        return sum(1 for line in f if "| Acc:" in line)

def is_running(log_file):
    """Check if experiment is actively running based on log file recency."""
    if not os.path.exists(log_file):
        return False
    rounds = count_rounds(log_file)
    if rounds >= 100:
        return False
    mtime = os.path.getmtime(log_file)
    return (time.time() - mtime) < 180  # active in last 3 min

def count_active():
    logs = (
        [f"{LOGS}/rc_fedbit_main.log", f"{LOGS}/fedavg_main.log"] +
        [f"{LOGS}/ablation_{t}.log" for t in ["A0_binary","A1_rank1","A2_ca_rs","A3_nia_cva","A4_full"]] +
        [f"{LOGS}/noniid_{t}.log" for t in ["alpha_0.05","alpha_0.1","alpha_0.3","alpha_1.0","IID"]] +
        [f"{LOGS}/channel_snr_{s}dB.log" for s in [3,7,15,20]]
    )
    return sum(1 for lg in logs if is_running(lg))

def start_exp(script, args, log_file):
    if count_rounds(log_file) >= 100:
        return False
    if is_running(log_file):
        return False
    cmd = (f"cd {PROJ} && nohup {PYTHON} experiments/{script} "
           f"{args} > {log_file} 2>&1 &")
    os.system(cmd)
    print(f"Started: {script} {args[:40]}...", flush=True)
    time.sleep(5)  # Give it time to start
    return True

def main():
    print(f"[Seqr] {time.strftime('%H:%M:%S')}", flush=True)
    
    rc = count_rounds(f"{LOGS}/rc_fedbit_main.log")
    fa = count_rounds(f"{LOGS}/fedavg_main.log")
    print(f"Phase2: rc={rc}/100 fa={fa}/100", flush=True)
    
    tags = ["A0_binary", "A1_rank1", "A2_ca_rs", "A3_nia_cva", "A4_full"]
    abl = [count_rounds(f"{LOGS}/ablation_{t}.log") for t in tags]
    abl_done = sum(1 for r in abl if r >= 100)
    print(f"Phase3 ablation: {abl_done}/5 done, rounds={abl}", flush=True)
    
    noniid_alphas = [0.05, 0.1, 0.3, 1.0, 100]
    noniid_tags = [("IID" if a >= 100 else f"alpha_{a}") for a in noniid_alphas]
    noniid_rounds = [count_rounds(f"{LOGS}/noniid_{t}.log") for t in noniid_tags]
    noniid_done = sum(1 for r in noniid_rounds if r >= 100)
    print(f"Phase4 noniid: {noniid_done}/5 done", flush=True)
    
    channel_snrs = [3, 7, 15, 20]
    ch_rounds = [count_rounds(f"{LOGS}/channel_snr_{s}dB.log") for s in channel_snrs]
    ch_done = sum(1 for r in ch_rounds if r >= 100)
    print(f"Phase5 channel: {ch_done}/4 done", flush=True)
    
    active = count_active()
    print(f"Active: {active} (max={MAX_PARALLEL})", flush=True)
    
    # Phase 3: Start ablation after rc_fedbit >= 90 or done
    if rc >= 90:
        for level, tag in enumerate(tags):
            if active >= MAX_PARALLEL:
                break
            log = f"{LOGS}/ablation_{tag}.log"
            if abl[level] < 100:
                if start_exp("run_ablation.py",
                    f"--level {level} --rounds 100 --n_clients 100 --participation 0.1 --local_epochs 5",
                    log):
                    active += 1
    
    # Phase 4: Non-IID (start when rc_fedbit nearly done or ablation started)
    if rc >= 80:
        for i, alpha in enumerate(noniid_alphas):
            if active >= MAX_PARALLEL:
                break
            log = f"{LOGS}/noniid_{noniid_tags[i]}.log"
            if noniid_rounds[i] < 100:
                if start_exp("run_noniid.py",
                    f"--alpha {alpha} --rounds 100 --n_clients 100 --participation 0.1 --local_epochs 5",
                    log):
                    active += 1
    
    # Phase 5: Channel analysis (after ablation halfway done)
    if abl_done >= 2:
        for i, snr in enumerate(channel_snrs):
            if active >= MAX_PARALLEL:
                break
            log = f"{LOGS}/channel_snr_{snr}dB.log"
            if ch_rounds[i] < 100:
                if start_exp("run_channel.py",
                    f"--snr {snr} --rounds 100 --n_clients 100 --participation 0.1 --local_epochs 5",
                    log):
                    active += 1
    
    print(f"[Seqr] Done", flush=True)

if __name__ == "__main__":
    main()
