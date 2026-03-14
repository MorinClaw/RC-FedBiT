#!/usr/bin/env python3
"""
Auto-sequencer: monitors running experiments and launches next phases.
Runs on seetacloud. Schedule to run every 10 minutes.
"""
import os, sys, json, subprocess, time

PROJ = "/root/RC-FedBiT"
PYTHON = "/root/miniconda3/bin/python"
LOGS = f"{PROJ}/logs"
RESULTS = f"{PROJ}/results"

def is_running(method_tag):
    """Check if a run_main/ablation/noniid/channel process is running for given tag."""
    import subprocess
    result = subprocess.run(["pgrep", "-f", method_tag], capture_output=True)
    return result.returncode == 0

def count_rounds(log_file):
    """Count completed rounds in a log file."""
    if not os.path.exists(log_file):
        return 0
    with open(log_file) as f:
        return sum(1 for line in f if " | Acc: " in line)

def phase2_done():
    """Check if Phase 2 main experiment is done."""
    rc = count_rounds(f"{LOGS}/rc_fedbit_main.log") >= 100
    fa = count_rounds(f"{LOGS}/fedavg_main.log") >= 100
    return rc and fa

def start_ablation(level):
    """Start ablation experiment for a given level."""
    tags = ["A0_binary", "A1_rank1", "A2_ca_rs", "A3_nia_cva", "A4_full"]
    tag = tags[level]
    log_file = f"{LOGS}/ablation_{tag}.log"
    if count_rounds(log_file) >= 100:
        print(f"Ablation {tag} already done")
        return
    if is_running(f"--level {level}"):
        print(f"Ablation {tag} already running")
        return
    cmd = (f"cd {PROJ} && nohup {PYTHON} experiments/run_ablation.py "
           f"--level {level} --rounds 100 --n_clients 100 --participation 0.1 "
           f"--local_epochs 5 > {log_file} 2>&1 &")
    os.system(cmd)
    print(f"Started ablation {tag}", flush=True)

def start_noniid(alpha):
    """Start non-IID analysis."""
    tag = f"IID" if alpha >= 100 else f"alpha_{alpha}"
    log_file = f"{LOGS}/noniid_{tag}.log"
    if count_rounds(log_file) >= 100:
        print(f"NonIID {tag} already done")
        return
    if is_running(f"--alpha {alpha}") and "noniid" in str(subprocess.run(["pgrep", "-af", f"--alpha {alpha}"], capture_output=True).stdout):
        print(f"NonIID {tag} already running")
        return
    cmd = (f"cd {PROJ} && nohup {PYTHON} experiments/run_noniid.py "
           f"--alpha {alpha} --rounds 100 --n_clients 100 --participation 0.1 "
           f"--local_epochs 5 > {log_file} 2>&1 &")
    os.system(cmd)
    print(f"Started NonIID {tag}", flush=True)

def start_channel(snr):
    """Start channel analysis."""
    tag = f"snr_{int(snr)}dB"
    log_file = f"{LOGS}/channel_{tag}.log"
    if count_rounds(log_file) >= 100:
        print(f"Channel {tag} already done")
        return
    cmd = (f"cd {PROJ} && nohup {PYTHON} experiments/run_channel.py "
           f"--snr {snr} --rounds 100 --n_clients 100 --participation 0.1 "
           f"--local_epochs 5 > {log_file} 2>&1 &")
    os.system(cmd)
    print(f"Started Channel {tag}", flush=True)

def count_active_experiments():
    result = subprocess.run(["pgrep", "-c", "-f", "run_main\|run_ablation\|run_noniid\|run_channel"], capture_output=True)
    try:
        return int(result.stdout.decode().strip())
    except:
        return 0

def main():
    print(f"[Sequencer] Checking status at {time.strftime('%H:%M:%S')}...")
    
    # Check Phase 2 status
    rc_rounds = count_rounds(f"{LOGS}/rc_fedbit_main.log")
    fa_rounds = count_rounds(f"{LOGS}/fedavg_main.log")
    print(f"Phase 2: rc_fedbit={rc_rounds}/100, fedavg={fa_rounds}/100")
    
    # Check ablation status  
    tags = ["A0_binary", "A1_rank1", "A2_ca_rs", "A3_nia_cva", "A4_full"]
    ablation_done = [count_rounds(f"{LOGS}/ablation_{t}.log") >= 100 for t in tags]
    print(f"Phase 3 ablation: {sum(ablation_done)}/5 done")
    
    active = count_active_experiments()
    print(f"Active experiments: {active}")
    
    # Phase 3: Start ablation if Phase 2 rc_fedbit is done or nearly done
    if rc_rounds >= 90:
        for level in range(5):
            if not ablation_done[level] and active < 3:
                start_ablation(level)
                active += 1
    
    # Check Phase 4 status
    noniid_alphas = [0.05, 0.1, 0.3, 0.5, 1.0, 100]
    noniid_tags = [f"IID" if a >= 100 else f"alpha_{a}" for a in noniid_alphas]
    noniid_done = [count_rounds(f"{LOGS}/noniid_{t}.log") >= 100 for t in noniid_tags]
    print(f"Phase 4 noniid: {sum(noniid_done)}/6 done")
    
    # Phase 4: Start after phase 3 is progressing
    if sum(ablation_done) >= 2 or rc_rounds >= 90:
        for alpha in noniid_alphas:
            if active < 4:
                tag = "IID" if alpha >= 100 else f"alpha_{alpha}"
                if count_rounds(f"{LOGS}/noniid_{tag}.log") < 100:
                    start_noniid(alpha)
                    active += 1
    
    # Check Phase 5 status
    snr_values = [3, 7, 10, 15, 20]
    channel_done = [count_rounds(f"{LOGS}/channel_snr_{s}dB.log") >= 100 for s in snr_values]
    print(f"Phase 5 channel: {sum(channel_done)}/5 done")
    
    # Phase 5: Start after some ablation done
    if sum(ablation_done) >= 3 and active < 3:
        for snr in snr_values:
            if count_rounds(f"{LOGS}/channel_snr_{snr}dB.log") < 100 and active < 4:
                start_channel(snr)
                active += 1
    
    print(f"[Sequencer] Done. Active now: {count_active_experiments()}")

if __name__ == "__main__":
    main()
