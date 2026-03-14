#!/usr/bin/env python3
"""每10分钟记录一次服务器状态"""
import subprocess, json, os
from datetime import datetime

LOG = "/root/RC-FedBiT/logs/monitor.json"
os.makedirs("/root/RC-FedBiT/logs", exist_ok=True)

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=10).strip()
    except:
        return "N/A"

data = {
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "gpu": run("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"),
    "processes": run("ps aux | grep python3 | grep -v grep | awk '{print $2, $3, $11}'"),
    "disk": run("df -h /root | tail -1"),
    "sanity_log": run("tail -5 /root/RC-FedBiT/logs/sanity.log 2>/dev/null"),
    "train_log": run("tail -3 /root/RC-FedBiT/logs/train.log 2>/dev/null"),
    "download_status": run("cat /root/RC-FedBiT/logs/download_progress.json 2>/dev/null"),
}

history = []
if os.path.exists(LOG):
    with open(LOG) as f:
        try: history = json.load(f)
        except: history = []
history.append(data)
history = history[-144:]  # 保留24小时 (144 × 10min)
with open(LOG, "w") as f:
    json.dump(history, f, indent=2, ensure_ascii=False)

print(f"[{data['time']}] GPU: {data['gpu'][:80]}")
print(f"  Processes: {data['processes'][:100]}")
print(f"  Train: {data['train_log'][:100]}")
