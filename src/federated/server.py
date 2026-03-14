import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np
from src.compression.rank1_compress import Rank1GradientCompressor
from src.channel.channel_adaptive import RayleighChannelSimulator

class FedBiTServer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.compressor = Rank1GradientCompressor()
        self.channel_sim = RayleighChannelSimulator(mean_snr_db=config.get("mean_snr_db", 10.0))
        self.prev_global_h1 = {}
        self.round = 0

    def aggregate(self, client_payloads, client_weights):
        layer_updates = defaultdict(list)
        for payload, weight in zip(client_payloads, client_weights):
            for name, data in payload.items():
                layer_updates[name].append((data, weight))

        global_delta = {}
        for name, updates in layer_updates.items():
            global_delta[name] = self._aggregate_layer(name, updates)

        global_delta = self._afc_calibrate(global_delta)

        lr = self.config.get("lr", 0.01)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_delta:
                    param.data -= lr * global_delta[name].to(param.device).view(param.shape)
        self.round += 1

    def _aggregate_layer(self, name, updates):
        # Check type of first update
        first_data = updates[0][0]

        # fp16 layers (bias, etc): simple weighted average
        if first_data.get("type") == "fp16":
            total_w = sum(w for _, w in updates)
            agg = sum(d["data"].float() * w for d, w in updates) / max(total_w, 1e-8)
            return agg

        # Rank-1 / binary layers
        # Key fix: compute weighted average of DECOMPRESSED gradients
        # (B_i * outer(h1_i, h2_i)) weighted by w_i
        # This is mathematically correct; separating B and h aggregation is wrong.
        grad_sum = None
        total_w = 0.0
        h1s, h2s, ws = [], [], []

        for data, w in updates:
            B = data["B"].float()
            if data.get("mode", "") == "binary_only":
                # No rank-1: use B only (scaled by mean abs gradient est = 1.0/sqrt(numel))
                scale = 1.0 / (B.numel() ** 0.5 + 1e-8)
                grad_i = B * scale
            else:
                h1 = data["h1"].float()
                h2 = data["h2"].float()
                if data.get("mode") == "rank1_int8":
                    h1 = h1 * data.get("h1_scale", torch.tensor(1.0)).float()
                    h2 = h2 * data.get("h2_scale", torch.tensor(1.0)).float()
                # Decompressed gradient: B ⊙ (h1 ⊗ h2)
                grad_i = B * torch.outer(h1, h2)
                h1s.append(h1); h2s.append(h2); ws.append(w)

            grad_sum = w * grad_i if grad_sum is None else grad_sum + w * grad_i
            total_w += w

        agg = grad_sum / max(total_w, 1e-8)

        # NIA-CVA: update prev_global_h1 for next round alignment tracking
        if h1s:
            self._nia_cva(name, h1s, h2s, ws)  # updates self.prev_global_h1

        return agg

    def _nia_cva(self, name, h1s, h2s, ws):
        if name in self.prev_global_h1:
            prev = self.prev_global_h1[name]
            aw = [abs(float(torch.dot(h, prev) / (h.norm() * prev.norm() + 1e-8))) for h in h1s]
        else:
            aw = [1.0] * len(h1s)
        fw = [w * a for w, a in zip(ws, aw)]
        tw = sum(fw) + 1e-8
        h1g = sum(w * h for w, h in zip(fw, h1s)) / tw
        h2g = sum(w * h for w, h in zip(fw, h2s)) / tw
        self.prev_global_h1[name] = h1g.detach()
        return h1g, h2g

    def _afc_calibrate(self, delta, beta0=0.1, lam=0.05):
        bt = beta0 * np.exp(-lam * self.round)
        if bt < 0.01:
            return delta
        return {k: (1 - bt) * v for k, v in delta.items()}
