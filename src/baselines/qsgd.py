"""QSGD baseline: 4-bit stochastic quantization."""
import torch
import torch.nn as nn
import numpy as np


def qsgd_quantize(tensor, bits=4):
    """4-bit stochastic quantization."""
    max_val = tensor.abs().max() + 1e-8
    levels = 2 ** (bits - 1) - 1  # 7 for 4-bit
    norm = tensor / max_val
    # Stochastic rounding
    floored = (norm * levels).floor()
    prob = (norm * levels) - floored
    rnd = torch.bernoulli(prob.abs())
    quantized = (floored + rnd * norm.sign()).clamp(-levels, levels).to(torch.int8)
    return quantized, max_val, levels


def qsgd_dequantize(quantized, max_val, levels):
    return quantized.float() * (max_val / levels)


class QSGDClient:
    def __init__(self, client_id, model, dataloader, config):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = config.get("device", "cuda")
        self.bits = config.get("qsgd_bits", 4)

    def train(self, global_weights):
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        orig = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(),
                              lr=self.config.get("local_lr", 0.01), momentum=0.9)
        crit = nn.CrossEntropyLoss()
        total_loss = 0
        for _ in range(self.config.get("local_epochs", 5)):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(x), y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
        new_w = self.model.state_dict()
        delta = {k: orig[k] - new_w[k] for k in orig}
        payload = {}
        total_bits = 0
        for name, d in delta.items():
            q, scale, levels = qsgd_quantize(d.float(), self.bits)
            payload[name] = {"q": q, "scale": scale, "levels": levels}
            total_bits += d.numel() * self.bits + 32  # bits + scale
        return payload, 1.0, {"loss": total_loss, "comm_bits": total_bits}


class QSGDServer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def aggregate(self, client_payloads, client_weights):
        total_w = sum(client_weights)
        agg = {}
        for name in client_payloads[0]:
            deq = [qsgd_dequantize(p[name]["q"], p[name]["scale"], p[name]["levels"]) * w
                   for p, w in zip(client_payloads, client_weights)]
            agg[name] = sum(deq) / max(total_w, 1e-8)
        lr = self.config.get("lr", 0.01)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in agg:
                    param.data -= lr * agg[name].to(param.device).view(param.shape)
