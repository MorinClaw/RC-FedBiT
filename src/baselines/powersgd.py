"""PowerSGD baseline: Rank-1 low-rank gradient compression without sign trick."""
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils.extmath import randomized_svd


class PowerSGDClient:
    def __init__(self, client_id, model, dataloader, config):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = config.get("device", "cuda")

    def _rank1_compress(self, delta_W):
        W_np = delta_W.detach().cpu().float().numpy()
        m, n = W_np.shape
        U, S, Vt = randomized_svd(W_np, n_components=1, random_state=42)
        sigma1 = S[0]
        p = U[:, 0] * np.sqrt(sigma1)  # left factor
        q = Vt[0, :] * np.sqrt(sigma1)  # right factor
        orig_bits = 32 * m * n
        comp_bits = 32 * (m + n)
        return torch.from_numpy(p).float(), torch.from_numpy(q).float(), orig_bits / max(comp_bits, 1)

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
            if d.dim() >= 2:
                W = d.view(d.shape[0], -1)
                p, q, cr = self._rank1_compress(W)
                payload[name] = {"type": "rank1", "p": p, "q": q, "shape": d.shape}
                m, n = W.shape
                total_bits += 32 * (m + n)
            else:
                payload[name] = {"type": "fp16", "data": d.half()}
                total_bits += d.numel() * 16
        return payload, 1.0, {"loss": total_loss, "comm_bits": total_bits}


class PowerSGDServer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def aggregate(self, client_payloads, client_weights):
        total_w = sum(client_weights)
        layer_updates = {}
        for payload, w in zip(client_payloads, client_weights):
            for name, data in payload.items():
                if name not in layer_updates:
                    layer_updates[name] = []
                layer_updates[name].append((data, w))

        agg = {}
        for name, updates in layer_updates.items():
            first = updates[0][0]
            if first["type"] == "fp16":
                agg[name] = sum(d["data"].float() * w for d, w in updates) / max(total_w, 1e-8)
            else:
                # Weighted sum of outer products
                delta_sum = sum(torch.outer(d["p"], d["q"]) * w for d, w in updates) / max(total_w, 1e-8)
                agg[name] = delta_sum

        lr = self.config.get("lr", 0.01)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in agg:
                    param.data -= lr * agg[name].to(param.device).view(param.shape)
