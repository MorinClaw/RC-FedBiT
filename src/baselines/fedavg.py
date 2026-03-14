"""FedAvg baseline: No compression, full FP32 gradient transmission."""
import torch
import torch.nn as nn


class FedAvgClient:
    def __init__(self, client_id, model, dataloader, config):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.device = config.get("device", "cuda")

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
        # Comm bits: 32 bits per float parameter
        n_params = sum(v.numel() for v in delta.values())
        comm_bits = n_params * 32
        return delta, 1.0, {"loss": total_loss, "comm_bits": comm_bits}


class FedAvgServer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def aggregate(self, client_deltas, client_weights):
        total_w = sum(client_weights)
        agg = {}
        for name in client_deltas[0]:
            agg[name] = sum(d[name].float() * w for d, w in zip(client_deltas, client_weights)) / max(total_w, 1e-8)
        lr = self.config.get("lr", 0.01)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in agg:
                    param.data -= lr * agg[name].to(param.device).view(param.shape)
