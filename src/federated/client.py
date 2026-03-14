import torch
import torch.nn as nn
from src.compression.rank1_compress import Rank1GradientCompressor
from src.channel.channel_adaptive import ChannelAdaptiveSelector

class FedBiTClient:
    def __init__(self, client_id, model, dataloader, config):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.compressor = Rank1GradientCompressor()
        self.ca = ChannelAdaptiveSelector(
            gamma_high=config.get("gamma_high",15.0),
            gamma_low=config.get("gamma_low",5.0),
            total_rounds=config.get("total_rounds",100))
        self.device = config.get("device","cuda")

    def train(self, global_weights, snr_db, t=0):
        self.model.load_state_dict(global_weights)
        self.model.to(self.device)
        orig = {k:v.clone() for k,v in self.model.state_dict().items()}
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(),
                              lr=self.config.get("local_lr",0.01), momentum=0.9)
        crit = nn.CrossEntropyLoss()
        total_loss = 0
        for _ in range(self.config.get("local_epochs",5)):
            for x, y in self.dataloader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = crit(self.model(x), y)
                loss.backward(); opt.step()
                total_loss += loss.item()
        new_w = self.model.state_dict()
        delta = {k: orig[k]-new_w[k] for k in orig}
        compressed, cr = self.compressor.compress_model(delta, device=self.device)
        payload = {}
        for name, data in compressed.items():
            if data["type"] == "rank1":
                lp, wt = self.ca.select_payload(snr_db, data["B"], data["h1"], data["h2"], t)
                payload[name] = lp
            else:
                payload[name] = data
        return payload, wt, {"loss": total_loss, "cr": cr, "snr_db": snr_db}
