
import sys, time
sys.path.insert(0, "/root/RC-FedBiT")
import torch, timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.data.partition import dirichlet_partition
from src.federated.client import FedBiTClient

DEVICE = "cuda"
transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
train_ds = datasets.CIFAR10("/root/RC-FedBiT/data/cifar10", train=True, download=False, transform=transform)
client_datasets = dirichlet_partition(train_ds, 100, alpha=0.5, seed=42)
client_loaders = [DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True) for ds in client_datasets]

m = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(DEVICE)
gw = {k:v.cpu().clone() for k,v in m.state_dict().items()}
config = {"device":DEVICE,"lr":0.01,"local_lr":0.01,"local_epochs":5,"gamma_high":15.0,"gamma_low":5.0,"total_rounds":100,"mean_snr_db":10.0}

t0 = time.time()
c = FedBiTClient(0, m, client_loaders[0], config)
payload, wt, stats = c.train(gw, 10.0, 0)
t1 = time.time()
print(f"Single client time: {t1-t0:.2f}s, loss={stats['loss']:.3f}, cr={stats['cr']:.1f}x")
print(f"Estimated per-round (10 clients): {(t1-t0)*10:.1f}s = {(t1-t0)*10/60:.1f}min")
