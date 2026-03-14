import torch
import numpy as np


def gpu_rank1_svd(W: torch.Tensor):
    """Fast GPU-based rank-1 SVD approximation using power iteration."""
    # W: (m, n) float tensor on GPU
    m, n = W.shape
    # Power iteration for rank-1 (much faster than full SVD on GPU)
    # Initialize random vector
    v = torch.randn(n, device=W.device, dtype=torch.float32)
    v = v / v.norm()
    for _ in range(3):  # 3 iterations is enough for rank-1
        u = W @ v
        u_norm = u.norm() + 1e-10
        u = u / u_norm
        v = W.t() @ u
        sigma = v.norm() + 1e-10
        v = v / sigma
    # Final sigma
    u = W @ v
    sigma = u.norm()
    if sigma > 1e-10:
        u = u / sigma
    return u, sigma, v  # u: (m,), sigma: scalar, v: (n,)


class Rank1GradientCompressor:
    def __init__(self, n_components=1, random_state=42):
        self.n_components = n_components
        self.random_state = random_state

    def compress(self, delta_W: torch.Tensor):
        """Compress a 2D weight matrix with rank-1 decomposition."""
        # Work on GPU if available
        if delta_W.device.type == "cpu" and torch.cuda.is_available():
            W = delta_W.float().cuda()
        else:
            W = delta_W.float()
        m, n = W.shape

        u1, sigma1, v1 = gpu_rank1_svd(W)
        h1 = torch.sqrt(sigma1) * u1  # (m,)
        h2 = torch.sqrt(sigma1) * v1  # (n,)

        B = torch.sign(W)
        B[B == 0] = 1

        h1_cpu = h1.cpu(); h2_cpu = h2.cpu(); B_cpu = B.cpu()
        error_bound = float((W - B * torch.outer(h1, h2)).norm())

        meta = {
            'sigma1': float(sigma1),
            'error_bound': error_bound,
            'compression_ratio': 32*m*n / (m*n + 32*(m+n))
        }
        dev = delta_W.device
        return B_cpu.to(dev), h1_cpu.to(dev), h2_cpu.to(dev), meta

    def decompress(self, B, h1, h2):
        return B.float() * torch.outer(h1, h2)

    def compress_model(self, model_delta: dict, device=None):
        """Compress all layers of a model delta."""
        compressed = {}; orig_bits = 0; comp_bits = 0
        for name, param in model_delta.items():
            if param.dim() >= 2:
                W = param.view(param.shape[0], -1)
                if device is not None and torch.cuda.is_available():
                    W_gpu = W.float().to(device)
                else:
                    W_gpu = W.float()
                B, h1, h2, meta = self.compress(W_gpu)
                compressed[name] = {
                    'type': 'rank1', 'B': B, 'h1': h1, 'h2': h2,
                    'shape': param.shape, 'meta': meta
                }
                m, n = W.shape
                orig_bits += 32*m*n; comp_bits += m*n + 32*(m+n)
            else:
                compressed[name] = {'type': 'fp16', 'data': param.half()}
                orig_bits += 32*param.numel(); comp_bits += 16*param.numel()
        return compressed, orig_bits / max(comp_bits, 1)
