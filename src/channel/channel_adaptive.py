import torch
import numpy as np

class ChannelAdaptiveSelector:
    def __init__(self, gamma_high=15.0, gamma_low=5.0, total_rounds=100):
        self.gamma_high_base = gamma_high
        self.gamma_low_base = gamma_low
        self.total_rounds = total_rounds

    def get_thresholds(self, t):
        factor = 1 + 0.5 * np.cos(np.pi * t / self.total_rounds)
        return self.gamma_high_base * factor, self.gamma_low_base * factor

    def select_payload(self, snr_db, B, h1, h2, t=0):
        gh, gl = self.get_thresholds(t)
        if snr_db > gh:
            return {'B':B,'h1':h1.float(),'h2':h2.float(),'mode':'rank1_fp32'}, 1.0
        elif snr_db > gl:
            h1q, s1 = self._q8(h1); h2q, s2 = self._q8(h2)
            return {'B':B,'h1':h1q,'h1_scale':s1,'h2':h2q,'h2_scale':s2,'mode':'rank1_int8'}, 0.85
        else:
            return {'B':B,'mode':'binary_only'}, 0.5

    def _q8(self, t):
        scale = t.abs().max() / 127.0 + 1e-8
        return (t / scale).round().clamp(-127,127).to(torch.int8), scale

class RayleighChannelSimulator:
    def __init__(self, mean_snr_db=10.0, seed=42):
        self.mean_snr_db = mean_snr_db
        self.rng = np.random.RandomState(seed)

    def sample_snr(self, n_clients):
        lin = 10 ** (self.mean_snr_db / 10)
        snr_lin = self.rng.exponential(lin, n_clients)
        return 10 * np.log10(snr_lin + 1e-10)
