### diffusion_model/diffusion.py
import torch
import numpy as np

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = self.alpha_hats[t].sqrt().unsqueeze(1)
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hats[t]).sqrt().unsqueeze(1)
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise

    def sample(self, model, shape, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_tensor)
            alpha = self.alphas[t].to(device)
            alpha_hat = self.alpha_hats[t].to(device)
            beta = self.betas[t].to(device)
            x = (1 / alpha.sqrt()) * (x - beta / (1 - alpha_hat).sqrt() * pred_noise)
            if t > 0:
                noise = torch.randn_like(x)
                x += beta.sqrt() * noise
        return x