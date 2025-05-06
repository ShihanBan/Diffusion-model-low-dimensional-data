### diffusion_model/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_diffusion_model(model, diffusion, dataloader, device, epochs=100):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.to(device)

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for batch in pbar:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.size(0),), device=device)
            noise = torch.randn_like(batch)
            x_noisy = diffusion.q_sample(batch, t, noise=noise)
            pred_noise = model(x_noisy, t)
            loss = criterion(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
