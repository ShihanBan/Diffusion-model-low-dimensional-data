### diffusion_model/utils.py
import matplotlib.pyplot as plt
import torch

def plot_samples(samples, title="Generated Samples"):
    samples = samples.detach().cpu().numpy()
    if samples.shape[1] == 2:
        plt.scatter(samples[:, 0], samples[:, 1])
    else:
        norms = torch.norm(torch.tensor(samples), dim=1).numpy()
        plt.plot(sorted(norms))
    plt.title(title)
    plt.show()