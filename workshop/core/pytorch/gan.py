"""Generative Adversarial Network (GAN) in PyTorch to model a 2D distribution."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Helper to generate a 2D ring (circle) distribution
def generate_real_data(num_samples=1000):
    np.random.seed(42)
    theta = np.random.uniform(0, 2 * np.pi, num_samples)
    r = 1.0 + np.random.normal(0, 0.08, num_samples)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.stack([x, y], axis=1).astype(np.float32)
    return torch.tensor(data)


# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.net(x)


# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def main(hook=None, config=None):
    from workshop.utils.hooks import NoOpProgressHook

    config = config or {}
    hook = hook or NoOpProgressHook()

    epochs = int(config.get("epochs", 150))
    latent_dim = int(config.get("latent_dim", 8))
    learning_rate = float(config.get("learning_rate", 0.001))
    batch_size = 64

    print("PyTorch Generative Adversarial Network (GAN)")
    print("=" * 45)
    print(f"Latent Dimension: {latent_dim}")
    print(f"Epochs: {epochs}, Learning Rate: {learning_rate}")
    print()

    if hook.is_cancelled():
        return
    hook.update_stage("Distribution Setup", 10)
    print("Generating target 2D circle distribution...")
    real_data = generate_real_data(1000)
    dataset = TensorDataset(real_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Distribution ready. Size: 1000 coordinate pairs.")

    if hook.is_cancelled():
        return
    hook.update_stage("Model Initialization", 20)

    netG = Generator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)

    criterion = nn.BCELoss()
    optG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    print("Generator and Discriminator models initialized.")

    if hook.is_cancelled():
        return
    hook.update_stage("Adversarial Training", 30)

    reporting_interval = max(1, epochs // 10)
    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        if hook.is_cancelled():
            return

        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        batches = 0

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            b_size = real_batch.size(0)

            # --- 1. Train Discriminator ---
            # Maximize log(D(x)) + log(1 - D(G(z)))
            netD.zero_grad()

            # Real samples
            label_real = torch.ones(b_size, 1).to(device)
            output_real = netD(real_batch)
            lossD_real = criterion(output_real, label_real)

            # Fake samples
            noise = torch.randn(b_size, latent_dim).to(device)
            fake_batch = netG(noise)
            label_fake = torch.zeros(b_size, 1).to(device)
            output_fake = netD(fake_batch.detach())
            lossD_fake = criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()
            epoch_d_loss += lossD.item()

            # --- 2. Train Generator ---
            # Maximize log(D(G(z)))
            netG.zero_grad()
            label_g = torch.ones(b_size, 1).to(device)  # Generator wants discriminator to think they're real
            output_g = netD(fake_batch)
            lossG = criterion(output_g, label_g)
            lossG.backward()
            optG.step()
            epoch_g_loss += lossG.item()

            batches += 1

        avg_d_loss = epoch_d_loss / batches
        avg_g_loss = epoch_g_loss / batches
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        # Update metrics
        if epoch % reporting_interval == 0 or epoch == epochs - 1:
            progress = 30 + int(50 * ((epoch + 1) / epochs))
            hook.update_stage("Adversarial Training", progress)
            hook.update_metrics({"epoch": epoch + 1, "d_loss": float(avg_d_loss), "g_loss": float(avg_g_loss)})
            print(f"Epoch {epoch + 1:3d}/{epochs}: D_Loss={avg_d_loss:.4f}, G_Loss={avg_g_loss:.4f}")

    if hook.is_cancelled():
        return
    hook.update_stage("Sampling", 85)
    print("\nGenerating final samples from Generator...")

    # Sample from generator
    netG.eval()
    with torch.no_grad():
        test_noise = torch.randn(500, latent_dim).to(device)
        generated_points = netG(test_noise).cpu().numpy()

    real_points = real_data.numpy()

    if hook.is_cancelled():
        return
    hook.update_stage("Visualization", 90)

    # 1. Distribution Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(real_points[:, 0], real_points[:, 1], color="blue", alpha=0.5, label="Real Distribution", s=15)
    plt.scatter(generated_points[:, 0], generated_points[:, 1], color="red", alpha=0.5, label="GAN Generated", s=15)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("GAN Real vs Generated Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hook.save_plot("gan_distribution_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Loss Curves Plot
    plt.figure(figsize=(6, 5))
    plt.plot(range(1, epochs + 1), d_losses, label="Discriminator Loss", color="blue", linewidth=2)
    plt.plot(range(1, epochs + 1), g_losses, label="Generator Loss", color="red", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("GAN Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    hook.save_plot("gan_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    hook.update_stage("Complete", 100)
