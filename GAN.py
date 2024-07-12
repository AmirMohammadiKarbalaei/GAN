import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np
import os 

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, 28 * 28 * 1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28 * 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
class GAN:
    def __init__(self, latent_dim=100, lr=0.0002, b1=0.5, b2=0.999):
        self.latent_dim = latent_dim

        # Loss function
        self.adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.adversarial_loss.to(self.device)

    def train(self, epochs, batch_size=64, sample_interval=200):
        # Configure data loader
        os.makedirs("mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            MNIST(
                "mnist",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = torch.ones(imgs.size(0), 1, device=self.device, requires_grad=False)
                fake = torch.zeros(imgs.size(0), 1, device=self.device, requires_grad=False)

                # Configure input
                real_imgs = imgs.to(self.device)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.randn(imgs.size(0), self.latent_dim, device=self.device)

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    self.sample_images(batches_done)


    def sample_images(self, batches_done):
        z = torch.randn(64, self.latent_dim, device=self.device)
        gen_imgs = self.generator(z)
        gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images 0 - 1

        fig, axs = plt.subplots(8, 8, figsize=(8, 8))
        cnt = 0
        for i in range(8):
            for j in range(8):
                axs[i, j].imshow(gen_imgs[cnt].detach().cpu().numpy().squeeze(), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()



gan = GAN()
gan.train(epochs=10000, batch_size=64, sample_interval=2000)