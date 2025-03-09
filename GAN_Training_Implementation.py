import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from torchvision.utils import save_image

class GANTrainer:
    def __init__(self, generator, discriminator, dataloader, latent_dim, device, 
                 loss_type='bce', lr=0.0002, b1=0.5, b2=0.999):
        """
        Initialize GAN trainer
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            dataloader: DataLoader for the dataset
            latent_dim: Dimension of latent space
            device: Device to run training on (cuda/cpu)
            loss_type: Type of GAN loss ('bce', 'lsgan', 'wgan')
            lr: Learning rate
            b1: Adam optimizer beta1
            b2: Adam optimizer beta2
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.dataloader = dataloader
        self.latent_dim = latent_dim
        self.device = device
        self.loss_type = loss_type
        
        # Initialize optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        # Initialize loss functions based on type
        if loss_type == 'bce':
            self.adversarial_loss = nn.BCEWithLogitsLoss()
        elif loss_type == 'lsgan':
            self.adversarial_loss = nn.MSELoss()
        elif loss_type == 'wgan':
            # WGAN doesn't use a traditional loss function
            self.clip_value = 0.01
        else:
            raise ValueError(f"Loss type {loss_type} not supported")
            
        # Create directories for saving results
        self.results_dir = f"results/{loss_type}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def train(self, n_epochs, sample_interval=10):
        """
        Train the GAN
        
        Args:
            n_epochs: Number of epochs to train
            sample_interval: Interval for sampling and saving generated images
        """
        # Metrics for tracking
        g_losses = []
        d_losses = []
        
        # Labels for BCE loss
        if self.loss_type in ['bce', 'lsgan']:
            real_label = 1
            fake_label = 0
        
        # Training loop
        for epoch in range(n_epochs):
            for i, (real_imgs, _) in enumerate(self.dataloader):
                # Move data to device
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                # -----------------
                # Train Discriminator
                # -----------------
                self.optimizer_D.zero_grad()
                
                # Sample noise for generator
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                
                # Generate a batch of images
                fake_imgs = self.generator(z)
                
                if self.loss_type == 'wgan':
                    # WGAN loss
                    d_loss = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs.detach()))
                    d_loss.backward()
                    self.optimizer_D.step()
                    
                    # Clip weights of discriminator
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)
                        
                else:  # BCE or LSGAN
                    # Adversarial ground truths
                    valid = torch.full((batch_size, 1), real_label, dtype=torch.float, device=self.device)
                    fake = torch.full((batch_size, 1), fake_label, dtype=torch.float, device=self.device)
                    
                    # Real images
                    real_pred = self.discriminator(real_imgs)
                    d_real_loss = self.adversarial_loss(real_pred, valid)
                    
                    # Fake images
                    fake_pred = self.discriminator(fake_imgs.detach())
                    d_fake_loss = self.adversarial_loss(fake_pred, fake)
                    
                    # Total discriminator loss
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    d_loss.backward()
                    self.optimizer_D.step()
                
                # -----------------
                # Train Generator
                # -----------------
                if self.loss_type == 'wgan':
                    # Train generator every 5 iterations
                    if i % 5 == 0:
                        self.optimizer_G.zero_grad()
                        # WGAN loss
                        g_loss = -torch.mean(self.discriminator(fake_imgs))
                        g_loss.backward()
                        self.optimizer_G.step()
                else:
                    self.optimizer_G.zero_grad()
                    # Loss for generator
                    fake_pred = self.discriminator(fake_imgs)
                    g_loss = self.adversarial_loss(fake_pred, valid)
                    g_loss.backward()
                    self.optimizer_G.step()
                
                # Track losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                # Print training progress
                if i % 100 == 0:
                    print(
                        f"[Epoch {epoch}/{n_epochs}] "
                        f"[Batch {i}/{len(self.dataloader)}] "
                        f"[D loss: {d_loss.item():.4f}] "
                        f"[G loss: {g_loss.item():.4f}]"
                    )
            
            # Save generated images at specified intervals
            if epoch % sample_interval == 0 or epoch == n_epochs - 1:
                self.sample_images(epoch)
        
        # Save the models
        torch.save(self.generator.state_dict(), f"models/{self.loss_type}_generator.pth")
        torch.save(self.discriminator.state_dict(), f"models/{self.loss_type}_discriminator.pth")
        
        return g_losses, d_losses
    
    def sample_images(self, epoch):
        """
        Save sample generated images
        
        Args:
            epoch: Current epoch number
        """
        # Generate images
        z = torch.randn(25, self.latent_dim).to(self.device)
        gen_imgs = self.generator(z)
        
        # Save images
        save_path = f"{self.results_dir}/{epoch}.png"
        save_image(gen_imgs.data[:25], save_path, nrow=5, normalize=True)
        
        return gen_imgs