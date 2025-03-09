import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy import linalg
from tqdm import tqdm
import os

class InceptionScore:
    def __init__(self, device='cuda'):
        # Load pretrained Inception model
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        self.device = device
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(device)
    
    def __call__(self, images, n_split=10, batch_size=32, eps=1e-16):
        """
        Calculate Inception Score
        
        Args:
            images: Tensor of shape (N, 3, H, W) in range [-1, 1]
            n_split: Number of splits for IS calculation
            batch_size: Batch size for inference
            eps: Small value to avoid numerical issues
            
        Returns:
            mean: Mean inception score
            std: Standard deviation of inception score
        """
        n_images = len(images)
        
        # Prepare predictions array
        preds = np.zeros((n_images, 1000))
        
        # Process images in batches
        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size].to(self.device)
            
            # Resize to 299x299 as required by Inception
            batch = self.up(batch)
            
            # Ensure values are in [0, 1] range
            batch = (batch + 1) / 2
            
            # Get model predictions
            with torch.no_grad():
                pred = F.softmax(self.model(batch), dim=1).cpu().numpy()
                
            preds[i:i + batch_size] = pred
            
        # Calculate scores across all splits
        scores = []
        n_part = n_images // n_split
        
        for i in range(n_split):
            part = preds[i * n_part:(i + 1) * n_part]
            
            # Calculate KL divergence
            p_y = np.mean(part, axis=0)
            scores.append(np.exp(np.mean(np.sum(part * (np.log(part + eps) - np.log(p_y + eps)), axis=1))))
            
        return np.mean(scores), np.std(scores)

class FID:
    def __init__(self, device='cuda'):
        # Load InceptionV3 with final classification layer removed
        self.model = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.fc = nn.Identity()  # Remove final classification layer
        self.model.eval()
        self.device = device
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(device)
        
    def __call__(self, real_images, fake_images, batch_size=32):
        """
        Calculate Fréchet Inception Distance
        
        Args:
            real_images: Tensor of real images (N, 3, H, W) in range [-1, 1]
            fake_images: Tensor of generated images (N, 3, H, W) in range [-1, 1]
            batch_size: Batch size for inference
            
        Returns:
            fid: Fréchet Inception Distance
        """
        # Get activations for real images
        real_activations = self._get_activations(real_images, batch_size)
        
        # Get activations for fake images
        fake_activations = self._get_activations(fake_images, batch_size)
        
        # Calculate mean and covariance
        mu1, sigma1 = self._calculate_statistics(real_activations)
        mu2, sigma2 = self._calculate_statistics(fake_activations)
        
        # Calculate FID
        fid = self._calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        return fid
    
    def _get_activations(self, images, batch_size):
        """Extract activations from the inception model"""
        n_images = len(images)
        activations = np.zeros((n_images, 2048))
        
        for i in range(0, n_images, batch_size):
            batch = images[i:i + batch_size].to(self.device)
            
            # Resize to 299x299 as required by Inception
            batch = self.up(batch)
            
            # Ensure values are in [0, 1] range
            batch = (batch + 1) / 2
            
            # Get activations
            with torch.no_grad():
                act = self.model(batch).squeeze().cpu().numpy()
                
            activations[i:i + batch_size] = act
            
        return activations
        
    def _calculate_statistics(self, activations):
        """Calculate mean and covariance"""
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma
        
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Fréchet distance between two distributions"""
        diff = mu1 - mu2
        
        # Product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Check for numerical errors
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            
        # Handle complex numbers if necessary
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean