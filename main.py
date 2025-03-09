import torch
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from torchvision.utils import save_image, make_grid

# Import our modules
from Dataset_Loading import get_dataset
from GAN_Modles_Implementations import DCGenerator, DCDiscriminator, WGANDiscriminator
from GAN_Training_Implementation import GANTrainer
from Evaulation_Metrics_Implementation import InceptionScore, FID

def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/bce", exist_ok=True)
    os.makedirs("results/lsgan", exist_ok=True)
    os.makedirs("results/wgan", exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    dataloader, img_shape = get_dataset(args.dataset, args.batch_size)
    print(f"Image shape: {img_shape}")
    
    # Training results
    results = {}
    
    # Train and evaluate each GAN variant
    for loss_type in ["bce", "lsgan", "wgan"]:
        print(f"\n{'='*50}")
        print(f"Training {loss_type.upper()} GAN")
        print(f"{'='*50}\n")
        
        # Initialize models
        generator = DCGenerator(args.latent_dim, img_shape).to(device)
        
        if loss_type == "wgan":
            discriminator = WGANDiscriminator(img_shape).to(device)
        else:
            discriminator = DCDiscriminator(img_shape).to(device)
            
        # Initialize trainer
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            latent_dim=args.latent_dim,
            device=device,
            loss_type=loss_type,
            lr=args.lr
        )
        
        # Train the GAN
        g_losses, d_losses = trainer.train(args.epochs, sample_interval=args.sample_interval)
        
        # Generate final images for evaluation
        print(f"\nGenerating images for evaluation...")
        num_eval_samples = min(args.num_eval_samples, len(dataloader.dataset))
        
        # Generate images
        eval_images = []
        with torch.no_grad():
            for i in range(0, num_eval_samples, args.batch_size):
                batch_size = min(args.batch_size, num_eval_samples - i)
                z = torch.randn(batch_size, args.latent_dim).to(device)
                images = generator(z)
                eval_images.append(images.cpu())
        
        eval_images = torch.cat(eval_images, dim=0)
        
        # Save a grid of generated images
        save_image(eval_images[:64], f"results/{loss_type}/final_grid.png", nrow=8, normalize=True)
        
        # Get real images for FID calculation
        real_images = []
        for imgs, _ in dataloader:
            real_images.append(imgs)
            if len(torch.cat(real_images, dim=0)) >= num_eval_samples:
                break
                
        real_images = torch.cat(real_images, dim=0)[:num_eval_samples]
        
        # Calculate metrics
        print(f"Calculating Inception Score...")
        is_model = InceptionScore(device)
        is_mean, is_std = is_model(eval_images)
        
        print(f"Calculating FID...")
        fid_model = FID(device)
        fid_score = fid_model(real_images, eval_images)
        
        # Store results
        results[loss_type] = {
            "inception_score": (is_mean, is_std),
            "fid": fid_score,
            "g_losses": g_losses,
            "d_losses": d_losses
        }
        
        print(f"\n{loss_type.upper()} GAN Results:")
        print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        print(f"FID Score: {fid_score:.2f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    
    # Generator losses
    plt.subplot(2, 1, 1)
    for loss_type in ["bce", "lsgan", "wgan"]:
        plt.plot(results[loss_type]["g_losses"][::100], label=f"{loss_type.upper()} G Loss")
    plt.title("Generator Losses")
    plt.legend()
    plt.grid(True)
    
    # Discriminator losses
    plt.subplot(2, 1, 2)
    for loss_type in ["bce", "lsgan", "wgan"]:
        plt.plot(results[loss_type]["d_losses"][::100], label=f"{loss_type.upper()} D Loss")
    plt.title("Discriminator Losses")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/loss_comparison.png")
    
    # Compare metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Inception Score comparison
    is_means = [results[loss_type]["inception_score"][0] for loss_type in ["bce", "lsgan", "wgan"]]
    is_stds = [results[loss_type]["inception_score"][1] for loss_type in ["bce", "lsgan", "wgan"]]
    
    ax1.bar(["BCE", "LSGAN", "WGAN"], is_means, yerr=is_stds, alpha=0.7, capsize=10)
    ax1.set_title("Inception Score (higher is better)")
    ax1.set_ylabel("Score")
    ax1.grid(True, axis='y')
    
    # FID comparison
    fid_scores = [results[loss_type]["fid"] for loss_type in ["bce", "lsgan", "wgan"]]
    
    ax2.bar(["BCE", "LSGAN", "WGAN"], fid_scores, alpha=0.7)
    ax2.set_title("FID (lower is better)")
    ax2.set_ylabel("Score")
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig("results/metrics_comparison.png")
    
    # Print final comparison
    print("\n\nFinal Comparison:")
    print(f"{'='*50}")
    print(f"{'Loss Type':<10} {'Inception Score':<20} {'FID':<10}")
    print(f"{'-'*50}")
    
    for loss_type in ["bce", "lsgan", "wgan"]:
        is_mean, is_std = results[loss_type]["inception_score"]
        fid = results[loss_type]["fid"]
        print(f"{loss_type.upper():<10} {is_mean:.2f} ± {is_std:.2f} {'':<10} {fid:.2f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAN Loss Function Comparison")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset to use (cifar10 or celeba)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of latent space")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--sample-interval", type=int, default=10, help="Interval between image sampling")
    parser.add_argument("--num-eval-samples", type=int, default=1000, help="Number of samples for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()
    main(args)