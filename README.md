# GAN_Loss_Evaluation
# Comparative Analysis of GAN Loss Functions

## Project Overview
This project explores the performance of three different GAN loss functions on the CIFAR-10 dataset:
1. Binary Cross-Entropy (BCE) Loss
2. Least Squares Loss (LS-GAN)
3. Wasserstein Loss (WGAN)

## Project Structure
```
gan-loss-comparison/
│
├── datasets/
│   └── cifar10/
│
├── models/
│   ├── generator.py
│   └── discriminator.py
│
├── losses/
│   ├── bce_loss.py
│   ├── lsgan_loss.py
│   └── wgan_loss.py
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Key Metrics
- **Inception Score (IS)**
- **Fréchet Inception Distance (FID)**
- Visual Image Quality

## Experimental Setup
- **Dataset**: CIFAR-10
- **Training**: 50 epochs per loss function
- **Model Architecture**: 
  - Generator: Transposed Convolutional Network
  - Discriminator: Convolutional Neural Network

## Expected Outcomes
Comparative analysis of GAN performance across different loss functions, highlighting strengths and limitations of each approach.
