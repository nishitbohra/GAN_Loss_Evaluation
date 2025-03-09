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

## Training Results

### Loss Curves
The following graph shows the Generator and Discriminator losses for each of the three loss functions over the training period:

![Generator and Discriminator Losses](https://github.com/nishitbohra/GAN_Loss_Evaluation/blob/main/results/loss_comparison.png)

### Performance Metrics
We evaluated the performance of each GAN variant using two key metrics:

![Inception Score and FID](https://github.com/nishitbohra/GAN_Loss_Evaluation/blob/main/results/metrics_comparison.png)

## Analysis of Results

### Loss Patterns
- **BCE Loss**: Both generator and discriminator losses stabilize around 0.7, showing consistent training dynamics
- **LSGAN Loss**: After initial convergence, stabilizes with lower loss values around 0.25
- **WGAN Loss**: Shows near-zero loss values, which is characteristic of Wasserstein distance

### Quality Metrics
- **Inception Score**: LSGAN achieves the highest score (~2.4), followed by BCE (~2.0), with WGAN significantly lower (~1.0)
- **FID Score**: LSGAN performs best with the lowest score (~100), BCE follows closely (~120), while WGAN shows much higher values (~370)

## Key Findings
1. **LSGAN** produces the highest quality and most diverse images based on both metrics
2. **BCE** offers solid performance as a traditional approach
3. **WGAN** shows stable training but underperforms in image quality metrics, suggesting potential implementation challenges

## Future Work
- Implement WGAN-GP (with gradient penalty) to improve Wasserstein GAN performance
- Test on higher resolution datasets
- Explore additional loss variants like Hinge Loss and Relativistic GANs
