[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Ready-green.svg)](https://scikit-learn.org/stable/)
[![Coursera Project](https://img.shields.io/badge/Portfolio-Coursera_Project-blue.svg)](https://coursera.org/share/f97524cda955c726afffa53999c5f532)

# 💳 FraudForge

FraudForge is an **end-to-end project on fraud detection and synthetic data generation** using **Generative Adversarial Networks (GANs)**.  

It started as a **Coursera-based baseline implementation** (vanilla GAN with Binary Cross-Entropy), but evolved into an exploration of **advanced GAN architectures** like **WGAN-GP**, with proper **evaluation pipelines** (RandomForest, classification metrics, feature distributions).  

This project shows how synthetic fraud samples can be generated to **balance datasets, improve model training, and stress-test fraud detection systems**.

## 📌 Overview

This repository contains my learnings and experiments with Generative Adversarial Networks (GANs) and Principal Component Analysis (PCA), inspired by a Coursera course and extended through independent research.

The goal is to understand how GANs generate synthetic data, how PCA aids in data exploration, and how advanced GAN techniques (like WGAN-GP) stabilize training.

---
## 🎓 What I Learned from Coursera

   - Clean and preprocess data for GANs
   - Employ GANs (Generative Adversarial Networks) for data generation
   - Apply PCA (Principal Component Analysis) for data exploration
## 📖 [Coursera Certificate](https://coursera.org/share/f97524cda955c726afffa53999c5f532)

## 🧠 What This Project Covers

1. **Baseline (Coursera-style Vanilla GAN)**
   - Implemented a simple generator & discriminator.
   - Used Binary Cross-Entropy loss.
   - Visualized learning progress with PCA scatter plots.
   - *Limitation:* Looked “mathematically okay” but didn’t evaluate usefulness downstream.

2. **Extended Approach (My Exploration)**
   - Implemented **Wasserstein GAN with Gradient Penalty (WGAN-GP)**.
   - Replaced BCE with Wasserstein loss for better stability.
   - Added **RandomForest evaluation pipeline**:
     - Checked how synthetic frauds help classification.
     - Reported **Accuracy, Precision, Recall ** (with special focus on Recall).
   - Generated a **balanced synthetic dataset** (fraud + genuine) and exported it as CSV for reuse.

---

## ⚙️ Architecture Overview

The project is divided into **two clear tracks**:

### 🔹 Vanilla GAN (Baseline)
- **Generator**: Fully connected layers with ReLU activations + BatchNorm.
- **Discriminator**: Dense layers with Sigmoid output.
- **Loss**: Binary Cross-Entropy.
- **Evaluation**: PCA visualization of real vs fake fraud samples.

### 🔹 WGAN-GP (Advanced)
- **Generator**: Similar structure, but trained with Wasserstein objective.
- **Discriminator (Critic)**: Removed Sigmoid; outputs Wasserstein score.
- **Loss**: Wasserstein loss + Gradient Penalty.
- **Evaluation**:  
  - Feature distribution histograms.  
  - RandomForest trained on real + synthetic data to assess downstream usefulness.  

---

## 📂 Project Structure

``` yaml
├── notebooks/
│ ├── 01_vanilla_gan_baseline.ipynb # Coursera-style GAN implementation
│ ├── 02_wgan_gp_extension.ipynb # Advanced GAN with RandomForest evaluation
│
├── datasets/
│ ├── Creditcard_dataset.csv # Original dataset
│ └── synthetic_creditcard_dataset.csv # Final generated dataset (fraud + genuine)
│
├── wgan checkpoints/
│ ├──critic.h5
│ ├──generator.h5
├── LICENSE
└── README.md
```
---

## 🚀 Running the Project

### ✅ Prerequisites
- Python 3.9+
- Jupyter Notebook
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
### Steps
- 1. Clone teh repo:
   ```bash
     git clone https://github.com/yourusername/FraudForge.git
     cd FraudForge
   ```
- 2. Open the notebooks in notebook folder:
-- Run 01_vanilla_gan_baseline.ipynb to reproduce the Coursera version.
-- Run 02_wgan_gp_extension.ipynb to see WGAN-GP with evaluation.
- 3. Check the generated datasets.
 
## Results obtained:
### 🔹 Training Summary (WGAN-GP)
Epoch 2000/2000 | Critic loss (mean): -4.6252 | Generator loss: -20.2002

### 🔹 TSTR Evaluation (Train on Synthetic, Test on Real)
- **Average Precision (AP):** 0.8566
- **ROC-AUC:** 0.9616  
- **Accuracy:** 0.9899
- **Precision:** 0.4886

### 🔹 Precision-Recall Curve
Classifier trained on **synthetic fraud + real genuine**, evaluated on **real test set**.

<img width="665" height="393" alt="image" src="https://github.com/user-attachments/assets/80d36e21-543d-47b3-bc2c-e0a7338f5182" />

## 🎯 Key Learnings

- Vanilla GANs can generate synthetic fraud data, but lack stability and downstream validation.

- WGAN-GP provides better convergence and avoids mode collapse.

- Simply training GANs isn’t enough — you must test synthetic data with real classifiers.

- Synthetic datasets can be a powerful tool for imbalanced fraud detection tasks.

**---**
To understand the WGAN-GP i recommend going through the jypter notebook where i have tried to explain a bit more in detail as to why WGAN-GP is better than a BCE Vanilla GAN.
---

## 📄 License

This project is licensed under the [![MIT License](https://img.shields.io/badge/License-MIT-blue.svg )](LICENSE).

## 📞 Contact

If you have any questions, feedback, or suggestions, feel free to reach out:

- 💼 GitHub Profile: [GitHub Link]( https://github.com/Retro-Jbit-Anon )
- 📧 Email: [Retro-Jbit-Anon](mailto:jidaarabbas@gmail.com)
