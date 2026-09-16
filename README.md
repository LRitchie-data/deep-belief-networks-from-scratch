# Deep Belief Networks & RBM from Scratch

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure **NumPy** implementation of Restricted Boltzmann Machines (RBM) and Deep Belief Networks (DBN). This project investigates the foundational impact of **greedy layer-wise unsupervised pre-training** and its role as an effective weight initialization and regularization scheme before supervised backpropagation on deep neural networks (DNN).

---

## 1. Introduction

The goal of this study is to evaluate the contribution of unsupervised pre-training in deep neural architectures. We implement from the ground up:
- **Generative architectures** (RBM, DBN) to model data distributions without labels.
- A **Deep Neural Network (DNN)** for classification of handwritten digits.

The benchmark systematically compares two approaches on the MNIST dataset:
1. **Pre-trained Network**: Layer-by-layer unsupervised pre-training via DBN followed by supervised fine-tuning (backpropagation).
2. **Randomly Initialized Network**: Standard weight initialization (Gaussian/Normal distribution) trained purely with supervised backpropagation.

---

## 2. Preliminary Study: Generative Models (Binary AlphaDigits)

We validate our generative algorithms on the **Binary AlphaDigits** dataset ($20 \times 16$ binary images). Here, the objective is not classification, but evaluating the network's ability to reconstruct and synthesize learned patterns.

### 2.1 Restricted Boltzmann Machine (RBM)
The RBM is an energy-based bipartite graphical model trained using the **Contrastive Divergence-1 (CD-1)** algorithm.

#### 2.1.1 Single-Class Generation ('A')
Training the RBM on a single character class ('A', visible dimension $p=320$, latent dimension $q=100$) allows the network to reconstruct and synthesize distinct variants of the letter.

<p align="center">
  <img src="assets/Générer A.png" width="70%" alt="Single Class Generation A" />
</p>

#### 2.1.2 Multimodal Modeling Capacity
We evaluated the RBM's capacity to simultaneously capture multiple characters. The objective is to verify whether the energy landscape can represent a multimodal probability distribution and spontaneously sample across modes via **Gibbs Sampling**.

<p align="center">
  <img src="assets/Lettre.png" width="85%" alt="Multimodal Generation" />
</p>

> **Analysis**: The RBM successfully learns the multimodal distribution. During Gibbs sampling transitions across potential energy wells, the network occasionally traverses intermediate state spaces, but overwhelmingly settles into stable energy basins corresponding to the learned letters.

#### 2.1.3 Convergence Analysis (Reconstruction Error)
We monitored the Mean Squared Error (MSE) between the input vectors and their reconstruction through the hidden units across 100 training epochs.

<p align="center">
  <img src="assets/Evolution erreur nombre epoch.png" width="55%" alt="Reconstruction Error MSE Curve" />
</p>

> **Convergence Dynamics**:
> - **Fast Learning Phase (Epochs 1–20)**: The MSE drops sharply as the RBM captures global topological structures.
> - **Fine-tuning Phase (Epochs 20–100)**: The curve stabilizes asymptotically, corresponding to local stroke refinements and validating gradient convergence under CD-1.

### 2.2 Deep Belief Network (DBN)
By stacking multiple RBMs in a greedy, layer-wise procedure, higher latent layers extract hierarchical, abstract representations. Image generation through deep Gibbs chains filters local high-frequency noise, producing cleaner boundaries than single-layer RBMs.

---

## 3. Supervised Classification: DNN on MNIST

We evaluate classification performance on **MNIST** ($28 \times 28$ grayscale images binarized at a 0.5 threshold). We compare two identical feedforward architectures trained with:
- **Pre-trained parameters**: 20 RBM epochs (CD-1) + 30 backpropagation epochs.
- **Random parameters**: Normal initialization + 30 backpropagation epochs.
- **Hyperparameters**: Learning Rate = 0.1, Batch Size = 64.

### 3.1 Criterion 1: Influence of Network Depth
Holding hidden layer width fixed at 200 units, we vary the depth from **2 to 4 hidden layers**.

<p align="center">
  <img src="assets/erreur nombre de couches.png" width="60%" alt="Classification Error vs Depth" />
</p>

> **Analysis**:
> - **Random Initialization (Orange)**: Error explodes when depth increases to 3 and 4 layers. With Sigmoid activations, standard backpropagation suffers severely from the **vanishing gradient problem**, attenuating error signals before they reach early layers.
> - **Pre-trained Network (Blue)**: Test error remains stable and low across 2, 3, and 4 layers. Greedy DBN pre-training places weights in a favorable basin of attraction, preventing vanishing gradients during supervised fine-tuning.

### 3.2 Criterion 2: Influence of Layer Width
Holding depth fixed at 2 hidden layers, we vary the width from **100 to 500 hidden neurons**.

> **Analysis**: Expanding representation capacity improves performance across both models. However, the pre-trained DBN network maintains a systematic advantage, confirming that DBN initialization acts as an effective regularizer.

### 3.3 Criterion 3: Influence of Training Dataset Size
With an architecture fixed at `[784, 200, 200, 10]`, we evaluate sample efficiency across dataset sizes ranging from **1,000 to 40,000 images**.

> **Analysis**:
> - **Low-Data Regime ($N = 1,000$)**: The performance gap is massive in favor of DBN. Unsupervised feature learning extracts intrinsic structural priors from unlabeled images, compensating for label scarcity.
> - **High-Data Regime ($N \ge 30,000$)**: The curves gradually converge. Pure supervised backpropagation finds viable minima when flooded with data, although pre-training still accelerates convergence speed.

---

## 4. Optimal Benchmark Configuration

After training on the MNIST dataset with an optimal `[784, 200, 200, 10]` architecture:
- **Final Test Error (Pre-trained DBN)**: **~4.0% (96.0% Accuracy)**
- **Final Test Error (Random Init)**: ~8.0% (92.0% Accuracy)

### Output Probability Confidence
For test samples (e.g. a handwritten '7'), the pre-trained classifier outputs sharply peaked categorical probability distributions ($P(y=7) > 0.95$, near 0 elsewhere), with test Cross-Entropy Loss converging cleanly to ~0.08.

---

## 5. Conclusion

This project demonstrates two core theoretical principles:
1. **Generative Representation**: Unsupervised RBM and DBN architectures model complex, multimodal binary image densities and synthesize coherent digits via Markov chain Gibbs sampling.
2. **Pre-training Advantage**: DBN initialization fundamentally outperforms random initialization whenever architectures are deep (mitigating vanishing gradients) or labeled samples are constrained.

While modern architectures frequently use ReLU and residual connections, greedy layer-wise pre-training remains one of the seminal breakthroughs that unlocked deep neural network training.

---

## Project Structure

```text
+-- assets/            # Benchmark figures and evaluation curves
+-- data/              # Binary datasets (MNIST & AlphaDigits)
+-- notebooks/         # Complete Google Colab / Jupyter notebook
+-- src/
¦   +-- rbm.py         # Contrastive Divergence CD-1 implementation
¦   +-- dbn.py         # Deep Belief Network layer stacking
¦   +-- dnn.py         # Feedforward & Backpropagation fine-tuning
¦   +-- utils.py       # Data loaders & pre-processing routines
+-- main.py            # Model entrypoint
+-- requirements.txt
```

# Quickstart & Reproducibility
## 1. Installation
Bash
git clone [https://github.com/LRitchie-data/deep-belief-networks-from-scratch.git](https://github.com/LRitchie-data/deep-belief-networks-from-scratch.git)
cd deep-belief-networks-from-scratch
pip install -r requirements.txt
## 2. Dataset Setup
Ensure the following binary files are placed in the working directory:

train-images-idx3-ubyte

train-labels-idx1-ubyte

t10k-images-idx3-ubyte

t10k-labels-idx1-ubyte

binaryalphadigs.mat

## 3. Run
Execute the pipeline via:

Bash
python main.py
Or open and run the notebook notebooks/exploration_generative_alpha.ipynb.

### References
Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A fast learning algorithm for deep belief nets. Neural Computation, 18(7), 1527-1554.

Bengio, Y., et al. (2007). Greedy layer-wise training of deep network
