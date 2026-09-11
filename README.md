# Deep Belief Networks & RBM from Scratch

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure **NumPy** implementation of Restricted Boltzmann Machines (RBM) and Deep Belief Networks (DBN). This project investigates greedy layer-wise unsupervised pre-training and its role as an effective weight initialization and regularization scheme before supervised backpropagation.

## Architecture & Features

- **Generative Modeling**: Contrastive Divergence (CD-1) on Bernoulli RBMs to approximate arbitrary binary image distributions.
- **Deep Representations**: Greedy layer-wise stacking of multiple RBMs forming a Deep Belief Network (DBN).
- **Supervised Fine-Tuning**: Classification layer with end-to-end backpropagation via Stochastic Gradient Descent (SGD).

## Project Structure

\\\	ext
+-- data/              # Dataset loading & scripts
+-- notebooks/         # Visual inspection & generation demos
+-- src/
¦   +-- rbm.py         # Restricted Boltzmann Machine (CD-1)
¦   +-- dbn.py         # Deep Belief Network layer-stack
¦   +-- dnn.py         # Fine-tuning & classification
¦   +-- utils.py       # Data parsers (MNIST & Binary AlphaDigits)
+-- main.py            # Benchmark script
+-- requirements.txt
\\\

## Getting Started

\\\ash
pip install -r requirements.txt
python main.py
\\\

## References
- Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). *A fast learning algorithm for deep belief nets*. Neural computation.
- Bengio, Y., et al. (2007). *Greedy layer-wise training of deep networks*. NeurIPS.
