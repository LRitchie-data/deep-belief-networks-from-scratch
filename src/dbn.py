from typing import List
import numpy as np
from src.rbm import RBM

class DBN:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.rbms = [
            RBM(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]

    def pretrain(self, X: np.ndarray, epochs: int = 20, lr: float = 0.05, batch_size: int = 64):
        input_data = X
        for idx, rbm in enumerate(self.rbms):
            n_samples = input_data.shape[0]
            for epoch in range(epochs):
                indices = np.random.permutation(n_samples)
                shuffled_data = input_data[indices]
                for i in range(0, n_samples, batch_size):
                    batch = shuffled_data[i:i + batch_size]
                    rbm.fit_batch(batch, lr)
            input_data = rbm.propup(input_data)
