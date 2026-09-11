from typing import List
import numpy as np
from src.dbn import DBN
from src.rbm import sigmoid

def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

class DeepClassifier:
    def __init__(self, layer_sizes: List[int], n_classes: int):
        self.dbn = DBN(layer_sizes)
        self.n_classes = n_classes
        self.W_clf = np.random.normal(0.0, 0.01, (layer_sizes[-1], n_classes))
        self.b_clf = np.zeros(n_classes)

    def pretrain_backbone(self, X: np.ndarray, epochs: int = 15, lr: float = 0.05, batch_size: int = 64):
        self.dbn.pretrain(X, epochs=epochs, lr=lr, batch_size=batch_size)

    def forward(self, X: np.ndarray):
        activations = [X]
        curr = X
        for rbm in self.dbn.rbms:
            curr = rbm.propup(curr)
            activations.append(curr)
        logits = np.dot(curr, self.W_clf) + self.b_clf
        probs = softmax(logits)
        activations.append(probs)
        return activations

    def fit(self, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 30, lr: float = 0.01, batch_size: int = 64):
        n_samples = X.shape[0]
        for ep in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for i in range(0, n_samples, batch_size):
                xb = X_shuffled[i:i + batch_size]
                yb = y_shuffled[i:i + batch_size]
                m = xb.shape[0]

                acts = self.forward(xb)
                probs = acts[-1]
                delta = (probs - yb)

                h_last = acts[-2]
                dW_c = np.dot(h_last.T, delta) / m
                db_c = np.mean(delta, axis=0)

                W_next = self.W_clf.copy()
                self.W_clf -= lr * dW_c
                self.b_clf -= lr * db_c

                curr_delta = delta
                for j in range(len(self.dbn.rbms) - 1, -1, -1):
                    rbm = self.dbn.rbms[j]
                    h_val = acts[j + 1]
                    deriv = h_val * (1.0 - h_val)
                    delta_h = np.dot(curr_delta, W_next.T) * deriv

                    dW = np.dot(acts[j].T, delta_h) / m
                    db = np.mean(delta_h, axis=0)

                    W_next = rbm.W.copy()
                    rbm.W -= lr * dW
                    rbm.b -= lr * db
                    curr_delta = delta_h

    def evaluate(self, X: np.ndarray, y_onehot: np.ndarray) -> float:
        probs = self.forward(X)[-1]
        preds = np.argmax(probs, axis=1)
        targets = np.argmax(y_onehot, axis=1)
        return float(np.mean(preds == targets))
