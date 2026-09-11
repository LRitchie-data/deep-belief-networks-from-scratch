import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

class RBM:
    def __init__(self, n_visible: int, n_hidden: int, std: float = 0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.random.normal(0.0, std, (n_visible, n_hidden))
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden)

    def propup(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(v, self.W) + self.b)

    def propdown(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(np.dot(h, self.W.T) + self.a)

    def sample_h(self, v: np.ndarray):
        p_h = self.propup(v)
        h_sample = (np.random.rand(*p_h.shape) < p_h).astype(np.float64)
        return p_h, h_sample

    def sample_v(self, h: np.ndarray):
        p_v = self.propdown(h)
        v_sample = (np.random.rand(*p_v.shape) < p_v).astype(np.float64)
        return p_v, v_sample

    def fit_batch(self, v0: np.ndarray, lr: float):
        batch_size = v0.shape[0]
        p_h0, h0 = self.sample_h(v0)
        p_v1, _ = self.sample_v(h0)
        p_h1 = self.propup(p_v1)

        grad_W = (np.dot(v0.T, p_h0) - np.dot(p_v1.T, p_h1)) / batch_size
        grad_a = np.mean(v0 - p_v1, axis=0)
        grad_b = np.mean(p_h0 - p_h1, axis=0)

        self.W += lr * grad_W
        self.a += lr * grad_a
        self.b += lr * grad_b

    def gibbs_sampling(self, num_samples: int, steps: int = 1000) -> np.ndarray:
        v = (np.random.rand(num_samples, self.n_visible) < 0.5).astype(np.float64)
        for _ in range(steps):
            _, h = self.sample_h(v)
            _, v = self.sample_v(h)
        return v
