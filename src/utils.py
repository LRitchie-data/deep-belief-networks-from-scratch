import os
import struct
import numpy as np
import scipy.io

def load_alpha_digits(char_indices, filepath='data/binaryalphadigs.mat'):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset introuvable : {filepath}")
    mat = scipy.io.loadmat(filepath)
    key = 'dat' if 'dat' in mat else 'data'
    data = mat[key]
    X = []
    for c in char_indices:
        imgs = data[c]
        if imgs.shape[0] == 1 and imgs.ndim > 1:
            imgs = imgs[0]
        for img in imgs:
            X.append(img.flatten())
    return np.array(X, dtype=np.float64)

def load_mnist_local(images_path, labels_path):
    with open(labels_path, 'rb') as f:
        data = f.read()
        y = np.frombuffer(data, dtype=np.uint8, offset=8)

    with open(images_path, 'rb') as f:
        data = f.read()
        images = np.frombuffer(data, dtype=np.uint8, offset=16)

    images = images.reshape(len(y), 28 * 28)
    X = (images / 255.0 > 0.5).astype(np.float64)
    y_onehot = np.eye(10)[y]
    return X, y_onehot
