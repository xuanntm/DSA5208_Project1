import numpy as np

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(float)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_grad(x): s = sigmoid(x); return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_grad(x): return 1 - np.tanh(x)**2

ACTIVATIONS = {
    "relu": (relu, relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
}

def mse_loss_and_grad(y_pred, y_true):
    err = y_pred - y_true
    loss = 0.5 * np.sum(err**2)
    grad = err
    return loss, grad
