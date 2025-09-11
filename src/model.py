import numpy as np
from utils import ACTIVATIONS

class OneHiddenNN:
    def __init__(self, input_dim, hidden_dim, activation='relu', seed=1234):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / max(1, input_dim))
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = rng.randn(hidden_dim) * np.sqrt(2.0 / max(1, hidden_dim))
        self.b2 = 0.0
        self.act, self.act_grad = ACTIVATIONS[activation]

    def forward(self, X):
        Z = X.dot(self.W1.T) + self.b1
        A = self.act(Z)
        out = A.dot(self.W2) + self.b2
        return out, (X, Z, A)

    def backward(self, cache, grad_out):
        X, Z, A = cache
        B = X.shape[0]
        gW2 = (grad_out[:, None] * A).sum(axis=0) / B
        gb2 = grad_out.sum() / B
        gA = np.outer(grad_out, self.W2)
        gZ = gA * self.act_grad(Z)
        gW1 = gZ.T.dot(X) / B
        gb1 = gZ.sum(axis=0) / B
        return {'W1': gW1, 'b1': gb1, 'W2': gW2, 'b2': gb2}

    def get_params_vector(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                               self.W2.ravel(), np.array([self.b2])])

    def set_params_vector(self, vec):
        h, d = self.W1.shape
        w1_size = h * d
        self.W1 = vec[:w1_size].reshape(self.W1.shape)
        idx = w1_size
        self.b1 = vec[idx:idx+h]
        idx += h
        self.W2 = vec[idx:idx+h]
        idx += h
        self.b2 = float(vec[idx])

    def get_grad_vector(self, grads):
        return np.concatenate([grads['W1'].ravel(), grads['b1'].ravel(),
                               grads['W2'].ravel(), np.array([grads['b2']])])

    def apply_update_from_vector(self, update_vec, lr):
        params = self.get_params_vector()
        self.set_params_vector(params - lr * update_vec)
