"""MLP escrito do zero em NumPy, usado nos Exercises 2, 3 e 4.

O mesmo código de forward, backward e update serve para classificação binária
(saída sigmoide + entropia cruzada binária) e multiclasse (saída softmax +
entropia cruzada categórica). Só mudam os argumentos: tamanhos das camadas,
tipo de saída, taxa de aprendizado e gerador aleatório.

Notação da aula: em cada camada, u = W a + b, com W no formato
(neurônios que recebem, neurônios que enviam).
"""
import numpy as np


class MLP:
    def __init__(self, layer_sizes, output="sigmoid", lr=0.1, rng=None):
        """layer_sizes: lista com o tamanho de cada camada, ex. [2, 8, 1].
        output: "sigmoid" (binário, 1 neurônio de saída) ou "softmax" (multiclasse).
        lr: taxa de aprendizado do gradiente descendente.
        rng: np.random.Generator usado para inicializar os pesos.
        """
        self.layer_sizes = layer_sizes
        self.output = output
        self.lr = lr
        # Pesos com desvio 1/sqrt(n_entradas) (inicialização de Xavier, boa para tanh); vieses em zero
        self.W = [rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_out, n_in))
                  for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    # ------------------------------------------------------------ ativações
    @staticmethod
    def tanh(u):
        return np.tanh(u)

    @staticmethod
    def sigmoid(u):
        return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))

    @staticmethod
    def softmax(u):
        e = np.exp(u - u.max(axis=1, keepdims=True))   # subtrai o máximo para não estourar
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------ forward
    def forward(self, X):
        """X: (N, n_entradas). Guarda a saída de cada camada para usar no backward."""
        self.activations = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            u = self.activations[-1] @ W.T + b          # pré-ativação da camada
            if i < len(self.W) - 1:
                a = self.tanh(u)                        # camadas ocultas: tanh
            elif self.output == "sigmoid":
                a = self.sigmoid(u)
            else:
                a = self.softmax(u)
            self.activations.append(a)
        return self.activations[-1]

    # ------------------------------------------------------------ perda
    def loss(self, Y_hat, Y):
        """Entropia cruzada média. Y tem o mesmo formato de Y_hat (one-hot no caso multiclasse)."""
        eps = 1e-12
        if self.output == "sigmoid":
            return float(-np.mean(Y * np.log(Y_hat + eps) + (1 - Y) * np.log(1 - Y_hat + eps)))
        return float(-np.mean(np.sum(Y * np.log(Y_hat + eps), axis=1)))

    # ------------------------------------------------------------ backward
    def backward(self, Y):
        """Calcula os gradientes de todos os pesos e vieses (regra da cadeia, sem autodiff).

        Tanto sigmoide + BCE quanto softmax + entropia cruzada categórica dão
        dL/du = (Y_hat - Y) / N na saída, então o mesmo código serve aos dois casos.
        """
        N = Y.shape[0]
        delta = (self.activations[-1] - Y) / N          # dL/du da camada de saída
        self.grad_W = [None] * len(self.W)
        self.grad_b = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            self.grad_W[i] = delta.T @ self.activations[i]   # dL/dW = delta^T a_anterior
            self.grad_b[i] = delta.sum(axis=0)               # dL/db = soma dos deltas
            if i > 0:
                # volta uma camada: dL/da = delta W, e multiplica pela derivada da tanh (1 - a^2)
                delta = (delta @ self.W[i]) * (1 - self.activations[i] ** 2)

    # ------------------------------------------------------------ update
    def update(self):
        """Gradiente descendente: p <- p - lr * dL/dp."""
        for i in range(len(self.W)):
            self.W[i] -= self.lr * self.grad_W[i]
            self.b[i] -= self.lr * self.grad_b[i]

    # ------------------------------------------------------------ treino e previsão
    def fit(self, X, Y, epochs):
        """Treino em lote completo. Devolve a perda de treino de cada época."""
        history = []
        for _ in range(epochs):
            Y_hat = self.forward(X)
            history.append(self.loss(Y_hat, Y))
            self.backward(Y)
            self.update()
        return np.array(history)

    def predict(self, X):
        Y_hat = self.forward(X)
        if self.output == "sigmoid":
            return (Y_hat[:, 0] >= 0.5).astype(int)
        return Y_hat.argmax(axis=1)

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))
