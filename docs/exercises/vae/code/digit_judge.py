"""Classificador de dígitos auxiliar, usado só nas análises. Não faz parte do VAE.

Uma rede simples (784 -> 256 -> 10) treinada no conjunto de treino. Serve de "juiz" para dizer se uma
imagem gerada parece um dígito, e qual: a confiança é a maior probabilidade da softmax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_judge(X_train, y_train, epochs=5):
    torch.manual_seed(42)
    clf = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True):
            loss = F.cross_entropy(clf(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()

    @torch.no_grad()
    def judge(imgs):
        """Dígito previsto e confiança para cada imagem (N, 784)."""
        p = F.softmax(clf(imgs), dim=1)
        confidence, label = p.max(1)
        return label.numpy(), confidence.numpy()

    return judge
