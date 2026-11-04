"""VAE para o MNIST: dados, modelo, perda e treino. Usado pelos Exercises 1 e 2.

O autograd do PyTorch calcula os gradientes. A reparametrização, os dois termos da perda e a
amostragem estão escritos aqui à mão: não uso classe de VAE de biblioteca nem torch.distributions.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

DATA = Path(__file__).resolve().parent / "data"     # o MNIST é baixado aqui na primeira execução
DATA.mkdir(exist_ok=True)
LR, BATCH, EPOCHS, BETA = 1e-3, 128, 20, 1.0        # Adam 1e-3, lote 128, 20 épocas, beta = 1


# ---------------------------------------------------------------- dados
def load_mnist():
    """Treino 50 000 e validação 10 000 (split do enunciado) e o teste oficial de 10 000.

    Devolve (X_train, y_train, X_val, y_val, X_test, y_test), com X em tensores (N, 784) e pixels em [0, 1].
    """
    train_full = datasets.MNIST(DATA, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(DATA, train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))

    def to_tensors(dataset, indices):
        # ToTensor() divide o uint8 por 255. Aplico a mesma conta de uma vez nas imagens escolhidas
        # pelo split, em vez de converter imagem por imagem (o resultado é o mesmo, só mais rápido).
        X = dataset.data[indices].reshape(-1, 784).float() / 255.0
        return X, dataset.targets[indices]

    X_train, y_train = to_tensors(train_full, torch.tensor(train.indices))
    X_val, y_val = to_tensors(train_full, torch.tensor(val.indices))
    X_test, y_test = to_tensors(test, torch.arange(len(test)))
    assert torch.equal(X_train[0], train[0][0].reshape(784))      # confere: igual ao ToTensor() do split
    assert torch.equal(X_val[0], val[0][0].reshape(784))
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------- modelo
class VAE(nn.Module):
    """Encoder 784 -> 512 -> 256 (ReLU), cabeças mu e log sigma^2 (256 -> d), decoder d -> 256 -> 512 -> 784."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)          # a rede prevê log sigma^2, que pode ser qualquer real
        self.decoder = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                                     nn.Linear(256, 512), nn.ReLU(),
                                     nn.Linear(512, 784), nn.Sigmoid())   # cada pixel vira uma probabilidade

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar, generator=None):
        """Truque da reparametrização: z = mu + sigma * eps, com eps ~ N(0, I).

        O sorteio fica todo em eps, que não depende dos parâmetros. Assim z é uma função
        diferenciável de mu e sigma, e o gradiente chega no encoder.
        """
        sigma = torch.exp(0.5 * logvar)               # sigma = exp(log sigma^2 / 2)
        eps = torch.randn(sigma.shape, generator=generator)
        return mu + sigma * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n):
        """Amostra do modelo: z ~ N(0, I) (a prior) e decodifica. Usa o gerador global do torch."""
        z = torch.randn(n, self.d)
        return self.decode(z)


# ---------------------------------------------------------------- perda
def loss_terms(x, x_hat, mu, logvar):
    """Os dois termos da perda em nats por imagem: soma nos 784 pixels (ou nas d dimensões), média nas imagens.

    Reconstrução: BCE somada nos pixels = -log p(x|z) com cada pixel Bernoulli.
    KL: forma fechada de KL(N(mu, sigma^2) || N(0, I)) = 1/2 * sum(mu^2 + sigma^2 - 1 - log sigma^2).
    """
    n = x.shape[0]
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / n     # "sum" e não "mean" (ver o relatório)
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / n
    return recon, kl


@torch.no_grad()
def evaluate(model, X, use_mean=False, seed=0):
    """Reconstrução e KL médios por imagem em X, com o modelo em modo de avaliação.

    Por padrão z é sorteado de q(z|x) (uma amostra por imagem), que é a estimativa usual do ELBO.
    O sorteio usa um gerador próprio com seed fixa, então avaliar não mexe na sequência aleatória do treino.
    Com use_mean=True o decoder recebe mu em vez de um z sorteado.
    """
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    total_recon = total_kl = 0.0
    for i in range(0, len(X), 1000):
        x = X[i:i + 1000]
        mu, logvar = model.encode(x)
        z = mu if use_mean else model.reparameterize(mu, logvar, gen)
        recon, kl = loss_terms(x, model.decode(z), mu, logvar)
        total_recon += recon.item() * len(x)
        total_kl += kl.item() * len(x)
    return total_recon / len(X), total_kl / len(X)


# ---------------------------------------------------------------- treino
def train_vae(d, X_train, X_val, epochs=EPOCHS, verbose=True):
    """Treina um VAE com latente de dimensão d. Devolve o modelo e o histórico dos dois termos por época."""
    torch.manual_seed(42)                             # seed fixada antes de construir cada modelo
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X_train), batch_size=BATCH, shuffle=True)
    history = {"train_recon": [], "train_kl": [], "val_recon": [], "val_kl": []}
    history["init_recon"], history["init_kl"] = evaluate(model, X_val)   # validação antes da 1ª época

    for epoch in range(1, epochs + 1):
        model.train()
        sum_recon = sum_kl = 0.0
        for (x,) in loader:
            x_hat, mu, logvar = model(x)
            recon, kl = loss_terms(x, x_hat, mu, logvar)
            loss = recon + BETA * kl                  # -ELBO por imagem
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_recon += recon.item() * len(x)        # média da época ponderada pelo tamanho do lote
            sum_kl += kl.item() * len(x)
        history["train_recon"].append(sum_recon / len(X_train))
        history["train_kl"].append(sum_kl / len(X_train))
        val_recon, val_kl = evaluate(model, X_val)    # validação ao fim de cada época
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)
        if verbose:
            print(f"   época {epoch:2d}: treino rec {history['train_recon'][-1]:7.2f}  KL {history['train_kl'][-1]:5.2f}"
                  f" | validação rec {val_recon:7.2f}  KL {val_kl:5.2f}")
    return model, history


def load_or_train(d, X_train, X_val):
    """Carrega o modelo salvo pelo Exercise 1 (se existir) ou treina de novo com a mesma seed."""
    path = DATA / f"vae_d{d}.pt"
    if path.exists():
        torch.manual_seed(42)
        model = VAE(d)
        model.load_state_dict(torch.load(path))
        model.eval()
        return model
    model, _ = train_vae(d, X_train, X_val, verbose=False)
    torch.save(model.state_dict(), path)
    return model
