"""Perceptron escrito do zero (NumPy) + geração de dados e figuras.

Usado por ex1_separable.py e ex2_overlapping.py. O modelo (degrau, previsão,
regra de atualização e laço de treino com pocket) está todo aqui; nada vem de
bibliotecas de ML.
"""
import numpy as np
import matplotlib.pyplot as plt

N_PER_CLASS = 1000
MAX_EPOCHS = 100
ETA = 0.01

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
COLORS = {0: "tab:blue", 1: "tab:orange"}


# ---------------------------------------------------------------- modelo

def step(z):
    """Função degrau: 1 se z >= 0, senão 0."""
    return np.where(z >= 0, 1, 0)


def predict(X, w, b):
    """Previsão do perceptron: degrau aplicado a w·x + b (X pode ser uma amostra ou uma matriz)."""
    return step(X @ w + b)


def accuracy(X, y, w, b):
    """Fração de acertos no dataset inteiro."""
    return float(np.mean(predict(X, w, b) == y))


def train_perceptron(X, y, eta, w0, b0=0.0, max_epochs=MAX_EPOCHS):
    """Regra de aprendizado do perceptron, amostra a amostra.

    Para cada amostra (x_i, y_i), na ordem fixa do dataset:
        y_hat = step(w·x_i + b)
        w <- w + eta * (y_i - y_hat) * x_i
        b <- b + eta * (y_i - y_hat)
    Só há atualização quando y_hat != y_i. O treino para quando uma época inteira
    passa sem nenhuma atualização (convergiu) ou ao atingir max_epochs.

    Além dos pesos correntes, guarda o "pocket": toda vez que uma atualização
    produz acurácia no dataset inteiro maior do que qualquer outra já vista,
    copia (w, b) para o bolso. O pocket não interfere no treino.
    """
    w = np.asarray(w0, dtype=float).copy()
    b = float(b0)

    pocket_w, pocket_b = w.copy(), b
    pocket_acc = accuracy(X, y, w, b)
    pocket_epoch = 0                       # 0 = pesos iniciais, antes de qualquer época

    acc_hist, pocket_hist, updates_hist = [], [], []
    converged = False

    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for x_i, y_i in zip(X, y):
            y_hat = 1 if x_i @ w + b >= 0 else 0
            error = y_i - y_hat            # +1, 0 ou -1
            if error != 0:
                w = w + eta * error * x_i
                b = b + eta * error
                n_updates += 1

                # Pocket: avalia no dataset inteiro depois de cada atualização
                acc_now = accuracy(X, y, w, b)
                if acc_now > pocket_acc:
                    pocket_w, pocket_b = w.copy(), b
                    pocket_acc, pocket_epoch = acc_now, epoch

        acc_hist.append(accuracy(X, y, w, b))   # acurácia no fim da época
        pocket_hist.append(pocket_acc)
        updates_hist.append(n_updates)

        if n_updates == 0:
            converged = True
            break

    return {
        "w": w, "b": b, "epochs": epoch, "converged": converged,
        "acc": acc_hist[-1], "acc_hist": np.array(acc_hist),
        "updates_hist": np.array(updates_hist),
        "pocket_w": pocket_w, "pocket_b": pocket_b,
        "pocket_acc": pocket_acc, "pocket_epoch": pocket_epoch,
        "pocket_hist": np.array(pocket_hist),
    }


# ---------------------------------------------------------------- dados

def make_dataset(rng, mean0, mean1, cov, n=N_PER_CLASS):
    """Sorteia as duas classes e embaralha a ordem uma única vez.

    O embaralhamento importa: sem ele o perceptron veria 1000 amostras da
    classe 0 seguidas de 1000 da classe 1. A ordem fica fixa durante o treino.
    """
    X0 = rng.multivariate_normal(mean0, cov, size=n)
    X1 = rng.multivariate_normal(mean1, cov, size=n)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    order = rng.permutation(len(y))
    return X[order], y[order]


# ---------------------------------------------------------------- figuras

def scatter_classes(ax, X, y, alpha=0.5):
    for c in (0, 1):
        ax.scatter(*X[y == c].T, s=10, alpha=alpha, color=COLORS[c], label=f"classe {c}")


def draw_boundary(ax, w, b, X, **kw):
    """Desenha a reta w1*x1 + w2*x2 + b = 0 dentro dos limites dos dados."""
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    if abs(w[1]) > 1e-12:
        ax.plot(x1, -(w[0] * x1 + b) / w[1], **kw)
    else:                                   # reta vertical
        ax.axvline(-b / w[0], **kw)


def plot_boundary_panel(ax, X, y, w, b, title, line_label, line_color="k"):
    """Dados + fronteira + pontos mal classificados (círculo vermelho)."""
    scatter_classes(ax, X, y, alpha=0.35)
    wrong = predict(X, w, b) != y
    ax.scatter(*X[wrong].T, s=28, facecolors="none", edgecolors="red", linewidths=0.9,
               label=f"mal classificados ({wrong.sum()})")
    draw_boundary(ax, w, b, X, color=line_color, lw=2, label=line_label)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)


def plot_data(X, y, means, title):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    scatter_classes(ax, X, y, alpha=0.45)
    ax.scatter(*np.vstack(means).T, marker="X", s=150, color="k", label="médias")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_title(title)
    ax.legend(); ax.set_aspect("equal")
    return fig


def unit(v):
    return v / np.linalg.norm(v)
