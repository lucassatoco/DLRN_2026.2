"""Exercise 2 — classificação binária que uma reta não resolve.

Gera figures/fig1.png, fig2.png, fig3.png e imprime os números do relatório.
Roda de qualquer diretório: python ex2_binary.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from mlp import MLP

rng = np.random.default_rng(42)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
COLORS = {0: "tab:blue", 1: "tab:orange"}

# ---------------------------------------------------------------- A — dados
# 500 pontos no centro (0, 0) e 250 em cada um dos centros (3, 3) e (-3, -3)
X, cluster = make_blobs(n_samples=[500, 250, 250], centers=[(0, 0), (3, 3), (-3, -3)],
                        cluster_std=1.2, random_state=42)
y = (cluster > 0).astype(int)                 # classe 1 = os dois clusters de fora
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"A) treino {X_train.shape}, teste {X_test.shape}, classe 1 no teste: {y_test.mean():.2f}")

fig, ax = plt.subplots(figsize=(6, 5.5))
for c in (0, 1):
    ax.scatter(*X[y == c].T, s=10, alpha=0.6, color=COLORS[c], label=f"classe {c}")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title("Figure 1 — Exercise 2: os 1000 pontos, coloridos por classe")
ax.legend(); ax.set_aspect("equal")
fig.savefig(FIG / "fig1.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- B — baseline linear
logreg = LogisticRegression().fit(X_train, y_train)
base_acc = logreg.score(X_test, y_test)
print(f"B) regressão logística: acurácia no teste = {base_acc:.4f}")
print(f"   coeficientes = {logreg.coef_[0].round(4)}, intercepto = {logreg.intercept_[0]:.4f}")
print(f"   fração do teste prevista como classe 1 = {logreg.predict(X_test).mean():.3f}")

# ---------------------------------------------------------------- C — MLP
SIZES, LR, EPOCHS = [2, 8, 1], 0.5, 2000
model = MLP(SIZES, output="sigmoid", lr=LR, rng=rng)
history = model.fit(X_train, y_train[:, None], EPOCHS)
acc_train = (model.predict(X_train) == y_train).mean()
acc_test = (model.predict(X_test) == y_test).mean()
print(f"C) arquitetura {SIZES} (tanh na oculta, sigmoide na saída), lr = {LR}, {EPOCHS} épocas, "
      f"{model.n_params()} parâmetros")
print(f"   perda de treino: inicial {history[0]:.4f}, final {history[-1]:.4f}")
print(f"   acurácia: treino {acc_train:.4f}, teste {acc_test:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, EPOCHS + 1), history, label="perda de treino (BCE)")
ax.set_xlabel("época"); ax.set_ylabel("entropia cruzada binária")
ax.set_title("Figure 2 — Exercise 2: perda de treino por época")
ax.legend()
fig.savefig(FIG / "fig2.png", bbox_inches="tight"); plt.close(fig)

# Fronteira de decisão: avalia a rede numa grade e pinta as duas regiões
x1, x2 = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
grid_pred = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.contourf(x1, x2, grid_pred, levels=[-0.5, 0.5, 1.5], colors=[COLORS[0], COLORS[1]], alpha=0.2)
for c in (0, 1):
    ax.scatter(*X_test[y_test == c].T, s=18, color=COLORS[c], edgecolors="k", linewidths=0.3,
               label=f"classe {c} (teste)")
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
ax.set_title(f"Figure 3 — Exercise 2: fronteira do MLP {SIZES} (teste: {acc_test:.1%})")
ax.legend(); ax.set_aspect("equal")
fig.savefig(FIG / "fig3.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- D — forma da região da classe 1
# Onde a região da classe 1 fica: proporção da grade em cada lado da diagonal x1 + x2 = 0
region1 = grid_pred == 1
s = x1 + x2
print(f"D) região da classe 1: {region1.mean():.1%} da grade; "
      f"acima de x1+x2=0: {region1[s > 0].mean():.1%}, abaixo: {region1[s < 0].mean():.1%}")
