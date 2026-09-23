"""Exercise 1 — dados linearmente separáveis.

Gera figures/fig1.png, fig2.png, fig3.png e imprime os números do relatório.
Roda de qualquer diretório: python ex1_separable.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron import (ETA, train_perceptron, predict, make_dataset, plot_data,
                        plot_boundary_panel, unit)

rng = np.random.default_rng(42)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- A — dados
mean0, mean1 = np.array([1.5, 1.5]), np.array([5.0, 5.0])
cov = np.array([[0.5, 0.0], [0.0, 0.5]])
X, y = make_dataset(rng, mean0, mean1, cov)

fig = plot_data(X, y, [mean0, mean1], "Figure 1 — Exercise 1: dados separáveis (1000 por classe)")
fig.savefig(FIG / "fig1.png", bbox_inches="tight"); plt.close(fig)
print("A) médias amostrais:", X[y == 0].mean(0).round(3), X[y == 1].mean(0).round(3))

# ---------------------------------------------------------------- C — treino (η = 0.01)
w0 = rng.normal(0.0, 0.01, size=2)          # inicialização do enunciado; b = 0
res = train_perceptron(X, y, eta=ETA, w0=w0)

print(f"C) w0 = {w0}")
print(f"   w = [{res['w'][0]:.5f}, {res['w'][1]:.5f}], b = {res['b']:.5f}")
print(f"   épocas = {res['epochs']} (convergiu: {res['converged']}), acurácia = {res['acc']:.4f}")
print(f"   atualizações por época: {res['updates_hist'].tolist()}")

fig, ax = plt.subplots(figsize=(6, 5.5))
plot_boundary_panel(ax, X, y, res["w"], res["b"],
                    "Figure 2 — Exercise 1: fronteira de decisão final (η = 0.01)",
                    "fronteira $w\\cdot x + b = 0$")
ax.set_aspect("equal")
fig.savefig(FIG / "fig2.png", bbox_inches="tight"); plt.close(fig)

epochs = np.arange(1, res["epochs"] + 1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(epochs, res["acc_hist"], "o-", label="acurácia no fim da época")
ax.set_xticks(epochs); ax.set_ylim(0.45, 1.02)
ax.set_xlabel("época"); ax.set_ylabel("acurácia (dataset inteiro)")
ax.set_title("Figure 3 — Exercise 1: acurácia por época (η = 0.01)")
ax.legend()
fig.savefig(FIG / "fig3.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- D — η = 1.0, mesmo w0 e mesma ordem
res1 = train_perceptron(X, y, eta=1.0, w0=w0)
cos_dir = float(unit(res["w"]) @ unit(res1["w"]))
print(f"D) η = 1.0: épocas = {res1['epochs']}, acurácia = {res1['acc']:.4f}, "
      f"w = {res1['w'].round(4)}, b = {res1['b']:.4f}, atualizações {res1['updates_hist'].tolist()}")
print(f"   w/||w||: η=0.01 {unit(res['w']).round(4)} | η=1.0 {unit(res1['w']).round(4)}")
print(f"   cosseno = {cos_dir:.4f}, ângulo = {np.degrees(np.arccos(np.clip(cos_dir, -1, 1))):.2f}°")
print(f"   ||w0|| = {np.linalg.norm(w0):.4f}; ||w|| final: "
      f"{np.linalg.norm(res['w']):.4f} (η=0.01) vs {np.linalg.norm(res1['w']):.4f} (η=1.0)")

# ---------------------------------------------------------------- D — partindo de w = 0, b = 0
z1 = train_perceptron(X, y, eta=0.01, w0=np.zeros(2))
z2 = train_perceptron(X, y, eta=1.0, w0=np.zeros(2))
print(f"D) w0 = 0: η=0.01 -> w = {z1['w']}, b = {z1['b']:.4f}, épocas = {z1['epochs']}")
print(f"           η=1.00 -> w = {z2['w']}, b = {z2['b']:.4f}, épocas = {z2['epochs']}")
print(f"   razão w = {z2['w'] / z1['w']}, razão b = {z2['b'] / z1['b']:.6f}")
print("   previsões idênticas:",
      bool(np.all(predict(X, z1["w"], z1["b"]) == predict(X, z2["w"], z2["b"]))),
      "| atualizações:", z1["updates_hist"].tolist(), z2["updates_hist"].tolist())
