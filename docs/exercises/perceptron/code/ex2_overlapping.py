"""Exercise 2 — dados sobrepostos, perceptron + pocket.

Gera figures/fig4.png, fig5.png, fig6.png e imprime os números do relatório.
Roda de qualquer diretório: python ex2_overlapping.py
"""
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perceptron import (ETA, train_perceptron, predict, accuracy, make_dataset, plot_data,
                        plot_boundary_panel, draw_boundary)

rng = np.random.default_rng(42)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- A — dados
mean0, mean1 = np.array([3.0, 3.0]), np.array([4.0, 4.0])
cov = np.array([[1.5, 0.0], [0.0, 1.5]])
X, y = make_dataset(rng, mean0, mean1, cov)

fig = plot_data(X, y, [mean0, mean1], "Figure 4 — Exercise 2: dados sobrepostos (1000 por classe)")
fig.savefig(FIG / "fig4.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- B — treino com pocket (mesma função do Ex. 1)
w0 = rng.normal(0.0, 0.01, size=2)
res = train_perceptron(X, y, eta=ETA, w0=w0)

print(f"B) épocas = {res['epochs']} (convergiu: {res['converged']})")
print(f"   FINAL : w = [{res['w'][0]:.5f}, {res['w'][1]:.5f}], b = {res['b']:.5f}, acurácia = {res['acc']:.4f}")
print(f"   POCKET: w = [{res['pocket_w'][0]:.5f}, {res['pocket_w'][1]:.5f}], b = {res['pocket_b']:.5f}, "
      f"acurácia = {res['pocket_acc']:.4f}, época {res['pocket_epoch']}")
u = res["updates_hist"]
print(f"   atualizações por época: média {u.mean():.0f}, mín {u.min()}, máx {u.max()}")
h = res["acc_hist"]
print(f"   acurácia corrente nas 100 épocas: mín {h.min():.4f}, máx {h.max():.4f}, média {h.mean():.4f}")
print(f"   pocket ao fim da época 1: {res['pocket_hist'][0]:.4f}")

# ---------------------------------------------------------------- C — figuras
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
plot_boundary_panel(axes[0], X, y, res["w"], res["b"],
                    f"Final (época {res['epochs']}): acurácia {res['acc']:.1%}",
                    "fronteira final", line_color="k")
draw_boundary(axes[0], res["pocket_w"], res["pocket_b"], X, color="green", lw=1.5, ls="--",
              label="fronteira pocket (ref.)")
axes[0].legend(loc="upper left", fontsize=8)
plot_boundary_panel(axes[1], X, y, res["pocket_w"], res["pocket_b"],
                    f"Pocket (época {res['pocket_epoch']}): acurácia {res['pocket_acc']:.1%}",
                    "fronteira pocket", line_color="green")
draw_boundary(axes[1], res["w"], res["b"], X, color="k", lw=1.5, ls="--",
              label="fronteira final (ref.)")
axes[1].legend(loc="upper left", fontsize=8)
fig.suptitle("Figure 5 — Exercise 2: fronteiras final e pocket "
             "(vermelho = mal classificado pela fronteira do painel)")
fig.tight_layout()
fig.savefig(FIG / "fig5.png", bbox_inches="tight"); plt.close(fig)

epochs = np.arange(1, res["epochs"] + 1)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(epochs, res["acc_hist"], lw=1.2, label="acurácia corrente (fim da época)")
ax.plot(epochs, res["pocket_hist"], lw=2, color="green", label="acurácia do pocket")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chute (50%)")
ax.set_ylim(0.4, 0.8)
ax.set_xlabel("época"); ax.set_ylabel("acurácia (dataset inteiro)")
ax.set_title("Figure 6 — Exercise 2: acurácia corrente vs. pocket (η = 0.01)")
ax.legend(loc="lower right")
fig.savefig(FIG / "fig6.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- D — onde estão as fronteiras
midpoint = (mean0 + mean1) / 2
for name, w, b in [("final", res["w"], res["b"]), ("pocket", res["pocket_w"], res["pocket_b"])]:
    p1 = predict(X, w, b).mean()
    dist = (midpoint @ w + b) / np.linalg.norm(w)
    ang = np.degrees(np.arctan2(w[1], w[0]))
    print(f"D) {name:>6}: prevê classe 1 em {p1:.1%} | ângulo de w = {ang:.1f}° "
          f"| distância com sinal de (3.5, 3.5) à reta = {dist:+.3f}")

# Teto teórico de uma reta: covariâncias iguais e isotrópicas -> mediatriz, erro Φ(−‖Δμ‖/2σ)
z = np.linalg.norm(mean1 - mean0) / (2 * sqrt(cov[0, 0]))
print(f"D) melhor reta teórica: Φ({z:.4f}) = {0.5 * (1 + erf(z / sqrt(2))):.4f}; "
      f"reta de Bayes (w=[1,1], b=-7) nesta amostra: {accuracy(X, y, np.array([1.0, 1.0]), -7.0):.4f}")

# ---------------------------------------------------------------- D — mais épocas ou η menor?
rows = []
for eta in (1.0, 0.01, 0.001):
    r = train_perceptron(X, y, eta=eta, w0=w0, max_epochs=500)
    tail = r["acc_hist"][-100:]
    rows.append({"eta": eta, "épocas": r["epochs"], "convergiu": r["converged"],
                 "acc final": r["acc"], "acc últ.100 mín": tail.min(), "acc últ.100 máx": tail.max(),
                 "pocket": r["pocket_acc"], "época pocket": r["pocket_epoch"],
                 "atualiz./época": r["updates_hist"].mean()})
print("D) 500 épocas, mesmo w0:")
print(pd.DataFrame(rows).round(4).to_string(index=False))
