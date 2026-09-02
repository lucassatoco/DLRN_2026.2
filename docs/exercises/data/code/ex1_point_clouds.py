"""Exercise 1 - Point clouds: geometry and spread in 2D.

Gera as quatro nuvens gaussianas, mede a razao de separacao e a taxa de mistura
para cada fator de escala, e salva as Figuras 1, 1b, 2 e 3 em ../figures/.

Uso:  python ex1_point_clouds.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Semente unica do relatorio.
rng = np.random.default_rng(42)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]
MEAN = np.array([[2, 3], [5, 6], [8, 1], [15, 4]], dtype=float)
STD = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]], dtype=float)
N = 100
SCALES = [0.5, 1.0, 2.0, 4.0]

# Ruido padrao sorteado uma unica vez e reaproveitado em todas as escalas, para
# que os paineis da Figura 2 mostrem as mesmas nuvens dilatando.
Z = rng.normal(0, 1, size=(len(MEAN), N, 2))


def generate_clouds(mean, std, z, scale=1.0):
    """As medias ficam fixas; todo desvio e multiplicado por `scale`."""
    X = mean[:, None, :] + (std * scale)[:, None, :] * z
    return X.reshape(-1, 2), np.repeat(np.arange(len(mean)), z.shape[1])


def separation_table(mean, std):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j)."""
    sigma_bar = std.mean(axis=1)
    linhas = []
    for i in range(len(mean)):
        for j in range(i + 1, len(mean)):
            dist = np.linalg.norm(mean[i] - mean[j])
            denom = sigma_bar[i] + sigma_bar[j]
            linhas.append(
                {
                    "pair": f"({i}, {j})",
                    "||mu_i - mu_j||": dist,
                    "sigma_bar_i + sigma_bar_j": denom,
                    "r_ij (s=1)": dist / denom,
                }
            )
    return pd.DataFrame(linhas).sort_values("r_ij (s=1)").reset_index(drop=True)


def mixing_rate(X, y, means):
    """Fracao de pontos cujo centro mais proximo nao e o da propria classe."""
    dists = np.linalg.norm(X[:, None, :] - means[None, :, :], axis=2)
    return float(np.mean(dists.argmin(axis=1) != y))


def figure_1(X, y, path):
    plt.figure(figsize=(8, 6))
    for c in range(len(NAMES)):
        m = y == c
        plt.scatter(X[m, 0], X[m, 1], s=14, alpha=0.7, label=NAMES[c])
    plt.scatter(MEAN[:, 0], MEAN[:, 1], marker="X", s=180, c="black",
                zorder=5, label="Class means")
    plt.title("Figure 1 - Four Gaussian clouds ($s = 1$)")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def figure_2(datasets, path):
    all_pts = np.vstack([X for X, _ in datasets.values()])
    xlim = (all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1)
    ylim = (all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, sc in zip(axes.flat, SCALES):
        X, y = datasets[sc]
        for c in range(len(NAMES)):
            m = y == c
            ax.scatter(X[m, 0], X[m, 1], s=10, alpha=0.6, label=NAMES[c])
        ax.scatter(MEAN[:, 0], MEAN[:, 1], marker="X", s=110, c="black",
                   zorder=5, label="Class means")
        ax.set_title(f"$s = {sc}$")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Figure 2 - Same means, growing spread (shared axes)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def figure_3(mixing, r_min, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(mixing.keys()), list(mixing.values()), marker="o",
            linewidth=2, markersize=9, label="mixing rate")
    for sc, v in mixing.items():
        ax.annotate(f"{v:.4f}", (sc, v), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=10)

    ax2 = ax.twinx()
    s_grid = np.linspace(min(SCALES), max(SCALES), 200)
    ax2.plot(s_grid, r_min / s_grid, linestyle="--", color="firebrick",
             linewidth=1.8, label="$r_{min}(s) = r_{min}(1)/s$")
    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=1.2)
    ax2.set_ylabel("$r_{min}$", color="firebrick")
    ax2.grid(False)

    ax.set_title("Figure 3 - Mixing rate vs. scale factor")
    ax.set_xlabel("scale factor $s$")
    ax.set_ylabel("mixing rate")
    handles = ax.get_lines() + ax2.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def figure_1b(X, y, path):
    """Particao por centro mais proximo: esboco das fronteiras do item C.2."""
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 500),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 500),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.linalg.norm(grid[:, None, :] - MEAN[None, :, :], axis=2).argmin(axis=1)
    zz = zz.reshape(xx.shape)

    plt.figure(figsize=(9, 6.5))
    plt.contourf(xx, yy, zz, levels=np.arange(-0.5, len(MEAN)), alpha=0.15)
    plt.contour(xx, yy, zz, levels=np.arange(-0.5, len(MEAN)),
                colors="black", linewidths=1.2, linestyles="--")
    for c in range(len(NAMES)):
        m = y == c
        plt.scatter(X[m, 0], X[m, 1], s=14, alpha=0.75, label=NAMES[c])
    plt.scatter(MEAN[:, 0], MEAN[:, 1], marker="X", s=170, c="black",
                zorder=5, label="Class means")
    plt.title("Figure 1b - Sketch of the decision boundaries")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    X1, y1 = generate_clouds(MEAN, STD, Z, scale=1.0)
    print("A - Figure 1")
    print(f"  X {X1.shape} | pontos por classe {np.bincount(y1)}")
    figure_1(X1, y1, os.path.join(FIG_DIR, "fig1.png"))

    datasets = {sc: generate_clouds(MEAN, STD, Z, scale=sc) for sc in SCALES}
    figure_2(datasets, os.path.join(FIG_DIR, "fig2.png"))

    print("\nB.2 - Separation ratio (s = 1)")
    sep = separation_table(MEAN, STD)
    print(sep.round(4).to_string(index=False))
    par_min = sep.iloc[0]["pair"]
    r_min = float(sep.iloc[0]["r_ij (s=1)"])
    print(f"  menor r_ij: {r_min:.4f} no par {par_min}")
    print(f"  em s = 2 (extrapolado): {r_min / 2:.4f}")

    print("\nB.3 - Mixing rate")
    mixing = {sc: mixing_rate(X, y, MEAN) for sc, (X, y) in datasets.items()}
    for sc in SCALES:
        print(f"  s = {sc:<4} mixing {mixing[sc]:.4f}   r_min(s) = {r_min / sc:.4f}")
    figure_3(mixing, r_min, os.path.join(FIG_DIR, "fig3.png"))

    print("\nC.2 - Figure 1b")
    figure_1b(X1, y1, os.path.join(FIG_DIR, "fig1b.png"))

    print("\nFiguras salvas em", os.path.normpath(FIG_DIR))


if __name__ == "__main__":
    main()
