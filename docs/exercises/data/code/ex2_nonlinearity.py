"""Exercise 2 - Non-linearity in higher dimensions.

Gera o Dataset I (gaussianas 5D deslocadas) e o Dataset II (cascas concentricas),
aplica PCA, mede distancia entre centros e raios, e salva as Figuras 4 e 5.

Uso:  python ex2_nonlinearity.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rng = np.random.default_rng(42)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DIM = 5
N_5D = 500

MEAN_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
MEAN_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])

# Sigma_A: correlacao positiva entre as features 0 e 1.
COV_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])

# Sigma_B: variancias maiores e correlacao negativa entre as features 0 e 1.
COV_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

RADIUS_C = {"mu": 2.0, "sigma": 0.4}
RADIUS_D = {"mu": 5.0, "sigma": 0.4}


def sample_shell(n, dim, mu_r, sigma_r):
    """Direcao uniforme na esfera unitaria, raio ~ Normal(mu_r, sigma_r)."""
    directions = rng.normal(0, 1, size=(n, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = np.clip(rng.normal(mu_r, sigma_r, n), 0, None)
    return directions * radii[:, None]


def centre_distance(X, y):
    """Distancia euclidiana entre os dois centroides, no espaco 5D original."""
    return float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))


def separate_dataset_ii(X, tau):
    """g(x) = sum(x_i^2) - tau^2; retorna 1 (casca) se g(x) > 0, senao 0."""
    return (np.sum(X ** 2, axis=1) > tau ** 2).astype(int)


def figure_4(P1, y1, evr1, P2, y2, evr2, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    painels = [
        (axes[0], P1, y1, ("Class A", "Class B"),
         "Dataset I - shifted Gaussians", evr1),
        (axes[1], P2, y2, ("Class C (core)", "Class D (shell)"),
         "Dataset II - concentric shells", evr2),
    ]
    for ax, P, y, nomes, titulo, evr in painels:
        for k, nome in enumerate(nomes):
            m = y == k
            ax.scatter(P[m, 0], P[m, 1], s=14, alpha=0.55, label=nome)
        ax.set_title(f"{titulo}\nPC1+PC2 = {evr.sum():.1%} da variancia")
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
        ax.legend(fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Figure 4 - PCA projection of both 5D datasets",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def figure_5(r_ds1, r_ds2, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, radii, titulo in [(axes[0], r_ds1, "Dataset I"),
                              (axes[1], r_ds2, "Dataset II")]:
        for nome, r in radii.items():
            ax.hist(r, bins=40, alpha=0.6, edgecolor="white",
                    label=f"{nome} (mean {r.mean():.2f})")
        ax.set_title(f"{titulo} - radius $\\|x\\|$")
        ax.set_xlabel("$\\|x\\|$")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
    fig.suptitle("Figure 5 - Radius histogram, both classes overlaid",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    XA = rng.multivariate_normal(MEAN_A, COV_A, N_5D)
    XB = rng.multivariate_normal(MEAN_B, COV_B, N_5D)
    X_ds1 = np.vstack([XA, XB])
    y_ds1 = np.concatenate([np.zeros(N_5D, dtype=int), np.ones(N_5D, dtype=int)])

    print("A - Dataset I", X_ds1.shape)
    print("  media A:", XA.mean(axis=0).round(3))
    print("  media B:", XB.mean(axis=0).round(3))
    print(f"  corr A(0,1): {np.corrcoef(XA[:, 0], XA[:, 1])[0, 1]:+.3f}  (positiva)")
    print(f"  corr B(0,1): {np.corrcoef(XB[:, 0], XB[:, 1])[0, 1]:+.3f}  (negativa)")

    XC = sample_shell(N_5D, DIM, RADIUS_C["mu"], RADIUS_C["sigma"])
    XD = sample_shell(N_5D, DIM, RADIUS_D["mu"], RADIUS_D["sigma"])
    X_ds2 = np.vstack([XC, XD])
    y_ds2 = np.concatenate([np.zeros(N_5D, dtype=int), np.ones(N_5D, dtype=int)])

    print("\nB - Dataset II", X_ds2.shape)
    print(f"  raio medio C: {np.linalg.norm(XC, axis=1).mean():.3f}")
    print(f"  raio medio D: {np.linalg.norm(XD, axis=1).mean():.3f}")
    print("  centro C:", XC.mean(axis=0).round(3))
    print("  centro D:", XD.mean(axis=0).round(3))

    pca1, pca2 = PCA(n_components=2), PCA(n_components=2)
    P1 = pca1.fit_transform(X_ds1)
    P2 = pca2.fit_transform(X_ds2)
    evr1 = pca1.explained_variance_ratio_
    evr2 = pca2.explained_variance_ratio_
    figure_4(P1, y_ds1, evr1, P2, y_ds2, evr2, os.path.join(FIG_DIR, "fig4.png"))

    print("\nC.1 / C.2 - Explained variance")
    print(f"  Dataset I : PC1 {evr1[0]:.4f} + PC2 {evr1[1]:.4f} = {evr1.sum():.4f}")
    print(f"  Dataset II: PC1 {evr2[0]:.4f} + PC2 {evr2[1]:.4f} = {evr2.sum():.4f}")

    print("\nC.3 - Distancia entre centros (5D)")
    print(f"  Dataset I : {centre_distance(X_ds1, y_ds1):.4f}")
    print(f"  Dataset II: {centre_distance(X_ds2, y_ds2):.4f}")

    r_ds1 = {"Class A": np.linalg.norm(XA, axis=1),
             "Class B": np.linalg.norm(XB, axis=1)}
    r_ds2 = {"Class C (core)": np.linalg.norm(XC, axis=1),
             "Class D (shell)": np.linalg.norm(XD, axis=1)}
    figure_5(r_ds1, r_ds2, os.path.join(FIG_DIR, "fig5.png"))

    tau = (RADIUS_C["mu"] + RADIUS_D["mu"]) / 2
    pred = separate_dataset_ii(X_ds2, tau)
    print("\nD.3 - Funcao separadora")
    print(f"  tau = {tau:.1f} (meio do caminho entre os raios do enunciado)")
    print(f"  fracao correta: {np.mean(pred == y_ds2):.4f}")

    print("\nFiguras salvas em", os.path.normpath(FIG_DIR))


if __name__ == "__main__":
    main()
