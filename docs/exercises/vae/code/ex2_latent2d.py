"""Exercise 2 — o mesmo VAE com d = 2, um espaço latente que dá para desenhar.

Gera figures/fig4.png, fig5.png e fig6.png e imprime os números do relatório.
A interpolação latente usa o modelo d = 16 salvo pelo Exercise 1 (se ele não existir, é treinado de novo
com a mesma seed). Roda de qualquer diretório: python ex2_latent2d.py
"""
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digit_judge import train_judge
from vae import DATA, EPOCHS, evaluate, load_mnist, load_or_train, train_vae

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
torch.set_num_threads(4)                     # o mesmo número de threads do Exercise 1

X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

# ---------------------------------------------------------------- A — VAE com d = 2
# Mesmo modelo, dados, seed, beta e orçamento do Exercise 1. Só muda d.
model2, hist2 = train_vae(2, X_train, X_val)
torch.save(model2.state_dict(), DATA / "vae_d2.pt")
model16 = load_or_train(16, X_train, X_val)
model2.eval(); model16.eval()

rec2, kl2 = evaluate(model2, X_test)         # teste só para os números finais
rec16, kl16 = evaluate(model16, X_test)
print("A) teste, nats por imagem:")
print(f"   d = 16: reconstrução {rec16:.2f}, KL {kl16:.2f}, -ELBO {rec16 + kl16:.2f}")
print(f"   d =  2: reconstrução {rec2:.2f}, KL {kl2:.2f}, -ELBO {rec2 + kl2:.2f}")
print(f"   diferença (d=2 - d=16): reconstrução {rec2 - rec16:+.2f}, KL {kl2 - kl16:+.2f}, -ELBO {rec2 + kl2 - rec16 - kl16:+.2f}")
print(f"   d = 2, validação: KL na época 1 = {hist2['val_kl'][0]:.2f}, na época {EPOCHS} = {hist2['val_kl'][-1]:.2f}; "
      f"reconstrução na época {EPOCHS} = {hist2['val_recon'][-1]:.2f}")

# ---------------------------------------------------------------- B — o teste codificado
with torch.no_grad():
    mu_test = model2.encode(X_test)[0].numpy()
    mu_train = model2.encode(X_train)[0].numpy()

colors = plt.cm.tab10(np.arange(10))
order = np.random.default_rng(0).permutation(len(mu_test))   # ordem aleatória: nenhum dígito fica escondido por baixo
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(mu_test[order, 0], mu_test[order, 1], s=3, alpha=0.6, color=colors[y_test.numpy()[order]])
for k in range(10):                                           # pontos vazios só para montar a legenda
    ax.scatter([], [], s=12, color=colors[k], label=f"{k}")
lim = norm.ppf(0.95)
ax.add_patch(plt.Rectangle((-lim, -lim), 2 * lim, 2 * lim, fill=False, ls="--", color="black",
                           label="área da Figure 5"))
ax.set_xlabel("$\\mu_1$"); ax.set_ylabel("$\\mu_2$")
ax.set_title("Figure 4 — Exercise 2: μ das 10 000 imagens de teste (d = 2), cor = dígito")
ax.legend(title="dígito", markerscale=1.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.savefig(FIG / "fig4.png", bbox_inches="tight"); plt.close(fig)

# Sobreposição medida: um k-vizinhos no plano latente, ajustado com o treino e aplicado no teste.
# Os pares que ele mais confunde são os dígitos que se misturam na Figure 4.
knn = KNeighborsClassifier(n_neighbors=15).fit(mu_train, y_train.numpy())
pred = knn.predict(mu_test)
print(f"B) acurácia de um 15-vizinhos no plano latente (teste): {(pred == y_test.numpy()).mean():.4f}")
conf = np.zeros((10, 10), int)
np.add.at(conf, (y_test.numpy(), pred), 1)
pairs = sorted(((conf[i, j] + conf[j, i], i, j) for i in range(10) for j in range(i + 1, 10)), reverse=True)
print("   pares mais confundidos (erros nos dois sentidos):", [(f"{i}-{j}", n) for n, i, j in pairs[:6]])
centers = np.array([np.median(mu_test[y_test.numpy() == k], axis=0) for k in range(10)])
print("   mediana de μ por dígito:", {k: centers[k].round(2).tolist() for k in range(10)})

# ---------------------------------------------------------------- juiz auxiliar para as análises
# Classificador de dígitos de digit_judge.py: diz "isto parece o dígito k" nas imagens geradas.
judge = train_judge(X_train, y_train)
print(f"   classificador auxiliar: acurácia na validação {(judge(X_val)[0] == y_val.numpy()).mean():.4f}")

# ---------------------------------------------------------------- C — o plano decodificado
q = np.linspace(0.05, 0.95, 15)
zs = norm.ppf(q)                             # quantis da N(0, 1): cada faixa tem a mesma massa da prior
print(f"C) valores de z na grade: {zs.round(2).tolist()}")
Z1, Z2 = np.meshgrid(zs, zs[::-1])           # linha de cima = z2 maior, como no gráfico da Figure 4
grid_z = torch.tensor(np.stack([Z1.ravel(), Z2.ravel()], 1), dtype=torch.float32)
with torch.no_grad():
    grid_img = model2.decode(grid_z)
tile = grid_img.reshape(15, 15, 28, 28).permute(0, 2, 1, 3).reshape(15 * 28, 15 * 28)

fig, ax = plt.subplots(figsize=(9, 9.3))
ax.imshow(tile, cmap="gray", vmin=0, vmax=1)
ticks = np.arange(15) * 28 + 13.5
ax.set_xticks(ticks); ax.set_xticklabels([f"{v:.2f}" for v in zs], rotation=90)
ax.set_yticks(ticks); ax.set_yticklabels([f"{v:.2f}" for v in zs[::-1]])
ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$"); ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
ax.set_title("Figure 5 — Exercise 2: decoder de d = 2 numa grade 15×15\nnos quantis q = 0.05, ..., 0.95 da N(0, 1)")
fig.savefig(FIG / "fig5.png", bbox_inches="tight"); plt.close(fig)

lab_grid, conf_grid = judge(grid_img)
print("   dígito que o classificador vê em cada célula da Figure 5 (linha de cima = z2 maior):")
for r in range(15):
    print("   " + " ".join(f"{l}{'?' if c < 0.6 else ' '}" for l, c in
                         zip(lab_grid[r * 15:(r + 1) * 15], conf_grid[r * 15:(r + 1) * 15])))
print(f"   células com confiança < 0.6 (marcadas com ?): {(conf_grid < 0.6).sum()} de 225")
print(f"   dígitos que aparecem na grade: {sorted(set(lab_grid.tolist()))}; "
      f"células por dígito: {np.bincount(lab_grid, minlength=10).tolist()}")


def cell_of(z):
    """Célula (linha, coluna) da Figure 5 mais próxima de um ponto z, contando de 1."""
    col = int(np.abs(zs - z[0]).argmin()) + 1
    row = int(np.abs(zs[::-1] - z[1]).argmin()) + 1
    return row, col


print("   célula da Figure 5 mais próxima da mediana de cada dígito:",
      {k: cell_of(centers[k]) for k in range(10)})

# ---------------------------------------------------------------- custo de d = 2 (validação)
with torch.no_grad():
    rec_val = {d: m.decode(m.encode(X_val)[0]) for d, m in [(2, model2), (16, model16)]}
yv = y_val.numpy()
for d in (16, 2):
    acc = (judge(rec_val[d])[0] == yv).mean()
    print(f"E2) d = {d:2d}: o classificador acerta o dígito em {acc:.4f} das reconstruções de validação")


def within_class_spread(imgs):
    """Distância quadrática média de cada imagem até a média do seu dígito (variação de estilo)."""
    return np.mean([((imgs[yv == k] - imgs[yv == k].mean(0)) ** 2).sum(1).mean().item() for k in range(10)])


s_orig = within_class_spread(X_val)
for d in (16, 2):
    print(f"    d = {d:2d}: variação dentro do dígito nas reconstruções = {within_class_spread(rec_val[d]) / s_orig:.1%} "
          f"da variação nas imagens originais")

# ---------------------------------------------------------------- D — interpolação
IA, IB = 3, 2                                # índices de teste: um 0 e um 1
print(f"D) índices de teste: a = {IA} (dígito {y_test[IA].item()}), b = {IB} (dígito {y_test[IB].item()})")
t = torch.linspace(0, 1, 10)[:, None]


def interpolate(xa, xb):
    """Linha de cima: mistura dos pixels. Linha de baixo: mistura dos mu do modelo d = 16, decodificada."""
    pix = (1 - t) * xa + t * xb
    with torch.no_grad():
        mua, mub = model16.encode(torch.stack([xa, xb]))[0]
        lat = model16.decode((1 - t) * mua + t * mub)
    return pix, lat


pix, lat = interpolate(X_test[IA], X_test[IB])

fig, axes = plt.subplots(2, 10, figsize=(13, 3.2))
for j in range(10):
    for row, imgs in enumerate([pix, lat]):
        axes[row, j].imshow(imgs[j].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
    axes[0, j].set_title(f"t = {t[j].item():.2f}", fontsize=9)
axes[0, 0].set_ylabel("pixels"); axes[1, 0].set_ylabel("latente\n(d = 16)")
fig.supxlabel("t (fração do caminho da imagem a até a imagem b)", y=-0.02)
fig.suptitle(f"Figure 6 — Exercise 2: interpolação da imagem de teste {IA} (dígito {y_test[IA].item()}) "
             f"até a {IB} (dígito {y_test[IB].item()})")
fig.savefig(FIG / "fig6.png", bbox_inches="tight"); plt.close(fig)

gray = lambda imgs: ((imgs > 0.1) & (imgs < 0.9)).float().mean(1).numpy() * 100   # % de pixels "cinza"
for name, imgs in [("pixels", pix), ("latente", lat)]:
    lab, c = judge(imgs)
    print(f"   {name:7s}: dígito visto {lab.tolist()}")
    print(f"            confiança   {np.round(c, 2).tolist()}")
    print(f"            % de pixels entre 0.1 e 0.9 {np.round(gray(imgs), 1).tolist()}")


# O mesmo teste nos 45 pares formados pelas primeiras imagens de teste de cada dígito (as da Figure 2).
# Só os 8 quadros do meio (t de 0.11 a 0.89) entram na média.
first_idx = [int((y_test == k).nonzero()[0]) for k in range(10)]
stats = {"pixels": {"cinza": [], "conf": []}, "latente": {"cinza": [], "conf": []}}
for i in range(10):
    for j in range(i + 1, 10):
        for name, imgs in zip(["pixels", "latente"], interpolate(X_test[first_idx[i]], X_test[first_idx[j]])):
            stats[name]["cinza"].append(gray(imgs[1:-1]).mean())
            stats[name]["conf"].append(judge(imgs[1:-1])[1].mean())
for name in stats:
    print(f"   45 pares, {name:7s}: pixels cinza nos quadros do meio {np.mean(stats[name]['cinza']):.1f}%, "
          f"confiança média do classificador {np.mean(stats[name]['conf']):.2f}")
print(f"   pares em que a linha latente tem menos pixels cinza que a de pixels: "
      f"{(np.array(stats['latente']['cinza']) < np.array(stats['pixels']['cinza'])).sum()} de 45")
