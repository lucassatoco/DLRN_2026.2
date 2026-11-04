"""Exercise 1 — VAE com d = 16 no MNIST.

Gera figures/fig1.png, fig2.png e fig3.png e imprime os números do relatório.
Salva o modelo treinado em code/data/vae_d16.pt, que o Exercise 2 usa na interpolação.
Roda de qualquer diretório: python ex1_mnist.py
"""
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digit_judge import train_judge
from vae import DATA, EPOCHS, VAE, evaluate, load_mnist, train_vae

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)
torch.set_num_threads(4)                     # fixa o número de threads para o resultado não variar entre máquinas

# ---------------------------------------------------------------- A — dados
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()
print(f"A) treino {tuple(X_train.shape)}, validação {tuple(X_val.shape)}, teste {tuple(X_test.shape)}; "
      f"pixels em [{X_train.min():.0f}, {X_train.max():.0f}]")

# ---------------------------------------------------------------- B — modelo
D = 16
print(f"B) {VAE(D)}")
print(f"   parâmetros: {sum(p.numel() for p in VAE(D).parameters())}")

# ---------------------------------------------------------------- C — treino
t0 = time.time()
model, hist = train_vae(D, X_train, X_val)
print(f"C) treino de {EPOCHS} épocas em {time.time() - t0:.0f} s")
torch.save(model.state_dict(), DATA / "vae_d16.pt")

# O teste só é usado aqui, depois do treino, para os números finais
test_recon, test_kl = evaluate(model, X_test)
print(f"C) teste: reconstrução {test_recon:.2f}, KL {test_kl:.2f}, -ELBO {test_recon + test_kl:.2f} nats por imagem")
print(f"   validação: KL na época 1 = {hist['val_kl'][0]:.2f}, na época {EPOCHS} = {hist['val_kl'][-1]:.2f}")

epochs = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, term, label in [(axes[0], "recon", "reconstrução (BCE)"), (axes[1], "kl", "KL")]:
    ax.plot(epochs, hist[f"train_{term}"], "o-", ms=3, label="treino (média da época)")
    ax.plot(epochs, hist[f"val_{term}"], "s-", ms=3, label="validação (fim da época)")
    ax.set_xlabel("época"); ax.set_ylabel("nats por imagem")
    ax.set_title(label); ax.set_xticks(epochs[::2]); ax.legend()
fig.suptitle(f"Figure 1 — Exercise 1: os dois termos da perda por época (d = {D})")
fig.tight_layout()
fig.savefig(FIG / "fig1.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- D — reconstruções e amostras
model.eval()
first_idx = [int((y_test == k).nonzero()[0]) for k in range(10)]    # primeira imagem de teste de cada dígito
x10 = X_test[first_idx]
with torch.no_grad():
    mu10, _ = model.encode(x10)
    rec10 = model.decode(mu10)                                       # reconstrução a partir de mu, sem sortear z
print(f"D) índices de teste da Figure 2 (dígitos 0 a 9): {first_idx}")

fig, axes = plt.subplots(2, 10, figsize=(12, 2.9))
for k in range(10):
    for row, imgs in enumerate([x10, rec10]):
        axes[row, k].imshow(imgs[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axes[row, k].set_xticks([]); axes[row, k].set_yticks([])
    axes[0, k].set_title(f"dígito {k}", fontsize=9)
axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("reconstrução\n(decodifica μ)")
fig.supxlabel("dígito (primeira imagem de teste de cada um)", y=-0.02)
fig.suptitle("Figure 2 — Exercise 1: primeira imagem de teste de cada dígito e sua reconstrução (d = 16)")
fig.savefig(FIG / "fig2.png", bbox_inches="tight"); plt.close(fig)

torch.manual_seed(0)
samples = model.sample(64)                                           # z ~ N(0, I), depois o decoder
grid = samples.reshape(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
fig, ax = plt.subplots(figsize=(6, 6.3))
ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
ax.set_xticks(np.arange(8) * 28 + 13.5); ax.set_xticklabels(range(1, 9))
ax.set_yticks(np.arange(8) * 28 + 13.5); ax.set_yticklabels(range(1, 9))
ax.set_xlabel("coluna da grade"); ax.set_ylabel("linha da grade")
ax.set_title("Figure 3 — Exercise 1: 64 amostras de z ~ N(0, I)\ndecodificadas (d = 16, torch.manual_seed(0))")
fig.savefig(FIG / "fig3.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- E — números para a análise (validação)
# 1) KL subindo no começo: quanto a reconstrução caiu enquanto o KL subia
print(f"E1) validação antes do treino: reconstrução {hist['init_recon']:.2f}, KL {hist['init_kl']:.2f}")
print(f"    KL de validação por época: {np.round(hist['val_kl'], 2).tolist()}")
print(f"    reconstrução de validação por época: {np.round(hist['val_recon'], 2).tolist()}")
k = int(np.argmax(np.diff(hist["val_kl"]) < 0))          # última época da subida inicial do KL (índice)
r0, r1 = hist["init_recon"], hist["val_recon"][k]
k0, k1 = hist["init_kl"], hist["val_kl"][k]
print(f"    do início à época {k + 1} (fim da subida contínua): KL {k0:.2f} -> {k1:.2f} ({k1 - k0:+.2f}), "
      f"reconstrução {r0:.2f} -> {r1:.2f} ({r1 - r0:+.2f}), total {r0 + k0:.2f} -> {r1 + k1:.2f}")
print(f"    da época 1 à época {k + 1}: KL {hist['val_kl'][0]:.2f} -> {k1:.2f} ({k1 - hist['val_kl'][0]:+.2f}), "
      f"reconstrução {hist['val_recon'][0]:.2f} -> {r1:.2f} ({r1 - hist['val_recon'][0]:+.2f})")

with torch.no_grad():
    mu_v, logvar_v = model.encode(X_val)
kl_dim = 0.5 * (mu_v.pow(2) + logvar_v.exp() - 1 - logvar_v).mean(0)  # KL médio por dimensão latente
print(f"E2) KL médio por dimensão (validação): {np.round(kl_dim.numpy(), 2).tolist()}")
active = kl_dim >= 0.1
print(f"    dimensões ativas (KL >= 0.1 nat): {active.sum().item()} de {D}")

# 2) Onde as amostras da prior caem: distância até o código mu de imagem real mais próximo, só nas
#    dimensões ativas. Referência: mu das 5 000 primeiras imagens de validação. Para comparar, faço a mesma
#    conta com códigos z ~ q(z|x) de outras 5 000 imagens de validação (que não estão na referência).
ref = mu_v[:5000, active]
gen = torch.Generator().manual_seed(1)
with torch.no_grad():
    z_post = model.reparameterize(mu_v[5000:], logvar_v[5000:], gen)   # códigos de imagens reais
    torch.manual_seed(0)
    z_prior = torch.randn(64, D)                                       # os mesmos z da Figure 3
d_post = torch.cdist(z_post[:, active], ref).min(1).values
d_prior = torch.cdist(z_prior[:, active], ref).min(1).values
thr = torch.quantile(d_post, 0.95)
far = d_prior > thr                                                    # amostras "longe" de qualquer imagem real
print(f"E2) distância mediana até o mu real mais próximo: códigos de imagens reais {d_post.median():.2f}, "
      f"amostras da prior {d_prior.median():.2f}")
print(f"    amostras da Figure 3 acima do quantil 95% dos códigos reais ({thr:.2f}): {far.sum().item()} de 64")

# 3) O decoder devolve probabilidades: fração de pixels "cinza" (entre 0.1 e 0.9)
gray = lambda imgs: ((imgs > 0.1) & (imgs < 0.9)).float().mean().item() * 100
print(f"E2) pixels entre 0.1 e 0.9: originais da Figure 2 {gray(x10):.1f}%, reconstruções {gray(rec10):.1f}%, "
      f"amostras da Figure 3 {gray(samples):.1f}%")
print(f"    nas amostras da Figure 3: {gray(samples[far]):.1f}% nas {far.sum().item()} longe, "
      f"{gray(samples[~far]):.1f}% nas outras {(~far).sum().item()}")
far_cells = [(i // 8 + 1, i % 8 + 1) for i in far.nonzero().flatten().tolist()]
print(f"    posições (linha, coluna) das amostras longe: {far_cells}")

# 3b) Parece um dígito? Confiança do classificador auxiliar (digit_judge.py)
judge = train_judge(X_train, y_train)
_, c_rec = judge(rec10)
_, c_smp = judge(samples)
f = far.numpy()
print(f"    confiança média do classificador: reconstruções da Figure 2 {c_rec.mean():.2f}, "
      f"amostras da Figure 3 {c_smp.mean():.2f} (longe {c_smp[f].mean():.2f}, outras {c_smp[~f].mean():.2f})")
print(f"    amostras da Figure 3 com confiança < 0.6: {(c_smp < 0.6).sum()} de 64 ({(c_smp[f] < 0.6).sum()} delas longe)")

# 4) Decodificar mu ou um z sorteado
rec_mu = evaluate(model, X_val, use_mean=True)[0]
rec_z = evaluate(model, X_val)[0]
print(f"E3) reconstrução de validação decodificando mu {rec_mu:.2f} vs z sorteado {rec_z:.2f} nats por imagem")
print(f"    sigma médio no encoder: {torch.exp(0.5 * logvar_v).mean():.3f}; |mu| médio: {mu_v.abs().mean():.3f}")
