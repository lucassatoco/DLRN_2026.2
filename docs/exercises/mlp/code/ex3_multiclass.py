"""Exercise 3 — multiclasse com a mesma rede do Exercise 2.

Gera figures/fig4.png e fig5.png e imprime os números do relatório.
Roda de qualquer diretório: python ex3_multiclass.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from mlp import MLP                      # mesma classe do Exercise 2, sem nenhuma alteração
from multiclass_data import load_multiclass, one_hot

rng = np.random.default_rng(42)
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

# ---------------------------------------------------------------- A — dados
X_train, X_test, y_train, y_test = load_multiclass()
print(f"A) treino {X_train.shape}, teste {X_test.shape}, amostras por classe no teste: {np.bincount(y_test)}")

# ---------------------------------------------------------------- C — treino (a rede é a do Exercise 2)
SIZES, LR, EPOCHS = [4, 16, 3], 0.5, 2000
model = MLP(SIZES, output="softmax", lr=LR, rng=rng)
history = model.fit(X_train, one_hot(y_train), EPOCHS)
pred = model.predict(X_test)
acc_train = (model.predict(X_train) == y_train).mean()
acc_test = (pred == y_test).mean()
print(f"C) arquitetura {SIZES} (tanh na oculta, softmax na saída), lr = {LR}, {EPOCHS} épocas, "
      f"{model.n_params()} parâmetros")
print(f"   perda de treino: inicial {history[0]:.4f}, final {history[-1]:.4f}")
print(f"   acurácia: treino {acc_train:.4f}, teste {acc_test:.4f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, EPOCHS + 1), history, label="perda de treino (entropia cruzada)")
ax.set_xlabel("época"); ax.set_ylabel("entropia cruzada categórica")
ax.set_title(f"Figure 4 — Exercise 3: perda de treino por época, MLP {SIZES}")
ax.legend()
fig.savefig(FIG / "fig4.png", bbox_inches="tight"); plt.close(fig)

cm = confusion_matrix(y_test, pred)
print("   matriz de confusão (linha = real, coluna = prevista):")
print(cm)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels([f"classe {k}" for k in range(3)])
ax.set_yticklabels([f"classe {k}" for k in range(3)])
ax.set_xlabel("classe prevista"); ax.set_ylabel("classe real")
ax.set_title(f"Figure 5 — Exercise 3: matriz de confusão\n(teste, acurácia {acc_test:.1%})")
fig.colorbar(im, ax=ax, label="nº de amostras")
fig.savefig(FIG / "fig5.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- D — par mais confundido
pairs = {(i, j): cm[i, j] + cm[j, i] for i in range(3) for j in range(i + 1, 3)}
worst = max(pairs, key=pairs.get)
print(f"D) erros por par de classes (nos dois sentidos): {pairs}; mais confundido: {worst}")


# Estrutura dos clusters: com shuffle=False o make_classification devolve os mesmos pontos e rótulos
# (só a ordem das linhas e das colunas muda, o que não altera distâncias), agrupados em blocos de 250
# por cluster (6 clusters = 2 por classe). Isso permite medir onde cada cluster está.
from sklearn.datasets import make_classification
X_all, y_all = make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
                                   n_repeated=0, n_classes=3, n_clusters_per_class=2,
                                   class_sep=1.0, random_state=42, shuffle=False)
cluster_id = np.repeat(np.arange(6), 250)
centers = np.vstack([X_all[cluster_id == k].mean(0) for k in range(6)])
cluster_class = [int(np.bincount(y_all[cluster_id == k]).argmax()) for k in range(6)]
print("   centros dos 6 clusters (classe: centro):")
for k in range(6):
    print(f"   cluster {k} (classe {cluster_class[k]}): {centers[k].round(2)}")
print("   pares de clusters de classes diferentes a menos de 2.5 de distância (vizinhos no hipercubo):")
for i in range(6):
    for j in range(i + 1, 6):
        d = np.linalg.norm(centers[i] - centers[j])
        if cluster_class[i] != cluster_class[j] and d < 2.5:
            print(f"   cluster {i} (classe {cluster_class[i]}) - cluster {j} (classe {cluster_class[j]}): {d:.2f}")
for (i, j) in pairs:
    d = min(np.linalg.norm(centers[a] - centers[b]) for a in range(6) for b in range(6)
            if {cluster_class[a], cluster_class[b]} == {i, j})
    print(f"   classes {i} e {j}: clusters mais próximos a {d:.2f}")
print(f"   rótulos trocados pelo flip_y padrão (1%): {(y_all != np.array(cluster_class)[cluster_id]).sum()} de 1500")
spread = [np.linalg.norm(X_all[cluster_id == k] - centers[k], axis=1).mean() for k in range(6)]
print(f"   distância média de um ponto ao centro do próprio cluster: {np.round(spread, 2).tolist()}")
mid_14 = (centers[1] + centers[4]) / 2
print(f"   cluster 2 (classe 2) até o ponto médio dos clusters 1 e 4 (classe 1): {np.linalg.norm(centers[2] - mid_14):.2f}; "
      f"metade da distância entre 1 e 4: {np.linalg.norm(centers[1] - centers[4]) / 2:.2f}")

logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(f"D) regressão logística (baseline linear): acurácia no teste = {logreg.score(X_test, y_test):.4f}")
