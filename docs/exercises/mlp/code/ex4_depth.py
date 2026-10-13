"""Exercise 4 — uma segunda camada oculta ajuda?

Mesmos dados, mesma divisão, mesma taxa de aprendizado e mesmo número de épocas do
Exercise 3. A única mudança é a profundidade. Gera figures/fig6.png e imprime a tabela.
Roda de qualquer diretório: python ex4_depth.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlp import MLP
from multiclass_data import load_multiclass, one_hot

FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(exist_ok=True)

X_train, X_test, y_train, y_test = load_multiclass()
Y_train = one_hot(y_train)

LR, EPOCHS = 0.5, 2000                       # mesmo orçamento do Exercise 3
ARCHS = {"Exercise 3 (1 camada oculta)": [4, 16, 3],
         "Exercise 4 (2 camadas ocultas)": [4, 16, 16, 3]}
SEEDS = [42, 43, 44]                         # seed 42 é a execução principal; 43 e 44 entram na média


def run(sizes, seed):
    """Treina uma rede com pesos iniciais vindos de default_rng(seed)."""
    rng = np.random.default_rng(seed)
    model = MLP(sizes, output="softmax", lr=LR, rng=rng)
    history = model.fit(X_train, Y_train, EPOCHS)
    return model, history, (model.predict(X_test) == y_test).mean(), (model.predict(X_train) == y_train).mean()


rows, curves = [], {}
for name, sizes in ARCHS.items():
    accs, losses, accs_train = [], [], []
    for seed in SEEDS:
        model, history, acc, acc_train = run(sizes, seed)
        accs.append(acc); losses.append(history[-1]); accs_train.append(acc_train)
        if seed == 42:
            curves[name] = history
            print(f"A) {name}: {sizes}, seed 42 -> perda final {history[-1]:.4f}, acurácia no teste {acc:.4f}")
    print(f"   {name}: acurácias por seed {np.round(accs, 4).tolist()}, perdas finais {np.round(losses, 4).tolist()}")
    rows.append({"modelo": name, "camadas ocultas": len(sizes) - 2, "parâmetros": model.n_params(),
                 "perda final de treino (seed 42)": losses[0], "acurácia no teste (seed 42)": accs[0],
                 "perda final de treino (média 3 seeds)": np.mean(losses),
                 "acurácia no treino (média 3 seeds)": np.mean(accs_train),
                 "acurácia no teste (média 3 seeds)": np.mean(accs),
                 "acurácia no teste (desvio 3 seeds)": np.std(accs)})

table = pd.DataFrame(rows).set_index("modelo")
print("B) tabela de comparação:")
print(table.round(4).to_string())
diff = rows[1]["acurácia no teste (média 3 seeds)"] - rows[0]["acurácia no teste (média 3 seeds)"]
print(f"B) diferença de acurácia (Ex. 4 - Ex. 3), média de 3 seeds: {diff * 100:+.2f} p.p.")

# Estabilidade do treino (seed 42): em quantas épocas a perda subiu em vez de cair
for name, history in curves.items():
    rises = np.diff(history)
    msg = f"C) {name}: a perda subiu em {(rises > 0).sum()} de {EPOCHS} épocas"
    if (rises > 0).any():
        k = rises.argmax()                      # maior subida: da época k+1 para a época k+2
        msg += f"; maior subida na época {k + 2}: {history[k]:.3f} -> pico {history[k:k + 5].max():.3f}"
    print(msg)

fig, ax = plt.subplots(figsize=(8, 4.5))
for name, history in curves.items():
    ax.plot(np.arange(1, EPOCHS + 1), history, label=f"{name}: {ARCHS[name]}")
ax.set_xlabel("época"); ax.set_ylabel("entropia cruzada categórica")
ax.set_title("Figure 6 — Exercises 3 e 4: perda de treino por época (seed 42)")
ax.legend()
fig.savefig(FIG / "fig6.png", bbox_inches="tight"); plt.close(fig)
