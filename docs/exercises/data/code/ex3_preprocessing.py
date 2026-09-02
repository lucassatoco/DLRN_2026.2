"""Exercise 3 - Preparing real-world data for a neural network.

Pre-processa o Spaceship Titanic (train.csv) para uma rede com ativacao tanh:
imputacao, one-hot, TotalSpend, log(1+x) e escala para [-1, 1]. Salva a Figura 6
e imprime todos os numeros da Results summary.

O arquivo train.csv deve estar em docs/exercises/data/dataset/train.csv
(baixe em https://www.kaggle.com/competitions/spaceship-titanic/data).

Uso:  python ex3_preprocessing.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

rng = np.random.default_rng(42)
SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
CSV_PATH = os.path.join(HERE, "..", "dataset", "train.csv")
os.makedirs(FIG_DIR, exist_ok=True)

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUM_COLS = ["Age"] + SPEND_COLS
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP_COLS = ["PassengerId", "Name", "Cabin"]
TARGET = "Transported"


def describe(df):
    """Item A: alvo, balanco, tipos, faltantes e estatisticas de gasto."""
    print("A.2 - Balanco de classes")
    for k, v in df[TARGET].value_counts(normalize=True).sort_index().items():
        print(f"  {str(k):<6} {v:.4f}  ({int(v * len(df))} passageiros)")

    print("\nA.3 - Tipos de feature")
    print(f"  numericas   ({len(NUM_COLS)}): {NUM_COLS}")
    print(f"  categoricas ({len(CAT_COLS)}): {CAT_COLS}")
    print(f"  descartadas ({len(DROP_COLS)}): {DROP_COLS}")

    print("\nA.4 - Valores faltantes")
    missing = pd.DataFrame({
        "faltantes": df.isna().sum(),
        "pct": (100 * df.isna().mean()).round(2),
    }).sort_values("faltantes", ascending=False)
    print(missing.to_string())
    print(f"  linhas com ao menos um NaN: {df.isna().any(axis=1).sum()} "
          f"({100 * df.isna().any(axis=1).mean():.1f}%)")

    print("\nA.5 - Colunas de gasto")
    stats = df[SPEND_COLS].agg(["mean", "median", "max", "std"]).T
    stats["pct_zero"] = [100 * (df[c] == 0).mean() for c in SPEND_COLS]
    print(stats.round(2).to_string())
    return missing, stats


def figure_6(raw_fc, scaled_fc, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].hist(raw_fc, bins=60, edgecolor="white", label="FoodCourt (raw)")
    axes[0].set_title("FoodCourt - before preprocessing")
    axes[0].set_xlabel("credits spent")
    axes[0].set_ylabel("count (log scale)")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[0].annotate(f"max = {raw_fc.max():,.0f}", xy=(0.55, 0.85),
                     xycoords="axes fraction", fontsize=11)

    axes[1].hist(scaled_fc, bins=60, edgecolor="white", color="seagreen",
                 label="FoodCourt (processed)")
    axes[1].set_title("FoodCourt - after log1p + MinMax to $[-1, 1]$")
    axes[1].set_xlabel("scaled value")
    axes[1].set_ylabel("count")
    axes[1].axvline(-1, color="grey", linestyle=":")
    axes[1].axvline(1, color="grey", linestyle=":")
    axes[1].legend(fontsize=9)

    fig.suptitle("Figure 6 - Effect of preprocessing on a heavy-tailed feature",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"train.csv nao encontrado em {os.path.normpath(CSV_PATH)}. "
            "Baixe o Spaceship Titanic no Kaggle e coloque o arquivo la."
        )

    df = pd.read_csv(CSV_PATH)
    print(f"{df.shape[0]} linhas x {df.shape[1]} colunas\n")
    describe(df)

    # ---- B: split antes de qualquer estatistica
    X_raw = df.drop(columns=DROP_COLS + [TARGET])
    y_raw = df[TARGET].astype(int)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.20, stratify=y_raw, random_state=SEED
    )
    print("\nB - Split 80/20 estratificado")
    print(f"  treino {X_tr_raw.shape}  taxa positiva {y_tr.mean():.4f}")
    print(f"  teste  {X_te_raw.shape}  taxa positiva {y_te.mean():.4f}")
    fc_mean = float(X_tr_raw["FoodCourt"].mean())
    fc_median = float(X_tr_raw["FoodCourt"].median())
    print(f"  FoodCourt no treino, antes de transformar: "
          f"media {fc_mean:.4f} | mediana {fc_median:.4f}")

    # ---- C.1: imputacao, ajustada somente no treino
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer.fit(X_tr_raw[NUM_COLS])
    cat_imputer.fit(X_tr_raw[CAT_COLS].astype(object))

    def impute(X):
        out = X.copy()
        out[NUM_COLS] = num_imputer.transform(X[NUM_COLS])
        out[CAT_COLS] = cat_imputer.transform(X[CAT_COLS].astype(object))
        return out

    X_tr, X_te = impute(X_tr_raw), impute(X_te_raw)
    print("\nC.1 - Imputacao (valores aprendidos no treino)")
    for c, v in zip(NUM_COLS, num_imputer.statistics_):
        print(f"  {c:<14} mediana -> {v}")
    for c, v in zip(CAT_COLS, cat_imputer.statistics_):
        print(f"  {c:<14} moda    -> {v}")

    # ---- C.2: one-hot; handle_unknown='ignore' cobre categoria so no teste
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(X_tr[CAT_COLS].astype(str))
    ohe_names = list(ohe.get_feature_names_out(CAT_COLS))
    tr_cat = pd.DataFrame(ohe.transform(X_tr[CAT_COLS].astype(str)),
                          columns=ohe_names, index=X_tr.index)
    te_cat = pd.DataFrame(ohe.transform(X_te[CAT_COLS].astype(str)),
                          columns=ohe_names, index=X_te.index)
    print(f"\nC.2 - One-hot: {len(CAT_COLS)} colunas -> {len(ohe_names)} indicadoras")
    demo = X_te[CAT_COLS].astype(str).head(1).copy()
    demo.iloc[0, 0] = "Pluto"
    n_hp = sum(n.startswith("HomePlanet") for n in ohe_names)
    print(f"  categoria inedita 'Pluto' -> soma do bloco HomePlanet = "
          f"{ohe.transform(demo)[0][:n_hp].sum():.0f}")

    # ---- C.3: TotalSpend
    for X in (X_tr, X_te):
        X["TotalSpend"] = X[SPEND_COLS].sum(axis=1)
    engineered = SPEND_COLS + ["TotalSpend"]

    # ---- C.4: log(1 + x)
    raw_fc = X_tr["FoodCourt"].copy()
    skew_antes = raw_fc.skew()
    for X in (X_tr, X_te):
        for c in engineered:
            X[c] = np.log1p(X[c])
    print(f"\nC.4 - log1p: assimetria de FoodCourt {skew_antes:.3f} -> "
          f"{X_tr['FoodCourt'].skew():.3f}")

    # ---- C.5: escala para [-1, 1]
    numeric_final = ["Age"] + engineered
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_tr[numeric_final])
    tr_num = pd.DataFrame(scaler.transform(X_tr[numeric_final]),
                          columns=numeric_final, index=X_tr.index)
    te_num = pd.DataFrame(scaler.transform(X_te[numeric_final]),
                          columns=numeric_final, index=X_te.index)
    X_train_final = pd.concat([tr_num, tr_cat], axis=1)
    X_test_final = pd.concat([te_num, te_cat], axis=1)

    tr_min, tr_max = X_train_final.values.min(), X_train_final.values.max()
    te_min, te_max = X_test_final.values.min(), X_test_final.values.max()
    print(f"\nC.5 - Escala: treino [{tr_min:.4f}, {tr_max:.4f}] | "
          f"teste [{te_min:.4f}, {te_max:.4f}]")

    # ---- D
    figure_6(raw_fc, X_train_final["FoodCourt"], os.path.join(FIG_DIR, "fig6.png"))

    checks = {
        "sem NaN no treino": int(X_train_final.isna().sum().sum()) == 0,
        "sem NaN no teste": int(X_test_final.isna().sum().sum()) == 0,
        "mesmas colunas nos dois": list(X_train_final.columns) == list(X_test_final.columns),
        "valores finitos": bool(np.isfinite(X_train_final.values).all()
                                and np.isfinite(X_test_final.values).all()),
        "treino em [-1, 1]": tr_min >= -1.0001 and tr_max <= 1.0001,
        "teste compativel com tanh": max(abs(te_min), abs(te_max)) < 2.0,
    }
    print("\nD.2 - Checagens finais")
    for nome, ok in checks.items():
        print(f"  [{'OK' if ok else 'FALHOU'}] {nome}")
    print(f"  shape treino: {X_train_final.shape}")
    print(f"  shape teste:  {X_test_final.shape}")

    print("\n=== Valores para a Results summary ===")
    print(f"  10 - share positiva em Transported: {df[TARGET].mean():.4f}")
    print(f"  11 - FoodCourt no treino: media {fc_mean:.4f} | mediana {fc_median:.4f}")
    print(f"  12 - shape do treino: {X_train_final.shape}")
    print(f"  13 - treino [{tr_min:.4f}, {tr_max:.4f}] | teste [{te_min:.4f}, {te_max:.4f}]")


if __name__ == "__main__":
    main()
