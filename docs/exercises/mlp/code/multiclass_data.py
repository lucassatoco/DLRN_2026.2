"""Dados dos Exercises 3 e 4 (os dois usam exatamente o mesmo conjunto e a mesma divisão)."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


def load_multiclass():
    """1500 amostras, 4 features informativas, 3 classes, 2 clusters por classe; divisão 80/20 estratificada."""
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4, n_redundant=0,
                               n_repeated=0, n_classes=3, n_clusters_per_class=2,
                               class_sep=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def one_hot(y, n_classes=3):
    return np.eye(n_classes)[y]
