
"Linear probing utilities."
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

__all__ = ["layerwise_probe"]

def layerwise_probe(layer_representations: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_state: int = 0):
    n_layers = layer_representations.shape[1]
    accs = []
    for layer in range(n_layers):
        X = layer_representations[:, layer, :]
        y = labels
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
        clf.fit(X_tr, y_tr)
        accs.append(accuracy_score(y_te, clf.predict(X_te)))
    return accs
