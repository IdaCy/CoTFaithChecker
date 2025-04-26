
"Contrastive PCA helpers."
import numpy as np
import matplotlib.pyplot as plt
from contrastive import CPCA

__all__ = ["run_cpca", "plot_cpca_projection"]

def run_cpca(background: np.ndarray, target: np.ndarray, n_components: int = 2):
    model = CPCA(n_components=n_components)
    model.fit(target, background)
    return model

def plot_cpca_projection(model: "CPCA", background, target, title):
    bg = model.transform(background)
    tg = model.transform(target)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(bg[:,0], bg[:,1], s=5, alpha=0.4, label="background")
    ax.scatter(tg[:,0], tg[:,1], s=5, alpha=0.7, label="target")
    ax.set_xlabel("cPC-1"); ax.set_ylabel("cPC-2"); ax.set_title(title); ax.legend()
    return fig
