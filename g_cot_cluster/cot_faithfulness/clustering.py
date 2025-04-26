
"Clustering & visualisation utilities."
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt

__all__ = ["cluster_kmeans", "embed_to_umap", "plot_clusters_2d"]

def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 12, random_state: int = 0):
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(embeddings)
    sil = silhouette_score(embeddings, labels)
    return labels, sil

def embed_to_umap(embeddings: np.ndarray, random_state: int = 0):
    reducer = umap.UMAP(random_state=random_state)
    return reducer.fit_transform(embeddings)

def plot_clusters_2d(coords: np.ndarray, labels, title: str):
    fig, ax = plt.subplots(figsize=(6,5))
    sc = ax.scatter(coords[:,0], coords[:,1], c=labels, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    return fig
