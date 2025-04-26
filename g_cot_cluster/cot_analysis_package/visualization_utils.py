
"""Plotting helpers for the experiments."""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
import hdbscan

__all__ = [
    "umap_projection",
    "plot_umap",
    "run_kmeans",
    "run_hdbscan",
    "plot_linear_probe_results",
]

def umap_projection(embeddings: np.ndarray,
                    n_neighbors: int = 30,
                    min_dist: float = 0.1,
                    random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        random_state=random_state)
    return reducer.fit_transform(embeddings)

def plot_umap(embeddings2d: np.ndarray,
              labels: list[str],
              categories: list[str],
              title: str = "UMAP of Segment Embeddings"):
    plt.figure(figsize=(8, 6))
    # map categories to ints for colour map
    cat2idx = {c: i for i, c in enumerate(categories)}
    colours = [cat2idx[l] for l in labels]
    scatter = plt.scatter(embeddings2d[:, 0], embeddings2d[:, 1],
                          c=colours, s=12, alpha=0.7)
    handles = [plt.Line2D([], [], marker="o", linestyle="",
                          label=c,
                          color=scatter.cmap(scatter.norm(cat2idx[c])))
               for c in categories]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1),
               loc="upper left", frameon=False)
    plt.title(title)
    plt.xlabel("UMAP‑1")
    plt.ylabel("UMAP‑2")
    plt.tight_layout()

def run_kmeans(embeddings: np.ndarray, n_clusters: int = 12):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    km.fit(embeddings)
    return km

def run_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 30):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusterer.fit(embeddings)
    return clusterer

def plot_linear_probe_results(layers: list[int], accuracies: list[float],
                              title: str = "Linear probe accuracy by layer"):
    plt.figure(figsize=(6, 4))
    plt.plot(layers, accuracies, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
