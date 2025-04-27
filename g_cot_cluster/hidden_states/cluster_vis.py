# scripts/cluster.py
import torch, umap, hdbscan, sklearn
from sklearn.metrics import silhouette_score
from utils.load_cache import load_cache   # returns pandas DF

df = load_cache("caches/layer_last.pt", hint_cond="sycophancy")  # one condition

X = torch.stack(df.vec.tolist()).numpy()
labels_true = df.phrase_category.astype("category").cat.codes

# ★ choose algorithm
clusterer = hdbscan.HDBSCAN(min_cluster_size=30).fit(X)
labels_pred = clusterer.labels_

sil = silhouette_score(X, labels_pred)  # store per condition
