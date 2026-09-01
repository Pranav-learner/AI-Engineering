from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate data first
X, _ = make_blobs(
    n_samples=500,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# 2. Initialize model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init="auto"
)

# 3. Fit and get labels
labels = model.fit_predict(X)
centers = model.cluster_centers_

# Elbow method




from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X)

#DBSCAN

from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps =0.5,
    min_samples=5
)

labels = model.fit_predict(X)

## GAUSSIAN MIXTURE MODELS
from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=3,
    random_state=42
)

model.fit(X)

labels = model.predict(X)
probabilities = model.predict_proba(X)