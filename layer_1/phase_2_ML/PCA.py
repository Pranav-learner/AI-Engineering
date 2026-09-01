from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_reduced = pca.fit_predict(X)

print(pca.explained_variance_ratio_)

# PCA is sensitive to feature scales.

'''Raw features
     ↓
StandardScaler
     ↓
PCA

is often appropriate.'''

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)

X_reduced = pca.fit_transform(X_scaled)