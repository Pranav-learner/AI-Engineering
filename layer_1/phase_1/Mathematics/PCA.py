import numpy as np

A = np.array([
    [2, 0],
    [0, 3]
])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
print(eigenvectors)


X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
], dtype=float)

X_centered = X - X.mean(axis=0)

print(X_centered)

covariance = np.cov(X_centered, rowvar=False)

print("Covariance matrix:\n", covariance)

eigenvalues, eigenvectors = np.linalg.eig(covariance)

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
print(eigenvectors)

from sklearn.decomposition import PCA

X = np.array([
    [1, 2],
    [2, 4],
    [3, 6],
    [4, 8],
    [5, 10]
], dtype=float)

pca = PCA(n_components=1)

X_reduced = pca.fit_transform(X)

print("Reduced:")
print(X_reduced)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)