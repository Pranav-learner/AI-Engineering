# ✂️ Module 17: Feature Selection & Dimensionality Reduction

---

## 1. Why Feature Selection Matters

Adding hundreds of noisy or redundant features increases training time, creates the "Curse of Dimensionality", and causes overfitting.

### 1.1 Methods for Feature Selection
1. **Correlation Pruning**: Remove one feature from any pair with $|r| > 0.90$.
2. **Model-Based Importance (Gini / Tree Importance)**: Use Random Forest Gini feature importances.
3. **Recursive Feature Elimination (RFE)**: Iteratively trains models and prunes the least important weights.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# 1. Tree Feature Importances
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("Top 5 Features:\n", importances.head(5))

# 2. RFE (Select top 5 features automatically)
rfe = RFE(estimator=rf, n_features_to_select=5)
rfe.fit(X_train, y_train)
selected_cols = X_train.columns[rfe.support_]
print("Selected by RFE:", selected_cols)
```

---

## 2. Principal Component Analysis (PCA)

PCA rotates data to align with directions of maximum variance (eigenvectors of covariance matrix):
$$\mathbf{Z} = \mathbf{X} \mathbf{V}_k$$

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train_scaled)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
```
