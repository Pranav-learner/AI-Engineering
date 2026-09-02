# 🧪 Module 16: Preprocessing & Production Pipelines

---

## 1. CONCEPT: Building End-to-End Leak-Proof Pipelines

A **Pipeline** chains transformers and models into a single atomic object:
$$\text{Raw Features} \longrightarrow \text{Imputer} \longrightarrow \text{Scaler} \longrightarrow \text{OneHotEncoder} \longrightarrow \text{Model}$$

---

## 2. Using `ColumnTransformer` for Heterogeneous Features

Different column types require different transformations:
- **Numerical Features** $\implies$ Impute Median $\to$ `StandardScaler`
- **Categorical Features** $\implies$ Impute Mode $\to$ `OneHotEncoder`

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

num_cols = ["income", "expense", "runway_days"]
cat_cols = ["user_profile", "city"]

# 1. Numerical Pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 2. Categorical Pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 3. Combined Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# 4. Full Production Pipeline with Model
full_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and Predict in 2 lines with ZERO data leakage!
full_pipeline.fit(X_train, y_train)
preds = full_pipeline.predict(X_test)
```
