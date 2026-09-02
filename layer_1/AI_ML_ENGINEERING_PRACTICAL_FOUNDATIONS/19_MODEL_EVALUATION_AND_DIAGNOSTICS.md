# 📊 Module 19: Model Evaluation & Diagnostic Thinking

---

## 1. Regression Metrics Suite

```python
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def evaluate_regressor(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}
```

---

## 2. Classification Metrics Suite

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_classifier(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred)
    }
```

---

## 3. Diagnostic Thinking: The 3 Health Checks

1. **Train Score vs. Test Score**:
   - High Train ($R^2=0.98$), Low Test ($R^2=0.60$) $\implies$ **Severe Overfitting**.
   - Low Train ($R^2=0.55$), Low Test ($R^2=0.52$) $\implies$ **Underfitting**.
2. **Residual Plot**: Plot $y_{\text{true}} - y_{\text{pred}}$. Residuals should be random Gaussian noise centered at 0.
3. **Class Breakdown**: Check recall per class to ensure rare risk classes are not ignored.
