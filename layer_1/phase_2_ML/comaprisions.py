import time
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================
# 1. Dataset Preparation
# ==========================================
print("Generating dataset...")
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_clusters_per_class=2,
    flip_y=0.03,  # 3% label noise to test robustness
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, Features: {X_train.shape[1]}")

# ==========================================
# 2. Define Model Zoo
# ==========================================
# Note: Scale-sensitive models are wrapped in a Pipeline with StandardScaler
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "k-NN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB())
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=False
    )
}

# ==========================================
# 3. Benchmark Pipeline
# ==========================================
results = []

print("\nRunning benchmark across 9 models...\n")

for name, model in models.items():
    # Measure Training Time
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    # Measure Inference Time (Latency for entire test set & per-sample)
    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = y_pred
    infer_time_total = (time.perf_counter() - start_infer) * 1000  # in ms
    infer_time_per_sample_us = (infer_time_total / len(X_test)) * 1000  # in microseconds

    # Metric Evaluations
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    # Model Size in Memory (Serialized size via pickle in KB)
    model_bytes = pickle.dumps(model)
    model_size_kb = len(model_bytes) / 1024

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc,
        "Train Time (s)": train_time,
        "Infer Latency (ms)": infer_time_total,
        "Latency/Sample (µs)": infer_time_per_sample_us,
        "Model Size (KB)": model_size_kb
    })

# ==========================================
# 4. Display Comparison Table
# ==========================================
df_results = pd.DataFrame(results)

# Format numerical columns for crisp presentation
df_display = df_results.copy()
df_display["Accuracy"] = df_display["Accuracy"].apply(lambda x: f"{x:.4f}")
df_display["Precision"] = df_display["Precision"].apply(lambda x: f"{x:.4f}")
df_display["Recall"] = df_display["Recall"].apply(lambda x: f"{x:.4f}")
df_display["F1 Score"] = df_display["F1 Score"].apply(lambda x: f"{x:.4f}")
df_display["ROC-AUC"] = df_display["ROC-AUC"].apply(lambda x: f"{x:.4f}")
df_display["Train Time (s)"] = df_display["Train Time (s)"].apply(lambda x: f"{x:.4f}s")
df_display["Infer Latency (ms)"] = df_display["Infer Latency (ms)"].apply(lambda x: f"{x:.2f}ms")
df_display["Latency/Sample (µs)"] = df_display["Latency/Sample (µs)"].apply(lambda x: f"{x:.2f}µs")
df_display["Model Size (KB)"] = df_display["Model Size (KB)"].apply(lambda x: f"{x:.1f} KB")

# Sort by F1 Score descending
df_display = df_display.sort_values(by="F1 Score", ascending=False).reset_index(drop=True)

print("=" * 105)
print("BENCHMARK EXPERIMENT RESULTS (Sorted by F1 Score)")
print("=" * 105)
print(df_display.to_string(index=False))
print("=" * 105)

# ==========================================
# 5. Key Trade-off Takeaways
# ==========================================
best_f1_model = df_results.loc[df_results["F1 Score"].idxmax()]["Model"]
fastest_infer_model = df_results.loc[df_results["Infer Latency (ms)"].idxmin()]["Model"]
smallest_model = df_results.loc[df_results["Model Size (KB)"].idxmin()]["Model"]
fastest_train_model = df_results.loc[df_results["Train Time (s)"].idxmin()]["Model"]

print(f"\n💡 EXPERIMENT SUMMARY & TRADE-OFFS:")
print(f" • Highest Performance (F1 / ROC-AUC) : {best_f1_model}")
print(f" • Fastest Inference Latency          : {fastest_infer_model}")
print(f" • Smallest Model Footprint           : {smallest_model}")
print(f" • Fastest Training Time              : {fastest_train_model}")
