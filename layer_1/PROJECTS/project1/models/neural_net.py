"""
neural_net.py - PyTorch Tabular Multi-Layer Perceptron (MLP) for expense forecasting & risk classification.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import RANDOM_SEED
from ..feature_engineering import FEATURE_COLUMNS


# =====================================================================
# 1. PYTORCH NEURAL NETWORK ARCHITECTURE
# =====================================================================

class TabularMLP(nn.Module):
    """
    A 2-Hidden-Layer Multi-Layer Perceptron (MLP) with Batch Normalization,
    ReLU activations, and Dropout for tabular financial data.
    """
    def __init__(self, in_features: int, is_classifier: bool = False, num_classes: int = 3):
        super().__init__()
        self.is_classifier = is_classifier

        # Hidden Layer 1
        self.layer1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.15)

        # Hidden Layer 2
        self.layer2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(p=0.10)

        # Output Head
        if is_classifier:
            self.head = nn.Linear(32, num_classes)  # 3 output logits for [Low, Medium, High]
        else:
            self.head = nn.Linear(32, 1)            # 1 scalar output for Next Month Expense

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Pass through Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Output Logits / Prediction
        out = self.head(x)
        return out


# =====================================================================
# 2. SCIKIT-LEARN COMPATIBLE WRAPPER
# =====================================================================

class PyTorchTabularModel(BaseEstimator):
    """
    Wraps the PyTorch TabularMLP inside the standard Scikit-Learn Estimator interface.
    Handles data normalization (StandardScaler), tensor conversion, and the training loop.
    """
    def __init__(
        self,
        is_classifier: bool = False,
        lr: float = 0.005,
        epochs: int = 150,
        batch_size: int = 16,
        weight_decay: float = 1e-4,
    ):
        self.is_classifier = is_classifier
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.scaler = StandardScaler()
        self.model: Optional[TabularMLP] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        torch.manual_seed(RANDOM_SEED)

        # 1. Scale Input Tabular Features
        if isinstance(X, pd.DataFrame):
            X_arr = X[FEATURE_COLUMNS].values if set(FEATURE_COLUMNS).issubset(X.columns) else X.values
        else:
            X_arr = X

        X_scaled = self.scaler.fit_transform(X_arr)
        in_features = X_scaled.shape[1]

        # 2. Initialize PyTorch Model
        self.model = TabularMLP(in_features=in_features, is_classifier=self.is_classifier)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # 3. Choose Loss Function
        if self.is_classifier:
            criterion = nn.CrossEntropyLoss()
            y_tensor = torch.tensor(y, dtype=torch.long)
        else:
            criterion = nn.MSELoss()
            # Normalize target for stable gradient descent
            self.y_mean = float(np.mean(y))
            self.y_std = float(np.std(y)) if np.std(y) > 0 else 1.0
            y_norm = (y - self.y_mean) / self.y_std
            y_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 4. Training Loop (Mini-Batch Gradient Descent)
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fitted before predicting!")

        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X_arr = X[FEATURE_COLUMNS].values if set(FEATURE_COLUMNS).issubset(X.columns) else X.values
        else:
            X_arr = X

        X_scaled = self.scaler.transform(X_arr)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.is_classifier:
                preds = torch.argmax(outputs, dim=1).numpy()
            else:
                preds_norm = outputs.squeeze(1).numpy()
                preds = (preds_norm * self.y_std) + self.y_mean  # De-normalize back to rupees

        return preds


# Helper factories
def get_pytorch_regressor() -> PyTorchTabularModel:
    return PyTorchTabularModel(is_classifier=False, epochs=150, lr=0.005)

def get_pytorch_classifier() -> PyTorchTabularModel:
    return PyTorchTabularModel(is_classifier=True, epochs=150, lr=0.005)


if __name__ == "__main__":
    from ..data_generator import generate_cohort_dataset
    from ..feature_engineering import build_cohort_feature_matrix, temporal_train_val_test_split
    from .classical_models import evaluate_classification, evaluate_regression

    print("Generating dataset to train PyTorch MLP...")
    tx_df = generate_cohort_dataset(num_users=16, num_days=730)
    feat_df = build_cohort_feature_matrix(tx_df)
    train_df, val_df, test_df = temporal_train_val_test_split(feat_df)

    X_train, y_reg_train, y_cls_train = train_df[FEATURE_COLUMNS], train_df["target_next_month_expense"].values, train_df["target_risk_tier"].values
    X_test, y_reg_test, y_cls_test = test_df[FEATURE_COLUMNS], test_df["target_next_month_expense"].values, test_df["target_risk_tier"].values

    print("\n--- 1. Training PyTorch MLP Regressor ---")
    mlp_reg = get_pytorch_regressor()
    mlp_reg.fit(X_train, y_reg_train)
    reg_preds = mlp_reg.predict(X_test)
    print("PyTorch MLP Regression Metrics:", evaluate_regression(y_reg_test, reg_preds))

    print("\n--- 2. Training PyTorch MLP Classifier ---")
    mlp_cls = get_pytorch_classifier()
    mlp_cls.fit(X_train, y_cls_train)
    cls_preds = mlp_cls.predict(X_test)
    cls_metrics = evaluate_classification(y_cls_test, cls_preds)
    print(f"PyTorch MLP Classification Metrics: F1={cls_metrics['F1_Macro']}, Acc={cls_metrics['Accuracy']}")
    print(f"Confusion Matrix:\n{cls_metrics['Confusion_Matrix']}")
