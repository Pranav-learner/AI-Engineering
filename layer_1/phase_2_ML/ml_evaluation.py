from sklearn.datasets import make_classification

#DATASET

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.9, 0.1],
    random_state=42
)

#TRAIN_TEST_SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#MODEL_TRAINING

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000
)


model.fit(X_train, y_train)

#MODEL_PREDICTION

y_pred = model.predict(X_test)

#EVALUATE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ROC-AUC

from sklearn.metrics import roc_auc_score

y_prob = model.predict_proba(X_test)[:, 1]

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_prob)
)