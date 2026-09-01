from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test,y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y    
)

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "feature_selection",
        SelectKBest(
            score_func = f_classif,
            k=10
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter=5000
        )
    )
])


param_grid = {
    "feature_selection__k": [
        5,
        10,
        15,
        20
    ],

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ]
}

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

search.fit(
    X_train,
    y_train
)

print(
    search.best_params_
)

print(
    search.best_score_
)

y_pred = search.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1:",
    f1_score(y_test, y_pred)
)

print(
    confusion_matrix(y_test, y_pred)
)