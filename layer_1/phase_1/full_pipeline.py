from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
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
        "selection",
        SelectKBest(
            score_func=f_classif,
            k=15
        )
    ),
    (
        "model",
        LogisticRegression(max_iter=5000)
    )
])

pipeline.fit(X_train, y_train)

print(
    "Test accuracy:",
    pipeline.score(X_test, y_test)
)

'''You now have:

Missing Data
     ↓
Scaling
     ↓
Feature Selection
     ↓
Model

in a reproducible pipeline.

🔥 Break the Pipeline

Now deliberately create bad versions.

Experiment A

Use:

k = 5
Experiment B
k = 10
Experiment C
k = 20
Experiment D
k = all

Compare.

Then compare:

StandardScaler

against:

MinMaxScaler

Then compare:

median imputation

against:

mean imputation

Record:

Technique
↓
Validation score
↓
Test score
↓
Training time
↓
Model complexity

You're now doing actual ML experimentation rather than simply learning definitions.'''