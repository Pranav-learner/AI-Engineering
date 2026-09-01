from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ("selection", SelectKBest(
        score_func=f_classif,
        k=10
    )),
    ("model", LogisticRegression(
        max_iter=5000
    ))
])

pipeline.fit(X_train, y_train)

print(
    "Test accuracy:",
    pipeline.score(X_test, y_test)
)

# Experiment with:

'''k = 5
k = 10
k = 20
k = 30
all features

Then compare.

The important question isn't:"

"Which k gives the highest number?"

#It's: How does feature dimensionality 
# affect generalization, complexity, and performance?'''

## NORMALISATION

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''Notice something extremely important:

TRAIN:
fit_transform()

TEST:
transform()

Not:

scaler.fit_transform(X_test)

Why?

Because the test set must remain unseen.

The scaler itself learns:

min
max

from the training data.

If you calculate those using the test set, you've leaked information.'''

## STANDARDISATION

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

'''Again:

fit → TRAIN ONLY
transform → TRAIN + TEST'''