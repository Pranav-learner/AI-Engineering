import numpy as np


class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):

            probabilities = self.predict_proba(X)

            error = probabilities - y

            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegressionScratch(
    learning_rate=0.01,
    epochs=2000
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions[:20])

probabilities = model.predict_proba(X_test)

print(probabilities[:20])

'''Experiment — Change the Threshold

This is where classification becomes much more interesting.'''

thresholds = [0.2, 0.3, 0.5, 0.7, 0.9]

for threshold in thresholds:

    predictions = model.predict(
        X_test,
        threshold=threshold
    )

    print(threshold, predictions.sum())