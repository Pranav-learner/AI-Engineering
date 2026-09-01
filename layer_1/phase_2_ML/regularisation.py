import numpy as np


class RidgeRegressionScratch:

    def __init__(self, learning_rate=0.01,
                 epochs=1000,
                 lambda_=0.1):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lambda_ = lambda_

        self.w = 0.0
        self.b = 0.0

    def predict(self, X):
        return self.w * X + self.b

    def fit(self, X, y):

        n = len(X)

        for _ in range(self.epochs):

            y_pred = self.predict(X)

            error = y - y_pred

            # MSE gradients
            dw = (-2 / n) * np.sum(X * error)
            db = (-2 / n) * np.sum(error)

            # Ridge penalty
            dw += 2 * self.lambda_ * self.w

            # Update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

'''Notice something important:

dw += 2 * self.lambda_ * self.w

That's the regularization term.'''

'''Why Don't We Usually Penalize the Bias?

Notice:

dw += 2 * lambda * w

but we don't do:

db += 2 * lambda * b

Why?

Because the bias isn't generally treated as a feature weight controlling model complexity in the same way.

This is a common implementation detail'''

'''Lasso From Scratch

Conceptually:

dw += lambda_ * sign(w)

because the derivative/subgradient of:

∣w∣

is approximately:

sign(w)

away from zero.

Implementation:

def lasso_gradient(w, lambda_):
    return lambda_ * np.sign(w)

Then:

dw += lambda_ * np.sign(self.w)

The behavior near zero is what allows Lasso to drive coefficients toward zero.'''


from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=200,
    n_features=20,
    n_informative=5,
    noise=10,
    random_state=42
)

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1)
lasso = Lasso(alpha=1)

ridge.fit(X, y)
lasso.fit(X, y)

print("Ridge:")
print(ridge.coef_)

print("\nLasso:")
print(lasso.coef_)