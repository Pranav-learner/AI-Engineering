import numpy as np

X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# prediction function: y= 2x+1
# though we get w = 2 and b = 1
# 
#but model donsnt know it so randomly starting with w = 0.0 and b= 0.0
w=0.0
b=0.0

y_pred = w * X + b

print(y_pred)

errors = y - y_pred

mse = np.mean(errors ** 2)

print(mse)

w = 1
b = 0

y_pred = w * X + b
mse = np.mean((y - y_pred) ** 2)

print(y_pred)
print(mse)


## MSE and MAE

def mse(y_true, y_pred):
    errors = y_true - y_pred
    return np.mean(errors ** 2)


def mae(y_true, y_pred):
    errors = y_true - y_pred
    return np.mean(np.abs(errors))

y_true = np.array([10, 20, 30, 40])
y_pred = np.array([11, 18, 35, 60])

print("MSE:", mse(y_true, y_pred))
print("MAE:", mae(y_true, y_pred))

#Linear Regression from Scratch

class LinearRegressionScratch:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = 0.0
        self.b = 0.0

        self.loss_history = []

    def predict(self, X):
        return self.w * X + self.b

    def mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def fit(self, X, y):

        n = len(X)

        for epoch in range(self.epochs):

            # 1. Prediction
            y_pred = self.predict(X)

            # 2. Error
            error = y - y_pred

            # 3. Loss
            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            # 4. Gradients
            dw = (-2 / n) * np.sum(X * error)
            db = (-2 / n) * np.sum(error)

            # 5. Parameter update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

        return self

X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

model = LinearRegressionScratch(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

print("w:", model.w)
print("b:", model.b)

predictions = model.predict(X)

print(predictions)

# experimebt with learning rate

import matplotlib.pyplot as plt

plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Loss")
plt.show()