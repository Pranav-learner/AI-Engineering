import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.DataFrame({
    "amount": [
        1000, 2000, 5000, 10000,
        20000, 30000, 50000, 100000,
        1500, 70000
    ],

    "foreign_transaction": [
        0, 0, 0, 1,
        1, 1, 0, 1,
        0, 1
    ],

    "unusual_location": [
        0, 0, 1, 1,
        1, 0, 0, 1,
        0, 1
    ],

    "fraud": [
        0, 0, 0, 1,
        1, 0, 0, 1,
        0, 1
    ]
})

X = data[
    [
        "amount",
        "foreign_transaction",
        "unusual_location"
    ]
]

y = data["fraud"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state = 42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions:")
print(predictions)

print("\nProbabilities (Fraud vs Not Fraud):")
print(probabilities)

print("\nActual:")
print(y_test.values)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))