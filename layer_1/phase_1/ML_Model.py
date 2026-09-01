from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load data
X,y = load_iris(return_X_y=True)

# Split 
X_train, X_test, y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=5  # cahnging the random state changes the accuracy.
    # Because your training and test samples changed.This is your first practical encounter with sampling variance.
)

# Create model
model = LogisticRegression(
    max_iter=200,
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:" , accuracy) 

## OVERFITTING

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=None,  # as depth it can go that will lead to 100% pure leaf and leading to overfitting
    random_state=42
)

model.fit(X_train,y_train)

train_accuracy = model.score(
    X_train,
    y_train
)

test_accuracy = model.score(
    X_test,
    y_test
)

print("Train Accuracy: ",train_accuracy)
print("Test Accuracy: ",test_accuracy)

#Now restrict the tree:
#model = DecisionTreeClassifier(
#    max_depth=2,
#    random_state=42
#)

#Compare.

#Then:

#max_depth=3

#You're experimentally observing:

#MODEL COMPLEXITY
       ↓
#training performance
       ↓
#generalization
       ↓
#overfitting
