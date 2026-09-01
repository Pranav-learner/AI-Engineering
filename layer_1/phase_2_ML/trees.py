from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Generate classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=3,
    criterion="gini",
    random_state=42
)

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(predictions)

# Experiment -- Tree depth

from sklearn.tree import DecisionTreeClassifier

depths = [1, 2, 3, 5, 10, 20]

for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(
        depth,
        train_score,
        test_score
    )

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train,y_train)

predictins = model.predict(X_test)



# EXPERIMENT TREE VS FOREST

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

print("Tree:")
print(tree.score(X_test, y_test))

print("Forest:")
print(forest.score(X_test, y_test))