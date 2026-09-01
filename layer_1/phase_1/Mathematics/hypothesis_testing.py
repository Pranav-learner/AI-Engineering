import numpy as np

np.random.seed(42)

old_scores = np.random.normal(
    loc=0.80,
    scale=0.08,
    size=1000
)

new_scores = np.random.normal(
    loc=0.83,
    scale=0.08,
    size=1000
)

print("Old mean:", old_scores.mean())
print("New mean:", new_scores.mean())

print("Old std:", old_scores.std())
print("New std:", new_scores.std())

difference = new_scores.mean() - old_scores.mean()

print("Improvement:", difference)

def confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    n = len(data)

    se = std / np.sqrt(n)

    lower = mean - 1.96 * se
    upper = mean + 1.96 * se

    return lower, upper


print(
    "Old CI:",
    confidence_interval(old_scores)
)

print(
    "New CI:",
    confidence_interval(new_scores)
)
# Now ask:

How much overlap exists?

# But be careful:

# CI overlap alone is not a universal test for whether two means differ significantly.

# For proper comparisons, use an appropriate statistical test.

# -----------

from scipy.stats import ttest_ind

result = ttest_ind(
    old_scores,
    new_scores,
    equal_var=False
)

print("t-statistic:", result.statistic)
print("p-value:", result.pvalue)