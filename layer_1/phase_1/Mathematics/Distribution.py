import numpy as np

np.random.seed(42)

# Bernoulli
bernoulli = np.random.binomial(
    n=1,
    p=0.7,
    size=20
)

print("Bernoulli:")
print(bernoulli)

# Binomial
binomial = np.random.binomial(
    n=100,
    p=0.1,
    size=20
)

print("\nBinomial:")
print(binomial)

# Gaussian
gaussian = np.random.normal(
    loc=100,
    scale=10,
    size=10000
)

print("\nGaussian mean:", gaussian.mean())
print("Gaussian std:", gaussian.std())

# Poisson
poisson = np.random.poisson(
    lam=10,
    size=10000
)

print("\nPoisson mean:", poisson.mean())
print("Poisson variance:", poisson.var())

# Exponential
exponential = np.random.exponential(
    scale=0.1,
    size=10000
)

print("\nExponential mean:", exponential.mean())


#Experminet 1

np.random.seed(42)

p = 0.7

for n in [10, 100, 1000, 10000, 100000]:
    samples = np.random.binomial(
        1,
        p,
        size=n
    )

    print(
        n,
        samples.mean()
    )

# Experiment 2
data = np.random.normal(100, 10, 10000)
data_with_outliers = np.append(
    data,
    [1000, 2000, 5000]
)
data = np.concatenate([
    np.random.normal(50, 5, 5000),
    np.random.normal(150, 5, 5000)
])


# Eperiment 3 

requests_per_second = 100

traffic = np.random.poisson(
    requests_per_second,
    size=60
)

print("Requests each second:")
print(traffic)

print("Average:", traffic.mean())
print("Peak:", traffic.max())

over_capacity = np.sum(traffic > 120)

print(
    "Seconds over capacity:",
    over_capacity
)

#                           PROBABILITY
#                              │
#             ┌────────────────┴────────────────┐
#             ↓                                 ↓
#       RANDOM VARIABLE                    EVENTS
#             │                                 │
#      ┌──────┴──────┐                    Conditional
#      ↓             ↓                    Probability
#   Discrete      Continuous                    │
#      │             │                         ↓
#      ↓             ↓                       Bayes
#   Bernoulli      Gaussian                     │
#   Binomial       Exponential                  ↓
#   Categorical    Uniform                 ML inference
#   Multinomial
#       │
#       ↓
#    DISTRIBUTIONS
#       │
#       ├── Expectation
#       ├── Variance
#       ├── Sampling
#       └── Uncertainty
#               │
#               ↓
#         MACHINE LEARNING
#               │
#        ┌──────┼──────┐
#        ↓      ↓      ↓
# Classification  Regression  LLMs
#        │                    │
#        ↓                    ↓
# P(class|x)          P(token|context)
#                            │
#                            ↓
#                         Softmax
#                            │
#                            ↓
#                     Token Distribution