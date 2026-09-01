import numpy as np

latencies = np.array([
    120, 130, 125, 140, 115,
    121, 128, 135, 118, 150
])

mean = np.mean(latencies)
variance = np.var(latencies)
std = np.std(latencies)

print("Mean:", mean)
print("Variance:", variance)
print("Std Dev:", std)

print("P50:", np.percentile(latencies, 50))
print("P95:", np.percentile(latencies, 95))
print("P99:", np.percentile(latencies, 99))

model_a = np.array([
    100, 102, 98, 101, 99,
    100, 103, 97, 101, 99
])

model_b = np.array([
    70, 150, 90, 180, 100,
    60, 200, 80, 120, 50
])