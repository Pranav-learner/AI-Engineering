import math


def magnitude(v):
    total = 0

    for x in v:
        total += x * x

    return math.sqrt(total)


def normalize(v):
    mag = magnitude(v)

    if mag == 0:
        raise ValueError("Cannot normalize zero vector")

    return [x / mag for x in v]


def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")

    result = 0

    for x, y in zip(a, b):
        result += x * y

    return result


def euclidean_distance(a, b):
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")

    total = 0

    for x, y in zip(a, b):
        difference = x - y
        total += difference * difference

    return math.sqrt(total)


def cosine_similarity(a, b):
    mag_a = magnitude(a)
    mag_b = magnitude(b)

    if mag_a == 0 or mag_b == 0:
        raise ValueError("Cosine similarity undefined for zero vector")

    return dot_product(a, b) / (mag_a * mag_b)


def matrix_multiply(A, B):

    if len(A[0]) != len(B):
        raise ValueError("Invalid matrix dimensions")

    rows = len(A)
    cols = len(B[0])

    result = [
        [0 for _ in range(cols)]
        for _ in range(rows)
    ]

    for i in range(rows):
        for j in range(cols):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


def transpose(A):

    return [
        [A[i][j] for i in range(len(A))]
        for j in range(len(A[0]))
    ]


A = [
    [1, 2],
    [3, 4]
]

B = [
    [5, 6],
    [7, 8]
]

print(matrix_multiply(A, B))

# Projection — VERY IMPORTANT

# Projection is one of the most useful concepts in this section.

# Suppose we have vector:

# A

# and we want to know:

# How much of A lies in the direction of B?

# We can project A onto B.

# The scalar projection is:

# proj
# B
# 	​

# (A)=
# ∣∣B∣∣
# 2
# A⋅B
# 	​

# B

# Let's use:

# A=[3,4]

# and:

# B=[1,0]

# Then:

# A⋅B=3

# and:

# ∣∣B∣∣
# 2
# =1

# Therefore:

# proj
# B
# 	​

# (A)=3[1,0]
# =[3,0]

# So:

# A = [3,4]


# horizontal component = [3,0]
# vertical component   = [0,4]

def dot_product(a, b):
    if len(a) != len(b):
        raise ValueError("Dimension mismatch")

    return sum(x * y for x, y in zip(a, b))


def magnitude_squared(v):
    return dot_product(v, v)


def vector_scale(v, scalar):
    return [scalar * x for x in v]


def projection(a, b):
    denominator = magnitude_squared(b)

    if denominator == 0:
        raise ValueError("Cannot project onto zero vector")

    coefficient = dot_product(a, b) / denominator

    return vector_scale(b, coefficient)


a = [3, 4]
b = [1, 0]

print(projection(a, b))