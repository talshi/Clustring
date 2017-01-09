from math import sqrt


def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(vector):
    return sqrt(dot_product(vector, vector))


def euclidean_similarity(v1, v2):
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)


def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)


def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude