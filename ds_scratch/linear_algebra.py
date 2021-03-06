import math
from typing import List

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """
    2つのベクトルの要素ごとの和。

    >>> add([1, 2, 3], [4, 5, 6])
    [5, 7, 9]
    """
    assert len(v) == len(w), "vectors must be same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """
    ベクトルの要素ごとの差。

    >>> subtract([5, 7, 9], [4, 5, 6])
    [1, 2, 3]
    """
    assert len(v) == len(w), "vectors must be same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    """
    可変個のベクトルの要素ごとの和。 vec1 + vec2 + ... + vec_n

    >>> vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])
    [16, 20]

    >>> vector_sum([[3, 3, 4]])
    [3, 3, 4]
    """
    assert vectors, "no vectors provided!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def scalar_multiply(k: float, v: Vector) -> Vector:
    """
    各要素に k を乗ずる。

    >>> scalar_multiply(2, [1, 2, 3])
    [2, 4, 6]
    """
    return [k * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """
    要素ごとの平均を求める。

    >>> vector_mean([[1, 2], [3, 4], [5, 6]])
    [3.0, 4.0]
    """
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> float:
    """
    ドット積 (内積) を求める。
    v_1 * w_1 + ... + v_n * w_n

    >>> dot([1, 2, 3], [4, 5, 6])
    32
    """
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    """
    各要素の2乗和を求める。
    v_1 * v_1 + ... + v_n * v_n

    >>> sum_of_squares([1, 2, 3])
    14
    """
    return dot(v, v)


def magnitude(v: Vector) -> float:
    """
    v のマグニチュード (大きさ、絶対値) を求める。

    >>> magnitude([3, 4])
    5.0
    """
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    """
    (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2 を求める。

    >>> squared_distance([1, 2, 3], [4, 5, 6])
    27
    """
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector) -> float:
    """
    v と w の距離を求める。

    >>> distance([5, 8], [2, 12])
    5.0
    """
    return magnitude(subtract(v, w))
