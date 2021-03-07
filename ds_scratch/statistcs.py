import math
from typing import Counter, List
from ds_scratch.linear_algebra import sum_of_squares


def mean(xs: List[float]) -> float:
    """
    xs の平均値を求める。

    >>> mean([1, 2, 3])
    2.0
    """
    return sum(xs) / len(xs)


def median(xs: List[float]) -> float:
    """
    xs の中央値を求める。

    >>> median([1, 10, 2, 9, 5]) == 5
    True
    >>> median([1, 9, 2, 10]) == (2 + 9) / 2
    True
    """
    n = len(xs)
    if n & 1:
        return sorted(xs)[n // 2]
    else:
        sorted_xs = sorted(xs)
        hi_midpoint = n // 2
        return (sorted_xs[hi_midpoint] + sorted_xs[hi_midpoint - 1]) / 2


def quantile(xs: List[float], p: float) -> float:
    """
    xs 中の p 百分位数を返す。

    >>> quantile([0, 11, 22, 33, 44, 55, 66, 77, 88, 99], 0.10)
    11
    >>> quantile([0, 11, 22, 33, 44, 55, 66, 77, 88, 99], 0.25)
    22
    >>> quantile([0, 11, 22, 33, 44, 55, 66, 77, 88, 99], 0.79)
    77
    >>> quantile([0, 11, 22, 33, 44, 55, 66, 77, 88, 99], 0.90)
    99
    """
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


def mode(xs: List[float]) -> List[float]:
    """
    xs の最頻値を求める。最頻値は1つとは限らないのでリストを返す。

    >>> set(mode([4, 1, 1, 1, 5, 4, 4, 2])) == {1, 4}
    True
    """
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]


def minmax_range(xs: List[float]) -> float:
    """
    xs の値の範囲 (xs の最大値と最小値の差の絶対値) を返す。

    >>> minmax_range([9, 1, 4, 1])
    8
    """
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> List[float]:
    """
    各要素 x_i を変換して、x_i と xs の平均との差とする (結果の平均が 0 となるように)。

    >>> de_mean([3, 1, 2]) == [1, -1, 0]
    True
    """
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    """
    xs の不偏分散を返す。
    平均との差の二乗の 'おおよそ' の平均 (n ではなく n - 1 で割る)。
    """
    n = len(xs)
    assert n >= 2, "分散は少なくとも2つ以上の要素を必要とする"

    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    """
    xs の標準偏差を求める。
    """
    return math.sqrt(variance(xs))


def interquartile_range(xs: List[float]) -> float:
    """
    xs の四分位範囲 (75パーセンタイルと25パーセンタイルの差) を求める。
    """
    return quantile(xs, 0.75) - quantile(xs, 0.25)
