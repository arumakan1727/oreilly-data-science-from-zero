from ds_scratch.linear_algebra import Vector, add, distance, dot, scalar_multiply, vector_mean
from typing import Callable, Iterator, List, Tuple, TypeVar
import random


def sum_of_squares(v: Vector) -> float:
    """
    vの各要素の二乗を合計する
    """
    return dot(v, v)


def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    """
    1変数の関数 f の x における微分係数を求める。
    """
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """
    関数fと変数ベクトルvに対するi番目の差分商を返す
    """
    w = [v_j + (h if j == i else 0)
         for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.001) -> List[float]:
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    v + (gradient * step_size) を計算する。
    """
    assert(len(v) == len(gradient))
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def sum_of_squares_gradient(v: Vector) -> Vector:
    """
    sum_of_squares() の勾配。 x**2 の微分は 2*x。
    """
    return [2 * v_i for v_i in v]


def do_gradient_descent_for_sum_of_squares() -> None:
    # 開始点を無作為に選択する
    v = [random.uniform(-10, 10) for _ in range(3)]

    for epoch in range(1000):
        grad = sum_of_squares_gradient(v)
        v = gradient_step(v, grad, -0.01)
        print(epoch, v)

    assert distance(v, [0, 0, 0]) < 0.01


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    """
    theta は (傾き, 切片) の2要素タプル。
    (theta がTuple型ではないのは、他の関数の引数の型がVectorである必要があるため])

    theta で表される一次関数に x を代入して、得られた y と引数の y との誤差を求めて、良さげな勾配を返す。
    """
    slope, intercept = theta
    predicted = slope * x + intercept  # モデルの予測
    error = (predicted - y)            # 誤差は (予測 - 実際の値)
    grad = [2 * error * x, 2 * error]  # 勾配を使って2乗誤差を最小にする
    return grad


def do_gradient_descent_in_model_adaptation() -> None:
    # xの範囲は [-50, +50)。yは常に20*x + 5。
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

    # ランダムな傾きと切片でスタートする
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    learning_rate = 0.001

    for epoch in range(5000):
        # 勾配の平均を求める
        grad = vector_mean([linear_gradient(x, y, theta) for x, y in inputs])

        # 勾配の方向に沿って進む
        theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1, "傾きはおおよそ 20"
    assert 4.9 < intercept < 5.1, "切片はおおよそ 5"


T = TypeVar('T')


def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """
    データセットから batch_size の大きさのミニバッチを生成する
    """
    # インデックスは 0 から開始し、1*batch_size, 2*batch_size, ... とする
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle:
        random.shuffle(batch_starts)

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


def do_gradient_descent_in_model_adaptation_with_minibatch() -> None:
    """
    ミニバッチ法による勾配下降法を行う。
    前述のアプローチ (do_gradient_descent_in_model_adaptation()) は、
    モデルのパラメータの更新のためにデータセット全体の勾配を評価する必要があった。
    データセットが大量で、複雑な勾配計算となる場合を考えると、より頻繁に勾配ステップを実行する必要がある。
    そのための手法の一つがミニバッチ法である。
    """
    # xの範囲は [-50, +50)。yは常に20*x + 5。
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

    # ランダムな傾きと切片でスタートする
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    learning_rate = 0.001

    for epoch in range(1000):
        for batch in minibatches(inputs, batch_size=20):
            grad = vector_mean([linear_gradient(x, y, theta) for x, y in batch])
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1, "傾きはおおよそ 20"
    assert 4.9 < intercept < 5.1, "切片はおおよそ 5"


def do_gradient_descent_in_model_adaptation_with_probability() -> None:
    """
    ミニバッチ法とは別のアプローチ。1データポイントずつ勾配を計算する。
    """
    # xの範囲は [-50, +50)。yは常に20*x + 5。
    inputs = [(x, 20 * x + 5) for x in range(-50, 50)]

    # ランダムな傾きと切片でスタートする
    theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

    learning_rate = 0.001

    for epoch in range(100):
        for x, y in inputs:
            grad = linear_gradient(x, y, theta)
            theta = gradient_step(theta, grad, -learning_rate)
        print(epoch, theta)

    slope, intercept = theta
    assert 19.9 < slope < 20.1, "傾きはおおよそ 20"
    assert 4.9 < intercept < 5.1, "切片はおおよそ 5"
