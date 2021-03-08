import math
import random


def uniform_pdf(x: float) -> float:
    """
    一様分布の確率密度関数。
    """
    return 1 if 0 <= x < 1 else 0


def uniform_cdf(x: float) -> float:
    """
    一様分布の確率変数が x 以下となる確率を返す。
    """
    if x < 0:
        return 0  # 一様分布は0を下回らない
    elif x < 1:
        return x  # 例えば、P(X <= 0.4) = 0.4 となる
    else:
        return 1  # 一様分布は最大で1


_SQRT_TWO_PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    平均が mu, 分散が sigma であるような正規分布の確率密度関数。
    """
    return (math.exp(-(x - mu) ** 2 / 2 / sigma ** 2)) / (_SQRT_TWO_PI * sigma)


def plot_various_normal_pdf() -> None:
    """
    (平均, 分散) が異なる複数の正規分布の確率密度関数のグラフをプロットして表示する。
    グラフの表示には tkinter が必要 (debian なら `sudo apt install python3-tk` で成功した)。
    """
    import matplotlib.pyplot as plt
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_pdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
    plt.plot(xs, [normal_pdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
    plt.plot(xs, [normal_pdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
    plt.plot(xs, [normal_pdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
    plt.legend()
    plt.title("Various Normal pdfs")
    plt.show()


def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    平均が mu, 分散が sigma であるような正規分布の累積分布関数。
    """
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2


def plot_various_normal_cdf() -> None:
    """
    (平均, 分散) が異なる複数の正規分布の累積分布関数のグラフをプロットして表示する。
    グラフの表示には tkinter が必要 (debian なら `sudo apt install python3-tk` で成功した)。
    """
    import matplotlib.pyplot as plt
    xs = [x / 10.0 for x in range(-50, 50)]
    plt.plot(xs, [normal_cdf(x, sigma=1) for x in xs], '-', label='mu=0, sigma=1')
    plt.plot(xs, [normal_cdf(x, sigma=2) for x in xs], '--', label='mu=0, sigma=2')
    plt.plot(xs, [normal_cdf(x, sigma=0.5) for x in xs], ':', label='mu=0, sigma=0.5')
    plt.plot(xs, [normal_cdf(x, mu=-1) for x in xs], '-.', label='mu=-1, sigma=1')
    plt.legend(loc=4)  # 凡例を右下に表示
    plt.title("Various Normal cdfs")
    plt.show()


def inv_normal_cdf(p: float,
                   mu: float = 0,
                   sigma: float = 1,
                   tolerance: float = 1e-6) -> float:
    """
    normal_cdf(x, mu, sigma) >= p を満たす最小の x を求める。
    すなわちこの関数は normal_cdf() の逆関数といえる。
    累積分布関数は単調増加なので伝家の宝刀 †二分探索† を用いて解を求める。
    """
    # 標準正規分布でない場合、標準正規分布からの差分を求める
    if mu != 0 or sigma != 1:
        return sigma * inv_normal_cdf(p, tolerance=tolerance) + mu

    low_z = -10.0
    hi_z = 10.0
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2
        mid_p = normal_cdf(mid_z)
        if mid_p < p:
            low_z = mid_z
        else:
            hi_z = mid_z

    return hi_z


def bernoulli_trial(p: float) -> int:
    """
    確率 p で 1, 確率 1-p で 0 を返す。
    ベルヌーイ試行。
    """
    return 1 if random.random() < p else 0


def binomial(n: int, p: float) -> int:
    """
    ベルヌーイ試行 bernoulli(p) を n 回行った時の合計を返す。
    二項分布。
    """
    return sum(bernoulli_trial(p) for _ in range(n))


def plot_binomial_histogram(p: float, n: int, num_points: int) -> None:
    """
    binomial(n, p) を num_points 回収集し、それらのヒストグラムをプロットして表示する。
    「標本数 (num_points) が十分大きければ、中心極限定理により二項分布が正規分布に近似する」ことを確かめるための関数。
    """
    from collections import Counter
    from matplotlib import pyplot as plt

    data = [binomial(n, p) for _ in range(num_points)]

    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            width=0.8,
            color='0.75')

    mu = p * n
    sigma = math.sqrt(n * p * (1 - p))

    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma) for i in xs]
    plt.plot(xs, ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
    plt.show()
