from typing import Tuple
import math
from ds_scratch.probability import normal_cdf
from ds_scratch.probability import inv_normal_cdf


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """
    Binomial(n, p) n相当するμとσを求める
    """
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma


# 変数がしきい値を下回る確率。 normal_cdf() と同じ。
normal_probability_below = normal_cdf


def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
    """
    N(mu, sigma) が lo よりも大きくなる確率。
    """
    return 1 - normal_cdf(lo, mu, sigma)


def normal_probability_between(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """
    N(mu, sigma) が lo と hi の間にある確率。
    """
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


def normal_probability_outside(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
    """
    N(mu, sigma) が lo と hi の間にない確率。
    """
    return 1 - normal_probability_between(lo, hi, mu, sigma)


def normal_upper_bound(prob: float, mu: float = 0, sigma: float = 1) -> float:
    """
    確率 P(z) >= prob となる z を求める。
    """
    return inv_normal_cdf(prob, mu, sigma)


def normal_lower_bound(prob: float, mu: float = 0, sigma: float = 1) -> float:
    """
    確率 P(z) >= 1 - prob となる z を求める。
    """
    return inv_normal_cdf(1 - prob, mu, sigma)


def normal_two_sided_bounds(prob: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
    """
    指定された確率を包含する (平均を中心に) 対象な境界を返す。
    """
    tail_prob = (1 - prob) / 2

    # 上側の境界は tail_prob 分上に
    ub = normal_lower_bound(tail_prob, mu, sigma)

    # 下側の境界は tail_prob 分下に
    lb = normal_upper_bound(tail_prob, mu, sigma)

    return lb, ub


# コイン投げの回数を n = 1000 とする。
# コインを n 回投げて表が出た回数を確率変数 X とする。
# 表が出る確率を p とする。
# コインに歪みがないことを示す帰無仮説H_0は p = 0.5 であり、対立仮説H_1は p != 0.5 (=歪みがある) である。

def test1() -> None:
    # コインに歪みは無いという仮説が真であるなら、X は平均μ=500, 分散σ=15.8の正規分布で近似できる。
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

    # 実際には真であるにもかかわらずH_0を棄却してしまうという
    # 「第一種の過誤」をどの程度受け入れるか(=有意性) を 5% とする。
    # p が実際に 0.5 に等しい場合を考えると、 X が以下の区間外となる確率は 5% である。
    lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
    assert round(lower_bound) == 469
    assert round(upper_bound) == 531

    # 実際にはH_0が偽であるにも関わらずH_0を棄却しないことを第二の過誤という。
    # 第二の過誤が起きない確率 (=検定力) は次のように計算できる。

    # pが0.5であると想定のもとで、95%の境界を確認する
    lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

    # p = 0.55であった場合の、μとσを計算する
    mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

    # 第二種過誤は、帰無仮説H_0が誤っているにも関わらず棄却せず、
    # Xが当初想定の領域(H_0の領域)に入ってしまっている場合に生じる。
    type2_prob = normal_probability_between(lo, hi, mu_1, sigma_1)
    power = 1 - type2_prob  # 0.887
    assert math.isclose(power, 0.887, rel_tol=1e-3)

    # 帰無仮説H_0が、 p == 0.5 ではなく p <= 0.5 であると仮定する。
    # この場合は片側検定を使う(X>>500 => H_0を棄却, X <= 500 => H_0を棄却しない)。
    hi = normal_upper_bound(0.95, mu_0, sigma_0)
    type2_prob = normal_probability_below(hi, mu_1, sigma_1)
    power = 1 - type2_prob  # 0.936
    assert math.isclose(power, 0.936, rel_tol=1e-3)

    # 上記の片側検定では、X が469より小さい値ならH_0を棄却しない。
    # Xが526と531の間の値は、H_1が真であるなら多少起こり得るが、
    # その場合には正しくH_0を棄却できるので、両側検定よりも強い検定であると言える。


def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
    """
    両側検定でp値を求める。
    値がN(mu, sigma)に従うとして、(上側でも下側でも)極端なxが現れる可能性はどの程度か
    """
    if x >= mu:
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        return 2 * normal_probability_below(x, mu, sigma)


# 片側検定でp値を求める。
upper_p_value = normal_probability_above
lower_p_value = normal_probability_below


def test2() -> None:
    """
    test1() とは別の検定方法。
    H_0が真であると仮定して、実際に観測された値と少なくとも同等に極端な値が生じる確率を求める。
    """
    # 表が530回出た場合、次の様に計算できる。
    mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)
    p = two_sided_p_value(530 - 0.5, mu_0, sigma_0)  # 530から0.5を引いているのは連続性補正に基づく(←よく分かってない)
    assert math.isclose(p, 0.062, rel_tol=1e-2)

    # ↑この場合、p値は有意性の5%よりも大きいため、帰無仮説H_0は棄却されない。

    # 一方、表が530回出た場合、次の様に計算でき、5%よりも小さい値のため帰無仮説を棄却する。
    p = two_sided_p_value(532 - 0.5, mu_0, sigma_0)
    assert math.isclose(p, 0.0463, rel_tol=1e-2)

    # 片側検定の場合、表が525回の場合は帰無仮説を棄却しない。
    upper_p_value(525 - 0.5, mu_0, sigma_0)  # 0.061

    # しかし、527回ならば棄却することになる。
    upper_p_value(527 - 0.5, mu_0, sigma_0)

    # Note:
    # normal_probability_above() などを使ってp値を計算する前に、
    # データがおよそ正規分布に従っていることを確かめる必要がある。
    # 正規分布であるかどうかを調べる統計手法は数多く存在するが、データをグラフ化するのが
    # 簡単で優れている (らしい)。


def test3() -> None:
    """
    test1(), test2() とは別の3番目の手法。

    表を1，裏を0とするベルヌーイ変数の平均をみることで歪みのあるコインに対する確率を推定できる。
    1000回の試行で525回の表が出たとすると、pの推定値は0.525である。
    この推定値がどの程度信頼できるか。
    """
    p_hat = 525 / 1000
    mu = p_hat
    sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)  # 0.0158

    # 正規分布の近似を使うと、pの正しい値が次の区間に入るのは「95%の確率で信頼できる」という結論になる。
    # 0.5 はこの信頼区間内にあるため、コインに歪みがあるとは結論付けられない。
    normal_two_sided_bounds(0.95, mu, sigma)  # [0.4940, 0.5560]

    # 一方、表が540回出た場合の信頼区間は次の通りで、コインに歪みがないという仮説は成立しない。
    p_hat = 540 / 1000
    mu = p_hat
    sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)  # 0.0158
    normal_two_sided_bounds(0.95, mu, sigma)  # [0.5091, 0.5709]


def B(alpha: float, beta: float) -> float:
    """
    ベータ分布。
    確率の総和が1となるように定数で正規化する。
    """
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)


def beta_pdf(x: float, alpha: float, beta: float) -> float:
    return x ** (alpha - 1) * (1 - x) ** (beta- 1) / B(alpha, beta)
