import numpy as np


def prospect_value(x, alpha=0.88, beta=0.88, lambda_=2.25):
    """
    Kahneman & Tversky (1992) value function.
    x      : gain or loss as a fraction of reference price (e.g. 0.05 = +5%)
    alpha  : curvature for gains  (diminishing sensitivity)
    beta   : curvature for losses (diminishing sensitivity)
    lambda_: loss aversion coefficient — empirically ~2.25
    """
    if x >= 0:
        return x ** alpha
    else:
        return -lambda_ * ((-x) ** beta)


def probability_weight(p, gamma=0.65):
    """
    Tversky & Kahneman probability weighting.
    Humans overweight small probabilities (lottery effect)
    and underweight large ones.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return p ** gamma / (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)


def disposition_effect_pressure(pnl_pct, shares):
    """
    Shefrin & Statman (1985): investors sell winners too early,
    hold losers too long.
    Returns a sell-pressure multiplier [0, 1].
    """
    if pnl_pct > 0:
        return min(1.0, pnl_pct * 8)   # sell pressure grows with gain
    else:
        return max(0.0, 1 - abs(pnl_pct) * 5)  # hold pressure grows with loss
