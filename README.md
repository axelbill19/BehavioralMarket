# Behavioral Market Simulator

An Agent-Based Model (ABM) of financial markets grounded in behavioral finance theory.
Instead of assuming rational agents, this model simulates how **real human psychology**
— loss aversion, herding, FOMO, and anchoring — shapes price dynamics and produces
the statistical properties observed in actual markets.

---

## Motivation

Classical finance assumes markets are populated by rational agents maximizing expected
utility. Real markets are not. The 2010 Flash Crash, the 2021 meme-stock mania, and
every major bubble in history are driven by behavioral biases, not rationality.

This project asks: **if we populate a market with psychologically realistic agents,
do realistic market properties emerge on their own?**

The answer: **yes**.

---

## Architecture

```
prospect_theory.py   Kahneman & Tversky value function + disposition effect
agents.py            Five agent types with distinct behavioral profiles
market.py            Mesa 3.x model — GARCH(1,1) + Merton jump-diffusion + herding
run.py               Full simulation with 7-panel visualization
compare.py           Scenario A vs B: humans-only vs humans + algorithms
sensitivity.py       Seven market ecosystems × 5 random seeds
validate.py          Empirical validation against 491 real S&P 500 stocks
```

---

## Agents

| Agent | Behavioral Basis | Speed | Psychology |
|---|---|---|---|
| `ValueInvestor` | Graham / Buffett | Slow (every 20 ticks) | Buys with margin of safety, ignores noise |
| `MomentumTrader` | FOMO + herding | Medium (every 10 ticks) | Chases rallies, amplifies drops |
| `PanicSeller` | Prospect Theory | Fast (every 7 ticks) | Loss-averse (λ=2.25), sells winners early |
| `AlgoTrader` | Trend-following / market-making | Every tick | No emotion — amplifies whatever direction exists |
| `NoiseTrader` | Uninformed retail | Medium (every 5 ticks) | Random with slight sentiment bias + media herding |

### Prospect Theory (Kahneman & Tversky, 1992)

The `PanicSeller` evaluates its position using the empirically validated value function:

```
V(x) =  x^α              if x ≥ 0   (gains — diminishing sensitivity)
V(x) = −λ · (−x)^β      if x < 0   (losses — loss aversion, λ = 2.25)
```

Loss aversion (λ = 2.25) means losing $100 feels 2.25× worse than gaining $100 feels
good. This causes panic selling at thresholds that a rational agent would ignore.

---

## Price Formation

```
P(t+1) = P(t) · exp(λ · net_flow · contagion · vol_regime)
                · exp(GARCH noise)
                · exp(Merton jump)
                · exp(−κ · log(P/P₀))      ← soft mean reversion
```

**Three mechanisms produce realistic statistical properties:**

1. **GARCH(1,1)** — volatility clustering: stressed periods beget more stress
2. **Merton jump-diffusion** — Poisson jumps (p=5% per tick, σ=12%) generate fat tails
3. **Herding signal** — agents observe last tick's order flow and amplify it

---

## Key Findings

### 1. Fat tails emerge from behavioral heterogeneity

Simulated log-returns exhibit **positive excess kurtosis** (avg. 3.0–8.0 across seeds
and scenarios), matching the empirical range of real S&P 500 stocks (median: 9.45).

The `Panic Mkt` scenario — dominated by loss-averse agents — best replicates the
kurtosis of real markets (8.01 vs. 9.45 real median). During the 2018–2023 period
(COVID + inflation + rate hikes), real markets were indeed panic-dominated.

### 2. Algorithms amplify, not create

Comparing the same market with and without algorithmic traders (same seed, same shocks):

| Metric | Humans only | Humans + Algos |
|---|---|---|
| Kurtosis | 6.94 | 1.47 |
| Trend persistence | Low | **High** |
| Crash speed at shock −10% | Gradual | **Instantaneous cascade** |

Algos do not create directional bias — they amplify whatever bias humans set.
This replicates the mechanism behind the 2010 Flash Crash.

### 3. Removing value investors creates bubbles

The `No Anchor` scenario (no value investors) produces **+249% average return** across
seeds — nobody sells when the price is expensive. Without a fundamental anchor,
momentum and noise traders push prices to unsustainable levels indefinitely.

### 4. The CLT problem in ABM

With 150 agents making moderate moves each tick, the Central Limit Theorem flattens
the return distribution toward Gaussian (negative kurtosis). The fix: **Merton
jump-diffusion** separates quiet ticks from rare large-jump ticks.
Fat tails require temporal contrast, not just agent heterogeneity.

**Rule of thumb:** `jump_sigma` must be ≥ 3× the std of agent-driven flow,
or agent moves dilute jump kurtosis toward zero.

---

## Sensitivity Analysis (7 Scenarios × 5 Seeds)

| Scenario | Description | Kurtosis | Max DD | Return | Interpretation |
|---|---|---|---|---|---|
| Baseline | Mixed market | +3.78 | 87% | −7% | Normal market regime |
| Panic Mkt | 70 loss-averse agents | +3.69 | 77% | +20% | Crisis / distress |
| No Anchor | Zero value investors | +2.67 | 74% | +249% | Speculative bubble |
| HFT Dom. | 75 trend algos | +0.79 | 82% | +7% | Near-Gaussian, trending |
| Random Walk | 150 noise traders only | +7.19 | 61% | +12% | EMH benchmark |
| Institutional | No retail | +4.19 | 84% | −56% | Informed, bearish |
| Herd Mkt | 70 momentum traders | +2.22 | 81% | +147% | Trend + high vol |

---

## Empirical Validation

Compared simulated statistics against **490 S&P 500 companies** (2018–2023, ~603K rows):

- **Kurtosis:** Panic Mkt scenario (8.01) best matches real median (9.45)
- **Skewness:** Real median −0.35; Baseline and Institutional scenarios match sign and order
- **Volatility gap:** Simulated ~80% vs real ~34% — reflects aggressive shock calibration;
  reducing shock magnitudes would close this gap

---

## How to Run

```bash
pip install mesa scipy matplotlib pandas numpy

# Full simulation (600 ticks, 150 agents, 4 shocks)
python run.py

# Compare humans-only vs humans + algorithms
python compare.py

# Sensitivity analysis (7 scenarios × 5 seeds)
python sensitivity.py

# Validate against real market data
python validate.py
```

---

## Project Structure

```
BehavioralMarket/
├── prospect_theory.py    Kahneman-Tversky value function
├── agents.py             Five behavioral agent types
├── market.py             Mesa 3.x ABM — GARCH + Merton + herding
├── run.py                Base simulation + visualization
├── compare.py            Humans vs Humans+Algos scenario comparison
├── sensitivity.py        Seven market ecosystem analysis
├── validate.py           Empirical validation vs real S&P 500 data
└── README.md
```

---

## References

- Kahneman, D. & Tversky, A. (1979). *Prospect Theory: An Analysis of Decision under Risk*. Econometrica.
- Tversky, A. & Kahneman, D. (1992). *Advances in Prospect Theory*. Journal of Risk and Uncertainty.
- Merton, R.C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous*. JFE.
- Shefrin, H. & Statman, M. (1985). *The Disposition to Sell Winners Too Early and Ride Losers Too Long*. Journal of Finance.
- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
- Farmer, J.D. & Foley, D. (2009). *The Economy Needs Agent-Based Modelling*. Nature.

---

## De la Simulación a la Estrategia

The ABM's job was to prove a mechanism, not to trade. The follow-up question — can
the same behavioral vocabulary (panic, bubble, herding) be computed on **real** market
data and used as a practical overlay? — lives outside this folder:

- `BehavioralMarket/regime_classifier.py` — classifies real daily returns into
  **PANIC / BUBBLE / HERD / NORMAL** using walk-forward (expanding, no look-ahead)
  percentiles of rolling kurtosis, skewness, drawdown and trend persistence. It does
  **not** match simulated ABM magnitudes against real ones (a Mesa tick isn't a
  trading day) — the four states are a shared vocabulary with the ABM, not a
  numeric fingerprint match.
- `behavioral_overlay_backtest.py` (repo root) — adds a fourth strategy, **S4 = Tech +
  Régimen + Conductual**, to the existing SAR+EMA200+RSI × 4-quadrant regime backtest
  (`tech_regime_backtest.py`), on the same 491 S&P 500 stocks (2018–2023).
- `live_dashboard.py` (repo root) — the current behavioral state now adjusts the daily
  action: it relaxes the Graham filter during capitulation (PANIC) and flags melt-up
  risk during BUBBLE, on top of the existing regime + fundamentals logic.

**Result (S3 vs S4, 2019-06 to 2023-11):** Sharpe 0.36 → 0.36 (flat), Calmar 0.46 → 0.47,
total return +51.2% → +52.3%. The overlay does **not** meaningfully improve
risk-adjusted returns in this backtest — reported honestly rather than oversold. What
it does validate well: the classifier's **PANIC** state fired on exactly
2020-03-09 to 2020-03-23, the height of the COVID crash, with zero false positives
in the 2019–2023 sample.

**Per-year robustness check (`print_yearly_breakdown` in `behavioral_overlay_backtest.py`):**
the real test of a behavioral overlay isn't the crash year — it's whether it *misfires*
during a slow, non-panic bear market. 2022 (S&P −19%, grinding rate-hike bear, the
"Institutional" ABM archetype far more than "Panic Mkt") is the natural stress test:

| Año  | S3 Total | S4 Total | PANIC | BUBBLE | Nota |
|---|---|---|---|---|---|
| 2019 | +6.6%  | +6.6%  | 0  | 12 | BUBBLE flag casi sin efecto (12 días, tamaño chico) |
| 2020 | +30.3% | +32.0% | 10 | 22 | Toda la ganancia de S4 viene de aquí — y aun así el Sharpe empeora (1.64→1.59) |
| 2021 | +15.7% | +15.0% | 0  | 4  | BUBBLE cortó tamaño 4 días sin que hubiera crash ese año — costo puro |
| 2022 | −13.9% | −13.9% | 0  | 0  | **Idéntico a S3** — el clasificador no dispara falsos PANIC/BUBBLE en el bear grind |
| 2023 | +9.3%  | +9.3%  | 0  | 0  | Idéntico a S3 |

Conclusión honesta: el clasificador pasa la prueba de robustez que importa (no
alucina pánico ni burbuja en un bear market lento sin capitulación), pero el
"edge" agregado de S4 vive enteramente en 2020 — y ahí es un empate mixto (mejor
retorno, peor Sharpe y Max DD), no una victoria limpia. Con un solo episodio de
pánico en la muestra, seguir afinando los umbrales sería sobreajustar a un evento
único. El valor real de esta pieza hoy es diagnóstico (clasifica bien el estado del
mercado para lectura humana en `live_dashboard.py`), no una fuente probada de alpha.

## Author

Built as a quantitative finance portfolio project combining behavioral economics,
econometric time-series models, and agent-based simulation.

Stack: Python · Mesa 3.x · NumPy · SciPy · Matplotlib · Pandas
