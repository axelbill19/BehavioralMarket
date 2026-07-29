import numpy as np
from scipy.stats import t as student_t
from mesa import Model
from agents import ValueInvestor, MomentumTrader, PanicSeller, AlgoTrader, NoiseTrader


class MarketModel(Model):
    """
    Agent-Based Market with behavioral heterogeneity.

    Three mechanisms that produce empirically realistic fat tails:

    1. GARCH-like vol multiplier — high vol begets high vol
    2. Student-t(df=4) microstructure noise — heavier tails than Gaussian
    3. Herding signal — agents observe last tick's order flow and amplify

    Exogenous shocks injected at specified ticks simulate news events.
    """

    def __init__(
        self,
        n_value=20,
        n_momentum=30,
        n_panic=25,
        n_algo_trend=15,
        n_algo_mm=10,
        n_noise=50,
        initial_price=100.0,
        market_impact=0.008,          # small — agents set direction, jumps set magnitude
        microstructure_noise=0.001,
        garch_persistence=0.88,
        garch_shock_weight=0.10,
        jump_intensity=0.05,          # 5% per tick — ~30 jumps per 600-tick run
        jump_sigma=0.12,              # 12% std — clearly larger than agent baseline (~2%)
        mean_reversion=0.003,         # stronger anchor — prevents $659 drift
        shocks=None,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.price = initial_price
        self.price_history = [initial_price]
        self.orders_history = []
        self._current_orders = []
        self._market_impact = market_impact
        self._noise_sigma = microstructure_noise
        self.tick = 0

        # GARCH state: rolling conditional volatility
        self._cond_vol = microstructure_noise          # starts at baseline
        self._garch_alpha = garch_shock_weight
        self._garch_beta  = garch_persistence
        self._baseline_var = microstructure_noise ** 2
        self._omega = (1 - garch_persistence - garch_shock_weight) * self._baseline_var

        self._jump_intensity  = jump_intensity
        self._jump_sigma      = jump_sigma
        self._mean_reversion  = mean_reversion
        self._initial_price   = initial_price

        # Herding signal: last tick's net order flow, visible to all agents
        self.last_net_flow = 0.0
        self.last_return   = 0.0

        # Shock schedule: {tick: log_return_magnitude}
        self._shocks = shocks or {}
        self.active_shock = 0.0     # magnitude of shock this tick (agents can read)
        self.shock_history = {}     # {tick: magnitude} — for plotting

        # Mesa 3.x: agents auto-register on __init__
        for _ in range(n_value):
            ValueInvestor(self, initial_price)
        for _ in range(n_momentum):
            MomentumTrader(self, initial_price)
        for _ in range(n_panic):
            PanicSeller(self, initial_price)
        for _ in range(n_algo_trend):
            AlgoTrader(self, initial_price, mode='trend')
        for _ in range(n_algo_mm):
            AlgoTrader(self, initial_price, mode='market_maker')
        for _ in range(n_noise):
            NoiseTrader(self, initial_price)

    def submit_order(self, direction: str, size: int = 1):
        self._current_orders.append((direction, size))

    def _update_garch(self, realized_return: float):
        """GARCH(1,1): var_t = omega + alpha*r_{t-1}^2 + beta*var_{t-1}"""
        prev_var = self._cond_vol ** 2
        new_var  = (self._omega
                    + self._garch_alpha * realized_return ** 2
                    + self._garch_beta  * prev_var)
        self._cond_vol = np.sqrt(max(new_var, 1e-8))

    def step(self):
        # ── inject exogenous shock ─────────────────────────────────────────
        self.active_shock = self._shocks.get(self.tick, 0.0)
        if self.active_shock != 0.0:
            self.price *= np.exp(self.active_shock)
            self.shock_history[self.tick] = self.active_shock

        # ── agent decisions ────────────────────────────────────────────────
        self._current_orders = []
        self.agents.shuffle_do('step')

        # ── price formation ───────────────────────────────────────────────
        buy_vol  = sum(s for d, s in self._current_orders if d == 'buy')
        sell_vol = sum(s for d, s in self._current_orders if d == 'sell')
        total    = buy_vol + sell_vol

        net_flow = 0.0
        if total > 0:
            net_flow = (buy_vol - sell_vol) / total

            # Non-linear amplification: one-sided order flow feeds on itself
            contagion = 1.0 + 0.6 * (net_flow ** 2)        # up to ~1.6x when fully one-sided
            vol_regime = min(self._cond_vol / self._noise_sigma, 3.0)  # cap at 3x
            effective_impact = self._market_impact * contagion * vol_regime
            effective_impact = min(effective_impact, 0.08)  # hard cap: max 8% per tick from flow

            self.price *= np.exp(effective_impact * net_flow)

        # Gaussian microstructure noise (GARCH-scaled)
        gauss_noise = np.random.normal(0, self._cond_vol)
        self.price *= np.exp(np.clip(gauss_noise, -0.06, 0.06))

        # Merton jump-diffusion: Poisson jumps produce the kurtosis excess
        if np.random.random() < self._jump_intensity:
            jump = np.random.normal(0, self._jump_sigma)
            self.price *= np.exp(jump)

        # Soft mean-reversion: prevents price drifting to zero or infinity
        log_dev = np.log(self.price / self._initial_price)
        self.price *= np.exp(-self._mean_reversion * log_dev)

        self.price = max(self.price, 1.0)

        # ── update state ───────────────────────────────────────────────────
        if len(self.price_history) >= 1:
            r = np.log(self.price / self.price_history[-1])
            self._update_garch(r)
            self.last_return = r

        self.last_net_flow = net_flow
        self.price_history.append(self.price)
        self.orders_history.append((self.tick, buy_vol, sell_vol))
        self.tick += 1

    def run(self, steps: int = 500):
        for _ in range(steps):
            self.step()
        return self.price_history, self.orders_history
