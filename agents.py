import numpy as np
from mesa import Agent
from prospect_theory import prospect_value, disposition_effect_pressure


class ValueInvestor(Agent):
    """
    Graham/Buffett-style. Has a private intrinsic value estimate.
    Buys with margin of safety, sells when overvalued. Acts slowly.
    Largely ignores short-term noise — but reacts to major crashes (buying opp).
    """
    FREQUENCY = 20

    def __init__(self, model, initial_price):
        super().__init__(model)
        self.intrinsic_value = initial_price * np.random.uniform(0.8, 1.2)
        self.margin_of_safety = np.random.uniform(0.25, 0.40)
        self.cash = np.random.uniform(5000, 15000)
        self.shares = np.random.randint(0, 100)
        self._last_acted = 0

    def step(self):
        price = self.model.price
        self.intrinsic_value *= (1 + np.random.normal(0, 0.005))

        # On a large crash shock, act immediately regardless of frequency
        big_crash = self.model.active_shock < -0.06
        due = (self.model.tick - self._last_acted) >= self.FREQUENCY

        if not due and not big_crash:
            return

        if price < self.intrinsic_value * (1 - self.margin_of_safety):
            if self.cash > price:
                # Buy more aggressively during panic (blood in the streets)
                pct = 0.50 if big_crash else 0.30
                qty = int(self.cash * pct / price)
                if qty > 0:
                    self.shares += qty
                    self.cash -= qty * price
                    self.model.submit_order('buy', qty)
        elif price > self.intrinsic_value * 1.20:
            qty = int(self.shares * 0.50)
            if qty > 0:
                self.shares -= qty
                self.cash += qty * price
                self.model.submit_order('sell', qty)

        self._last_acted = self.model.tick


class MomentumTrader(Agent):
    """
    Trend-chaser. Amplifies moves through FOMO and herding.
    Checks both price momentum AND the live order flow herding signal.
    """
    FREQUENCY = 10  # acts infrequently — creates quiet spells between bursts

    def __init__(self, model, initial_price):
        super().__init__(model)
        self.cash = np.random.uniform(3000, 10000)
        self.shares = np.random.randint(0, 50)
        self.fomo_threshold = np.random.uniform(0.015, 0.05)
        self.fear_threshold = np.random.uniform(0.015, 0.05)
        self.lookback = np.random.randint(5, 12)
        self._last_acted = 0

    def step(self):
        if (self.model.tick - self._last_acted) < self.FREQUENCY:
            return
        if len(self.model.price_history) < self.lookback + 1:
            return

        price = self.model.price
        recent_return = (price / self.model.price_history[-self.lookback]) - 1

        # Herding amplifier: if last tick was strongly one-directional, pile on
        herd_boost = 1.0 + 1.2 * abs(self.model.last_net_flow)

        if recent_return > self.fomo_threshold and self.cash > price:
            qty = int(self.cash * 0.45 * herd_boost / price)
            qty = min(qty, int(self.cash / price))  # cap at available cash
            if qty > 0:
                self.shares += qty
                self.cash -= qty * price
                self.model.submit_order('buy', qty)

        elif recent_return < -self.fear_threshold and self.shares > 0:
            qty = int(self.shares * min(0.90, 0.55 * herd_boost))
            if qty > 0:
                self.shares -= qty
                self.cash += qty * price
                self.model.submit_order('sell', qty)

        self._last_acted = self.model.tick


class PanicSeller(Agent):
    """
    Kahneman loss-averse agent. Anchors to reference price.
    Reacts to both cumulative P&L (Prospect Theory) and sudden shocks.
    Disposition effect: sells winners early, holds losers too long... until panic.
    """
    FREQUENCY = 7   # deliberate — panic builds, then breaks all at once

    def __init__(self, model, initial_price):
        super().__init__(model)
        self.cash = np.random.uniform(2000, 8000)
        self.shares = np.random.randint(10, 80)
        self.reference_price = initial_price
        self._last_acted = 0

    def step(self):
        if (self.model.tick - self._last_acted) < self.FREQUENCY:
            return

        price = self.model.price
        pnl_pct = (price - self.reference_price) / self.reference_price
        subjective_value = prospect_value(pnl_pct)
        sell_pressure = disposition_effect_pressure(pnl_pct, self.shares)

        # External shock triggers immediate panic regardless of P&L
        shock_panic = self.model.active_shock < -0.04

        if shock_panic or (pnl_pct < -0.04 and subjective_value < -0.25):
            # PANIC — loss aversion overwhelms rational calculation
            panic_pct = 0.95 if shock_panic else 0.80
            qty = int(self.shares * panic_pct)
            if qty > 0:
                self.shares -= qty
                self.cash += qty * price
                self.model.submit_order('sell', qty)
                self.reference_price = price   # new (painful) reference point

        elif pnl_pct > 0.06 and sell_pressure > 0.5:
            # Disposition effect: sell winners too early
            qty = int(self.shares * 0.45)
            if qty > 0:
                self.shares -= qty
                self.cash += qty * price
                self.model.submit_order('sell', qty)
                self.reference_price = price

        elif pnl_pct > 0.02 and self.cash > price * 5 and not shock_panic:
            # False confidence on small gains — buy more at peak
            qty = int(self.cash * 0.20 / price)
            if qty > 0:
                self.shares += qty
                self.cash -= qty * price
                self.model.submit_order('buy', qty)
                self.reference_price = (self.reference_price + price) / 2

        self._last_acted = self.model.tick


class AlgoTrader(Agent):
    """
    Algorithmic trader. Every tick, no emotion.
    Trend-algo amplifies moves. Market-maker provides liquidity.
    Both react instantly to shocks — algos were first to cause Flash Crash 2010.
    """
    FREQUENCY = 1

    def __init__(self, model, initial_price, mode='trend'):
        super().__init__(model)
        self.mode = mode
        self.cash = np.random.uniform(10000, 50000)
        self.shares = np.random.randint(50, 200)
        self.lookback = np.random.randint(3, 8)

    def step(self):
        if len(self.model.price_history) < self.lookback + 1:
            return

        price = self.model.price
        shock = self.model.active_shock

        if self.mode == 'trend':
            sma = np.mean(self.model.price_history[-self.lookback:])

            # On a big negative shock, trend-algos accelerate the crash
            if shock < -0.04 and self.shares > 0:
                qty = int(self.shares * 0.60)
                if qty > 0:
                    self.shares -= qty
                    self.cash += qty * price
                    self.model.submit_order('sell', qty)
                return

            if price > sma * 1.001 and self.cash > price:
                qty = int(self.cash * 0.35 / price)
                if qty > 0:
                    self.shares += qty
                    self.cash -= qty * price
                    self.model.submit_order('buy', qty)
            elif price < sma * 0.999 and self.shares > 0:
                qty = int(self.shares * 0.35)
                if qty > 0:
                    self.shares -= qty
                    self.cash += qty * price
                    self.model.submit_order('sell', qty)

        elif self.mode == 'market_maker':
            # Market-makers widen spreads during shocks (liquidity withdrawal)
            if abs(shock) > 0.05:
                return   # step back from market during extreme events

            if len(self.model.price_history) >= 2:
                tick_ret = (price / self.model.price_history[-2]) - 1
                if tick_ret < -0.008 and self.cash > price:
                    qty = int(self.cash * 0.08 / price)
                    if qty > 0:
                        self.shares += qty
                        self.cash -= qty * price
                        self.model.submit_order('buy', qty)
                elif tick_ret > 0.008 and self.shares > 10:
                    qty = int(self.shares * 0.08)
                    if qty > 0:
                        self.shares -= qty
                        self.cash += qty * price
                        self.model.submit_order('sell', qty)


class NoiseTrader(Agent):
    """
    Uninformed retail. Random with slight directional bias.
    During shocks, noise traders freeze or panic-sell (herding with media).
    """
    FREQUENCY = 5

    def __init__(self, model, initial_price):
        super().__init__(model)
        self.cash = np.random.uniform(1000, 5000)
        self.shares = np.random.randint(0, 30)
        self.bullish_bias = np.random.uniform(-0.15, 0.15)
        self._last_acted = 0

    def step(self):
        if (self.model.tick - self._last_acted) < self.FREQUENCY:
            return

        shock = self.model.active_shock
        if shock < -0.05 and self.shares > 0:
            # Media panic — noise traders dump holdings
            qty = int(self.shares * np.random.uniform(0.3, 0.7))
            if qty > 0:
                self.shares -= qty
                self.cash += qty * self.model.price
                self.model.submit_order('sell', qty)
            self._last_acted = self.model.tick
            return

        buy_prob = 0.50 + self.bullish_bias
        # Herding: tilt toward the crowd if flow is strong
        buy_prob += 0.20 * self.model.last_net_flow

        if np.random.random() < buy_prob:
            if self.cash > self.model.price:
                qty = np.random.randint(1, 6)
                self.shares += qty
                self.cash -= qty * self.model.price
                self.model.submit_order('buy', qty)
        else:
            if self.shares > 0:
                qty = np.random.randint(1, max(2, self.shares // 2))
                self.shares -= qty
                self.cash += qty * self.model.price
                self.model.submit_order('sell', qty)

        self._last_acted = self.model.tick
