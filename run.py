import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import norm, kurtosis as scipy_kurtosis
from market import MarketModel


# ── helpers ──────────────────────────────────────────────────────────────────

def rolling_std(arr, window):
    out = []
    for i in range(len(arr)):
        w = arr[max(0, i - window):i + 1]
        out.append(float(np.std(w)) if len(w) > 1 else 0.0)
    return out


def max_drawdown(prices):
    peak, max_dd = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        max_dd = max(max_dd, (peak - p) / peak)
    return max_dd


# ── scenario definition ───────────────────────────────────────────────────────

SHOCKS = {
    150: -0.06,   # -6%  bad earnings / macro scare
    300: +0.05,   # +5%  Fed pivot surprise
    450: -0.10,   # -10% flash-crash type event
    520: +0.04,   # +4%  partial recovery news
}

SHOCK_LABELS = {
    150: "Macro scare\n-6%",
    300: "Fed pivot\n+5%",
    450: "Flash crash\n-10%",
    520: "Recovery\n+4%",
}


# ── plotting ──────────────────────────────────────────────────────────────────

DARK_BG = '#161b22'
BLUE    = '#58a6ff'
GREEN   = '#3fb950'
RED     = '#f85149'
ORANGE  = '#d29922'
PURPLE  = '#bc8cff'
YELLOW  = '#e3b341'
GRAY    = '#8b949e'


def style_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_title(title, color='white', fontsize=9, fontweight='bold', pad=5)
    ax.tick_params(colors=GRAY, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.5)


def mark_shocks(ax, shock_dict, shock_labels, ymin, ymax):
    for tick, mag in shock_dict.items():
        color = RED if mag < 0 else GREEN
        ax.axvline(tick, color=color, linewidth=1.2, alpha=0.7, linestyle='--')
        label = shock_labels.get(tick, f"{mag:+.0%}")
        ax.text(tick + 3, ymin + (ymax - ymin) * 0.05,
                label, color=color, fontsize=6.5, va='bottom')


# ── simulation ────────────────────────────────────────────────────────────────

def run_simulation(steps=600, seed=42, shocks=None):
    if shocks is None:
        shocks = SHOCKS

    print("=" * 62)
    print("  Mercado Conductual  |  GARCH + Student-t + Herding + Shocks")
    print("=" * 62)
    print(f"  150 agentes  |  {steps} ticks  |  seed={seed}")
    print(f"  Shocks programados: {len(shocks)}")
    for tick, mag in sorted(shocks.items()):
        print(f"    tick {tick:>4}: {mag:+.1%}")
    print("=" * 62)

    model = MarketModel(seed=seed, shocks=shocks)
    prices, orders = model.run(steps=steps)

    prices_arr = np.array(prices)
    log_rets   = np.diff(np.log(prices_arr))   # log returns
    ticks      = [o[0] for o in orders]
    buy_vols   = np.array([o[1] for o in orders], dtype=float)
    sell_vols  = np.array([o[2] for o in orders], dtype=float)
    roll_vol   = rolling_std(log_rets.tolist(), 30)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Simulacion de Mercado Conductual  |  Prospect Theory + GARCH + Shocks Exogenos',
        fontsize=13, fontweight='bold', color='white', y=0.985
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.32)

    # 1. Precio (spans all 3 cols)
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, 'Trayectoria del Precio  —  los shocks desencadenan reacciones en cascada')
    ax1.plot(prices_arr, color=BLUE, linewidth=1.0, zorder=3)
    ax1.axhline(100, color=GRAY, linestyle='--', linewidth=0.7, alpha=0.5)
    ax1.set_ylabel('Precio ($)', color=GRAY, fontsize=8)
    ymin1, ymax1 = ax1.get_ylim()
    mark_shocks(ax1, model.shock_history, SHOCK_LABELS, ymin1, ymax1)

    # 2. Log-retornos
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, 'Log-Retornos por Tick')
    ax2.plot(log_rets, color=GREEN, linewidth=0.55, alpha=0.85)
    ax2.axhline(0, color=GRAY, linestyle='--', linewidth=0.6)
    ax2.set_ylabel('Log-retorno', color=GRAY, fontsize=8)
    mark_shocks(ax2, model.shock_history, {}, log_rets.min(), log_rets.max())

    # 3. Distribución de retornos — fat tails
    ax3 = fig.add_subplot(gs[1, 1])
    kurt_val = float(scipy_kurtosis(log_rets))
    style_ax(ax3, f'Distribucion de Retornos  (kurtosis={kurt_val:.2f})')
    ax3.hist(log_rets, bins=70, color=PURPLE, alpha=0.75, density=True, label='Simulado')
    mu, sigma = log_rets.mean(), log_rets.std()
    x_range = np.linspace(log_rets.min(), log_rets.max(), 400)
    ax3.plot(x_range, norm.pdf(x_range, mu, sigma), color=RED,
             linewidth=2, label='Normal teorica')
    ax3.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # 4. QQ-plot (muestra fat tails visualmente)
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, 'QQ-Plot vs Normal  (colas gordas = puntos fuera de la linea)')
    sorted_rets = np.sort(log_rets)
    n = len(sorted_rets)
    theoretical_q = norm.ppf(np.linspace(0.01, 0.99, n), mu, sigma)
    ax4.scatter(theoretical_q, sorted_rets, s=2, color=YELLOW, alpha=0.6)
    ax4.plot([theoretical_q[0], theoretical_q[-1]],
             [theoretical_q[0], theoretical_q[-1]],
             color=RED, linewidth=1.5, label='Normal perfecta')
    ax4.set_xlabel('Cuantiles teoricos', color=GRAY, fontsize=7)
    ax4.set_ylabel('Cuantiles observados', color=GRAY, fontsize=7)
    ax4.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # 5. Flujo de ordenes
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'Flujo de Ordenes (compras vs ventas)')
    ax5.fill_between(ticks, buy_vols,  alpha=0.65, color=GREEN, label='Compras')
    ax5.fill_between(ticks, -sell_vols, alpha=0.65, color=RED,   label='Ventas')
    ax5.axhline(0, color=GRAY, linewidth=0.5)
    ax5.set_ylabel('Volumen', color=GRAY, fontsize=8)
    ax5.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')
    mark_shocks(ax5, model.shock_history, {}, -sell_vols.max(), buy_vols.max())

    # 6. Volatilidad rodante — clustering
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, 'Volatilidad Rodante (30 ticks)  —  clustering post-shock')
    ax6.plot(roll_vol, color=ORANGE, linewidth=1.0)
    ax6.set_ylabel('Volatilidad', color=GRAY, fontsize=8)
    mark_shocks(ax6, model.shock_history, {}, 0, max(roll_vol) * 1.1)

    # 7. Drawdown
    ax7 = fig.add_subplot(gs[2, 2])
    style_ax(ax7, 'Drawdown desde Maximo')
    running_max = np.maximum.accumulate(prices_arr)
    drawdown_series = (running_max - prices_arr) / running_max * 100
    ax7.fill_between(range(len(drawdown_series)), drawdown_series,
                     color=RED, alpha=0.55)
    ax7.set_ylabel('Drawdown (%)', color=GRAY, fontsize=8)
    ax7.invert_yaxis()
    mark_shocks(ax7, model.shock_history, {}, 0, drawdown_series.max() * 1.1)

    out_path = 'C:/Users/Corsair/Desktop/Python/BehavioralMarket/simulacion.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\nGrafica guardada: {out_path}")
    plt.show()

    # ── estadisticas ──────────────────────────────────────────────────────────
    total_ret = (prices[-1] / prices[0] - 1) * 100
    max_dd    = max_drawdown(prices_arr) * 100
    skew      = float(np.mean(((log_rets - log_rets.mean()) / log_rets.std()) ** 3))

    print("\n── Estadisticas ───────────────────────────────────────────")
    print(f"  Precio inicial   : ${prices[0]:.2f}")
    print(f"  Precio final     : ${prices[-1]:.2f}")
    print(f"  Retorno total    : {total_ret:+.1f}%")
    print(f"  Max drawdown     : {max_dd:.1f}%")
    print(f"  Kurtosis exceso  : {kurt_val:.2f}  (mercados reales: 4-8)")
    print(f"  Skewness         : {skew:.2f}  (negativo = cola izq mas gruesa)")
    print("───────────────────────────────────────────────────────────")
    print("  GARCH + Student-t + herding producen fat tails realistas.")
    print("  Cada shock desencadena reacciones distintas por tipo de agente:")
    print("    Algo-trend  -> venden al instante (Flash Crash)")
    print("    PanicSeller -> loss aversion los paraliza... luego avalancha")
    print("    ValueInvest -> ven oportunidad, compran el dip")
    print("    Noise       -> siguen a los medios, venden tarde")
    print("───────────────────────────────────────────────────────────\n")


if __name__ == '__main__':
    run_simulation(steps=600, seed=42)
