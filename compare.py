"""
Comparación de escenarios:
  A — Solo agentes humanos  (ValueInvestor + MomentumTrader + PanicSeller + NoiseTrader)
  B — Humanos + Algoritmos  (agrega 15 trend-algos + 10 market-makers)

Mismo seed, mismos shocks. La diferencia emerge del comportamiento de los algos.
Historia: el shock del -10% en tick 450 dispara un Flash Crash en B pero no en A.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as scipy_kurtosis, norm
from market import MarketModel

# ── configuración ─────────────────────────────────────────────────────────────

SEED   = 42
STEPS  = 600
SHOCKS = {150: -0.06, 300: +0.05, 450: -0.10, 520: +0.04}

SHOCK_LABELS = {
    150: "Macro\n-6%",
    300: "Fed\n+5%",
    450: "Flash\n-10%",
    520: "Recovery\n+4%",
}

# ── colores ───────────────────────────────────────────────────────────────────

DARK_BG  = '#161b22'
BLUE_A   = '#58a6ff'   # escenario A — solo humanos
RED_B    = '#f85149'   # escenario B — humanos + algos
GREEN    = '#3fb950'
ORANGE   = '#d29922'
PURPLE   = '#bc8cff'
GRAY     = '#8b949e'


def style_ax(ax, title, subtitle=''):
    ax.set_facecolor(DARK_BG)
    full_title = f'{title}\n{subtitle}' if subtitle else title
    ax.set_title(full_title, color='white', fontsize=9, fontweight='bold', pad=5)
    ax.tick_params(colors=GRAY, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.5)


def mark_shocks(ax, ymin, ymax):
    for tick, mag in SHOCKS.items():
        color = RED_B if mag < 0 else GREEN
        ax.axvline(tick, color=color, linewidth=1.1, alpha=0.65, linestyle='--')
        label = SHOCK_LABELS.get(tick, f'{mag:+.0%}')
        ax.text(tick + 4, ymin + (ymax - ymin) * 0.06,
                label, color=color, fontsize=6, va='bottom', alpha=0.9)


def rolling_std(arr, window=30):
    out = []
    for i in range(len(arr)):
        w = arr[max(0, i - window): i + 1]
        out.append(float(np.std(w)) if len(w) > 1 else 0.0)
    return out


def max_drawdown(prices):
    peak, dd = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        dd = max(dd, (peak - p) / peak)
    return dd


# ── simulación ────────────────────────────────────────────────────────────────

def run_scenario(label, n_algo_trend, n_algo_mm):
    model = MarketModel(
        n_algo_trend=n_algo_trend,
        n_algo_mm=n_algo_mm,
        shocks=SHOCKS,
        seed=SEED,
    )
    prices, orders = model.run(steps=STEPS)
    prices_arr = np.array(prices)
    log_rets   = np.diff(np.log(prices_arr))
    buy_vols   = np.array([o[1] for o in orders], dtype=float)
    sell_vols  = np.array([o[2] for o in orders], dtype=float)

    kurt  = float(scipy_kurtosis(log_rets))
    skew_ = float(np.mean(((log_rets - log_rets.mean()) / log_rets.std()) ** 3))
    dd    = max_drawdown(prices_arr)

    print(f'  {label}')
    print(f'    Precio final  : ${prices[-1]:.2f}  ({(prices[-1]/prices[0]-1)*100:+.1f}%)')
    print(f'    Max drawdown  : {dd*100:.1f}%')
    print(f'    Kurtosis      : {kurt:.2f}')
    print(f'    Skewness      : {skew_:.2f}')
    print()

    return {
        'prices': prices_arr,
        'log_rets': log_rets,
        'buy_vols': buy_vols,
        'sell_vols': sell_vols,
        'roll_vol': rolling_std(log_rets.tolist()),
        'kurt': kurt,
        'skew': skew_,
        'dd': dd,
    }


# ── visualización ─────────────────────────────────────────────────────────────

def plot_comparison(A, B):
    fig = plt.figure(figsize=(17, 12))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Comparacion de Escenarios  |  Solo Humanos  vs  Humanos + Algoritmos\n'
        'Mismo seed, mismos shocks — la diferencia emerge del comportamiento algoritmico',
        fontsize=12, fontweight='bold', color='white', y=0.985
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.30)

    ticks = np.arange(STEPS)

    # ── 1. Precio (full width) ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1,
             'Trayectoria del Precio',
             'El shock -10% (tick 450) produce cascada mucho mayor con algos')
    ax1.plot(A['prices'], color=BLUE_A, linewidth=1.2, label='A — Solo humanos', zorder=3)
    ax1.plot(B['prices'], color=RED_B,  linewidth=1.2, label='B — Humanos + Algos', zorder=3, alpha=0.9)
    ax1.axhline(100, color=GRAY, linestyle='--', linewidth=0.7, alpha=0.5)
    ax1.set_ylabel('Precio ($)', color=GRAY, fontsize=8)
    ymin1, ymax1 = ax1.get_ylim()
    mark_shocks(ax1, ymin1, ymax1)
    leg = ax1.legend(fontsize=8.5, facecolor='#161b22', edgecolor='#30363d',
                     labelcolor='white', loc='upper left')

    # ── 2. Zoom al Flash Crash (tick 430-480) ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    zoom_s, zoom_e = 420, 490
    style_ax(ax2,
             'Zoom: Flash Crash (tick 450)',
             'Velocidad de caida — algos aceleran la cascada')
    ax2.plot(range(zoom_s, zoom_e), A['prices'][zoom_s:zoom_e],
             color=BLUE_A, linewidth=1.5, label='Solo humanos')
    ax2.plot(range(zoom_s, zoom_e), B['prices'][zoom_s:zoom_e],
             color=RED_B,  linewidth=1.5, label='Con algos')
    ax2.axvline(450, color=RED_B, linewidth=1.2, linestyle='--', alpha=0.7)
    ax2.text(451, ax2.get_ylim()[0] * 1.02, 'Shock\n-10%', color=RED_B, fontsize=6.5)
    ax2.set_ylabel('Precio ($)', color=GRAY, fontsize=8)
    ax2.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # ── 3. Distribución de retornos superpuesta ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3,
             'Distribucion de Log-Retornos',
             f'A kurt={A["kurt"]:.2f}  |  B kurt={B["kurt"]:.2f}')
    ax3.hist(A['log_rets'], bins=60, color=BLUE_A, alpha=0.55, density=True,
             label=f'A (kurt={A["kurt"]:.1f})')
    ax3.hist(B['log_rets'], bins=60, color=RED_B,  alpha=0.55, density=True,
             label=f'B (kurt={B["kurt"]:.1f})')
    mu = np.concatenate([A['log_rets'], B['log_rets']]).mean()
    sg = np.concatenate([A['log_rets'], B['log_rets']]).std()
    x  = np.linspace(
        min(A['log_rets'].min(), B['log_rets'].min()),
        max(A['log_rets'].max(), B['log_rets'].max()), 300)
    ax3.plot(x, norm.pdf(x, mu, sg), color='white', linewidth=1.5,
             linestyle='--', label='Normal ref.', alpha=0.6)
    ax3.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # ── 4. Drawdown acumulado ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4,
             'Drawdown desde Maximo Historico',
             f'A max={A["dd"]*100:.0f}%  |  B max={B["dd"]*100:.0f}%')

    def cum_dd(prices):
        running_max = np.maximum.accumulate(prices)
        return (running_max - prices) / running_max * 100

    ax4.fill_between(range(len(A['prices'])), cum_dd(A['prices']),
                     alpha=0.45, color=BLUE_A, label='A — Solo humanos')
    ax4.fill_between(range(len(B['prices'])), cum_dd(B['prices']),
                     alpha=0.45, color=RED_B,  label='B — Con algos')
    ax4.set_ylabel('Drawdown (%)', color=GRAY, fontsize=8)
    ax4.invert_yaxis()
    ymin4, ymax4 = ax4.get_ylim()
    mark_shocks(ax4, ymax4, ymin4)
    ax4.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # ── 5. Volatilidad rodante superpuesta ────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0:2])
    style_ax(ax5,
             'Volatilidad Rodante (30 ticks)  —  clustering post-shock',
             'Los algos amplian los picos de volatilidad y retrasan la disipacion')
    ax5.plot(A['roll_vol'], color=BLUE_A, linewidth=1.0, label='A — Solo humanos')
    ax5.plot(B['roll_vol'], color=RED_B,  linewidth=1.0, label='B — Con algos', alpha=0.85)
    ax5.set_ylabel('Volatilidad', color=GRAY, fontsize=8)
    ymin5 = 0
    ymax5 = max(max(A['roll_vol']), max(B['roll_vol'])) * 1.15
    mark_shocks(ax5, ymin5, ymax5)
    ax5.legend(fontsize=7.5, facecolor='#161b22', edgecolor='#30363d', labelcolor='white')

    # ── 6. Tabla resumen ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor(DARK_BG)
    ax6.axis('off')
    ax6.set_title('Resumen Comparativo', color='white', fontsize=9,
                  fontweight='bold', pad=5)

    rows = [
        ['Metrica',          'A: Humanos',        'B: + Algos'],
        ['Precio final',     f'${A["prices"][-1]:.1f}', f'${B["prices"][-1]:.1f}'],
        ['Retorno total',    f'{(A["prices"][-1]/100-1)*100:+.1f}%',
                             f'{(B["prices"][-1]/100-1)*100:+.1f}%'],
        ['Max Drawdown',     f'{A["dd"]*100:.1f}%', f'{B["dd"]*100:.1f}%'],
        ['Kurtosis exceso',  f'{A["kurt"]:.2f}',   f'{B["kurt"]:.2f}'],
        ['Skewness',         f'{A["skew"]:.2f}',   f'{B["skew"]:.2f}'],
        ['Max vol rodante',  f'{max(A["roll_vol"]):.4f}', f'{max(B["roll_vol"]):.4f}'],
    ]

    col_colors = [['#1f2937']*3] + [
        ['#161b22', '#1a3a5c', '#3a1a1a']
        for _ in rows[1:]
    ]

    tbl = ax6.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc='center',
        loc='center',
        cellColours=col_colors[1:],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#30363d')
        if r == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 1:
            cell.set_text_props(color=BLUE_A)
        elif c == 2:
            cell.set_text_props(color=RED_B)
        else:
            cell.set_text_props(color=GRAY)

    out_path = 'C:/Users/Corsair/Desktop/Python/BehavioralMarket/comparacion.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Grafica guardada: {out_path}')
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 62)
    print('  Comparacion A vs B  |  mismo seed, mismos shocks')
    print('=' * 62)
    print()

    print('Corriendo Escenario A — Solo humanos (sin algos)...')
    A = run_scenario('Escenario A — Solo humanos', n_algo_trend=0, n_algo_mm=0)

    print('Corriendo Escenario B — Humanos + Algoritmos...')
    B = run_scenario('Escenario B — Humanos + Algoritmos', n_algo_trend=15, n_algo_mm=10)

    plot_comparison(A, B)
