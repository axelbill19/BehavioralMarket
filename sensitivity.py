"""
Análisis de sensibilidad — ¿cómo cambia el mercado si cambias la mezcla de agentes?

Cada escenario es un "tipo de mercado" diferente:
  Baseline      → mercado mixto normal
  Panic Mkt     → dominado por loss-aversion (crisis financiera)
  No Anchor     → sin value investors (burbuja sin corrección fundamental)
  HFT Dom.      → algoritmos 5x (mercado moderno de alta frecuencia)
  Random Walk   → solo noise traders (hipótesis de mercado eficiente pura)
  Institutional → sin retail (mercado institucional puro)
  Herd Mkt      → momentum traders 2x (FOMO extremo, tendencias fuertes)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as sci_kurt
from market import MarketModel

# ── configuración ─────────────────────────────────────────────────────────────

SEEDS  = [42, 7, 13, 21, 99]     # promediamos 5 seeds para robustez
STEPS  = 600
SHOCKS = {150: -0.06, 300: +0.05, 450: -0.10, 520: +0.04}

SCENARIOS = [
    {
        'name': 'Baseline',
        'desc': 'Mercado mixto\n(setup actual)',
        'kwargs': dict(n_value=20, n_momentum=30, n_panic=25,
                       n_algo_trend=15, n_algo_mm=10, n_noise=50),
        'color': '#58a6ff',
    },
    {
        'name': 'Panic Mkt',
        'desc': 'Dominado por\nloss-aversion',
        'kwargs': dict(n_value=10, n_momentum=20, n_panic=70,
                       n_algo_trend=5, n_algo_mm=5, n_noise=40),
        'color': '#f85149',
    },
    {
        'name': 'No Anchor',
        'desc': 'Sin value investors\n(burbuja sin piso)',
        'kwargs': dict(n_value=0, n_momentum=40, n_panic=30,
                       n_algo_trend=15, n_algo_mm=10, n_noise=55),
        'color': '#d29922',
    },
    {
        'name': 'HFT Dom.',
        'desc': 'Algoritmos 5x\n(mercado moderno)',
        'kwargs': dict(n_value=10, n_momentum=15, n_panic=10,
                       n_algo_trend=75, n_algo_mm=20, n_noise=20),
        'color': '#bc8cff',
    },
    {
        'name': 'Random Walk',
        'desc': 'Solo noise traders\n(EMH pura)',
        'kwargs': dict(n_value=0, n_momentum=0, n_panic=0,
                       n_algo_trend=0, n_algo_mm=0, n_noise=150),
        'color': '#8b949e',
    },
    {
        'name': 'Institutional',
        'desc': 'Sin retail\n(mercado institucional)',
        'kwargs': dict(n_value=40, n_momentum=30, n_panic=20,
                       n_algo_trend=25, n_algo_mm=20, n_noise=0),
        'color': '#3fb950',
    },
    {
        'name': 'Herd Mkt',
        'desc': 'Momentum 2x\n(FOMO extremo)',
        'kwargs': dict(n_value=10, n_momentum=70, n_panic=20,
                       n_algo_trend=10, n_algo_mm=5, n_noise=35),
        'color': '#e3b341',
    },
]

# ── métricas ──────────────────────────────────────────────────────────────────

def max_drawdown(prices):
    peak, dd = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        dd   = max(dd, (peak - p) / peak)
    return dd


def rolling_std(arr, w=30):
    out = []
    for i in range(len(arr)):
        chunk = arr[max(0, i - w): i + 1]
        out.append(float(np.std(chunk)) if len(chunk) > 1 else 0.0)
    return out


def run_scenario(kwargs, seeds=SEEDS):
    kurtosis_list, dd_list, ret_list, vol_list = [], [], [], []
    all_prices = []

    for seed in seeds:
        m = MarketModel(**kwargs, shocks=SHOCKS, seed=seed)
        prices, _ = m.run(steps=STEPS)
        pa = np.array(prices)
        lr = np.diff(np.log(pa))

        kurtosis_list.append(float(sci_kurt(lr)))
        dd_list.append(max_drawdown(pa))
        ret_list.append((pa[-1] / pa[0]) - 1)
        vol_list.append(float(np.std(lr)))
        all_prices.append(pa)

    # representativo = seed mediano por precio final
    median_idx = int(np.argsort(ret_list)[len(ret_list) // 2])

    return {
        'kurt':      np.mean(kurtosis_list),
        'kurt_std':  np.std(kurtosis_list),
        'dd':        np.mean(dd_list),
        'ret':       np.mean(ret_list),
        'vol':       np.mean(vol_list),
        'rep_price': all_prices[median_idx],
    }


# ── visualización ─────────────────────────────────────────────────────────────

DARK_BG = '#161b22'
GRAY    = '#8b949e'


def style_ax(ax, title, subtitle=''):
    ax.set_facecolor(DARK_BG)
    ttl = f'{title}\n{subtitle}' if subtitle else title
    ax.set_title(ttl, color='white', fontsize=8.5, fontweight='bold', pad=5)
    ax.tick_params(colors=GRAY, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.5)


def bar_chart(ax, values, errs, label, colors, ylabel):
    names = [s['name'] for s in SCENARIOS]
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, alpha=0.82, width=0.65,
                  yerr=errs if errs is not None else None,
                  error_kw=dict(ecolor=GRAY, capsize=3, linewidth=1))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel(ylabel, color=GRAY, fontsize=8)
    ax.axhline(0, color=GRAY, linewidth=0.6, linestyle='--')
    return bars


def plot_sensitivity(results):
    colors = [s['color'] for s in SCENARIOS]
    names  = [s['name'] for s in SCENARIOS]

    fig = plt.figure(figsize=(17, 13))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Analisis de Sensibilidad  |  7 Mezclas de Agentes x 5 Seeds\n'
        'Como cambia el mercado segun quien lo domina',
        fontsize=12, fontweight='bold', color='white', y=0.985
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.32)

    # ── 1. Trayectorias representativas (full width) ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1, 'Trayectoria de Precio por Escenario (seed mediano de 5)',
             'Mismo seed base, mismos shocks — solo cambia la mezcla de agentes')
    for s, res in zip(SCENARIOS, results):
        ax1.plot(res['rep_price'], color=s['color'], linewidth=1.1,
                 label=s['name'], alpha=0.85)
    ax1.axhline(100, color=GRAY, linestyle='--', linewidth=0.6, alpha=0.5)
    ax1.set_ylabel('Precio ($)', color=GRAY, fontsize=8)
    for tick, mag in SHOCKS.items():
        ax1.axvline(tick, color='white', linewidth=0.6, alpha=0.25, linestyle=':')
    leg = ax1.legend(ncol=7, fontsize=7, facecolor='#161b22',
                     edgecolor='#30363d', labelcolor='white',
                     loc='upper left', bbox_to_anchor=(0, 1.0))

    # ── 2. Kurtosis (con error bar) ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, 'Kurtosis Exceso', '(media ± std de 5 seeds)  >0 = fat tails')
    kurts  = [r['kurt']     for r in results]
    k_stds = [r['kurt_std'] for r in results]
    bar_chart(ax2, kurts, k_stds, 'kurtosis', colors, 'Kurtosis exceso')
    ax2.axhspan(4, 8, alpha=0.08, color='white', label='Rango real')
    ax2.text(6.3, 5.5, 'Rango\nreal', color=GRAY, fontsize=6.5, ha='center')

    # ── 3. Max Drawdown ───────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, 'Max Drawdown Promedio', 'menor = mercado mas estable')
    dds = [r['dd'] * 100 for r in results]
    bar_chart(ax3, dds, None, 'dd', colors, 'Max drawdown (%)')

    # ── 4. Retorno total promedio ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, 'Retorno Total Promedio', 'sesgo alcista/bajista de cada ecosistema')
    rets = [r['ret'] * 100 for r in results]
    bar_chart(ax4, rets, None, 'ret', colors, 'Retorno total (%)')

    # ── 5. Volatilidad media ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, 'Volatilidad Media (std log-ret)', '')
    vols = [r['vol'] * 100 for r in results]
    bar_chart(ax5, vols, None, 'vol', colors, 'Volatilidad (%)')

    # ── 6. Spider / radar chart — perfil de cada mercado ─────────────────────
    ax6 = fig.add_subplot(gs[2, 1], polar=True)
    ax6.set_facecolor(DARK_BG)
    ax6.set_title('Perfil de Riesgo por Escenario\n(normalizado 0-1)',
                  color='white', fontsize=8.5, fontweight='bold', pad=15)

    metrics_labels = ['Kurtosis', 'Drawdown', 'Volatilidad', 'Retorno\n(abs)']
    N = len(metrics_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    raw = np.array([
        [abs(r['kurt'])    for r in results],
        [r['dd']           for r in results],
        [r['vol']          for r in results],
        [abs(r['ret'])     for r in results],
    ])
    norm_raw = raw / (raw.max(axis=1, keepdims=True) + 1e-9)

    for i, (s, res) in enumerate(zip(SCENARIOS, results)):
        vals = norm_raw[:, i].tolist()
        vals += vals[:1]
        ax6.plot(angles, vals, color=s['color'], linewidth=1.2, alpha=0.8)
        ax6.fill(angles, vals, color=s['color'], alpha=0.07)

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics_labels, color='white', fontsize=7.5)
    ax6.tick_params(colors=GRAY, labelsize=6.5)
    ax6.set_facecolor(DARK_BG)
    ax6.spines['polar'].set_color('#30363d')
    ax6.yaxis.set_visible(False)

    # ── 7. Tabla resumen ──────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor(DARK_BG)
    ax7.axis('off')
    ax7.set_title('Tabla Resumen', color='white', fontsize=8.5,
                  fontweight='bold', pad=5)

    headers = ['Escenario', 'Kurt', 'DD%', 'Ret%', 'Vol%']
    rows = []
    for s, r in zip(SCENARIOS, results):
        rows.append([
            s['name'],
            f'{r["kurt"]:+.1f}',
            f'{r["dd"]*100:.0f}%',
            f'{r["ret"]*100:+.0f}%',
            f'{r["vol"]*100:.2f}%',
        ])

    tbl = ax7.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.55)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor('#161b22')
        cell.set_edgecolor('#30363d')
        if row == 0:
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 0:
            cell.set_text_props(color=SCENARIOS[row - 1]['color'], fontweight='bold')
        else:
            cell.set_text_props(color=GRAY)

    out = 'C:/Users/Corsair/Desktop/Python/BehavioralMarket/sensibilidad.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Grafica guardada: {out}')
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 62)
    print('  Analisis de sensibilidad  |  7 escenarios x 5 seeds')
    print('=' * 62)

    results = []
    for s in SCENARIOS:
        print(f'  Corriendo: {s["name"]:<14} ', end='', flush=True)
        res = run_scenario(s['kwargs'])
        results.append(res)
        print(f'kurt={res["kurt"]:+.2f}  dd={res["dd"]*100:.0f}%  '
              f'ret={res["ret"]*100:+.0f}%  vol={res["vol"]*100:.2f}%')

    print()
    plot_sensitivity(results)
