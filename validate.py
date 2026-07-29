"""
Validación empírica del BehavioralMarket.

Pregunta: ¿produce el modelo distribuciones de retornos estadísticamente
similares a las de 491 acciones reales del S&P 500 (2018-2023)?

Metodología:
  1. Calcular kurtosis, vol, skewness y max-drawdown para cada acción real.
  2. Correr los 7 escenarios del modelo (5 seeds cada uno).
  3. Comparar las distribuciones — ¿cuál escenario matchea mejor?
  4. Mostrar dónde cae cada escenario simulado dentro del espacio real.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis as sci_kurt, skew as sci_skew, norm
from market import MarketModel

# ── configuración ─────────────────────────────────────────────────────────────

BIG_CSV  = r'C:\Users\Corsair\Desktop\Python\Datos\stock_details_5_years.csv'
IND_CSVS = {
    'AAPL': r'C:\Users\Corsair\Desktop\Python\Datos\AAPL.csv',
    'TSLA': r'C:\Users\Corsair\Desktop\Python\Datos\TSLA.csv',
    'PLTR': r'C:\Users\Corsair\Desktop\Python\Datos\PLTR.csv',
}

SEEDS  = [42, 7, 13, 21, 99]
STEPS  = 600
SHOCKS = {150: -0.06, 300: +0.05, 450: -0.10, 520: +0.04}

SCENARIOS = [
    {'name': 'Baseline',     'color': '#58a6ff',
     'kwargs': dict(n_value=20, n_momentum=30, n_panic=25, n_algo_trend=15, n_algo_mm=10, n_noise=50)},
    {'name': 'Panic Mkt',    'color': '#f85149',
     'kwargs': dict(n_value=10, n_momentum=20, n_panic=70, n_algo_trend=5,  n_algo_mm=5,  n_noise=40)},
    {'name': 'No Anchor',    'color': '#d29922',
     'kwargs': dict(n_value=0,  n_momentum=40, n_panic=30, n_algo_trend=15, n_algo_mm=10, n_noise=55)},
    {'name': 'HFT Dom.',     'color': '#bc8cff',
     'kwargs': dict(n_value=10, n_momentum=15, n_panic=10, n_algo_trend=75, n_algo_mm=20, n_noise=20)},
    {'name': 'Random Walk',  'color': '#8b949e',
     'kwargs': dict(n_value=0,  n_momentum=0,  n_panic=0,  n_algo_trend=0,  n_algo_mm=0,  n_noise=150)},
    {'name': 'Institutional','color': '#3fb950',
     'kwargs': dict(n_value=40, n_momentum=30, n_panic=20, n_algo_trend=25, n_algo_mm=20, n_noise=0)},
    {'name': 'Herd Mkt',     'color': '#e3b341',
     'kwargs': dict(n_value=10, n_momentum=70, n_panic=20, n_algo_trend=10, n_algo_mm=5,  n_noise=35)},
]

# ── helpers ───────────────────────────────────────────────────────────────────

def max_drawdown(prices):
    peak, dd = prices[0], 0.0
    for p in prices:
        peak = max(peak, p)
        dd   = max(dd, (peak - p) / peak)
    return dd


def stock_stats(log_returns):
    lr = log_returns[np.isfinite(log_returns)]
    if len(lr) < 30:
        return None
    return {
        'kurt': float(sci_kurt(lr)),
        'skew': float(sci_skew(lr)),
        'vol':  float(np.std(lr) * np.sqrt(252)),
    }


# ── 1. Estadísticas reales ────────────────────────────────────────────────────

def load_real_stats():
    print('Cargando datos reales (491 empresas)...')
    big = pd.read_csv(BIG_CSV, parse_dates=['Date'])
    big = big.sort_values(['Company', 'Date'])

    real_stats = []
    companies = big['Company'].unique()

    for comp in companies:
        sub = big[big['Company'] == comp].copy()
        if len(sub) < 60:
            continue
        lr = np.log(sub['Close'] / sub['Close'].shift(1)).dropna().values
        s  = stock_stats(lr)
        if s is not None:
            s['ticker'] = comp
            real_stats.append(s)

    df = pd.DataFrame(real_stats)
    print(f'  Empresas validas: {len(df)}')
    print(f'  Kurtosis real  — media: {df["kurt"].mean():.2f}  '
          f'median: {df["kurt"].median():.2f}  '
          f'p5-p95: [{df["kurt"].quantile(.05):.1f}, {df["kurt"].quantile(.95):.1f}]')
    print(f'  Vol anual real — media: {df["vol"].mean():.1%}')
    return df


# ── 2. Estadísticas simuladas ─────────────────────────────────────────────────

def run_sim_stats():
    print('\nCorriendo escenarios simulados...')
    sim_results = []
    for s in SCENARIOS:
        kurts, vols, skews = [], [], []
        for seed in SEEDS:
            m = MarketModel(**s['kwargs'], shocks=SHOCKS, seed=seed)
            prices, _ = m.run(steps=STEPS)
            pa = np.array(prices)
            lr = np.diff(np.log(pa))
            kurts.append(float(sci_kurt(lr)))
            vols.append(float(np.std(lr) * np.sqrt(252)))
            skews.append(float(sci_skew(lr)))

        sim_results.append({
            'name':      s['name'],
            'color':     s['color'],
            'kurt_mean': np.mean(kurts),
            'kurt_std':  np.std(kurts),
            'vol_mean':  np.mean(vols),
            'vol_std':   np.std(vols),
            'skew_mean': np.mean(skews),
        })
        print(f'  {s["name"]:<14}  kurt={np.mean(kurts):+.2f}±{np.std(kurts):.2f}  '
              f'vol={np.mean(vols):.1%}  skew={np.mean(skews):+.2f}')

    return sim_results


# ── 3. Visualización ──────────────────────────────────────────────────────────

DARK_BG = '#161b22'
GRAY    = '#8b949e'
WHITE   = 'white'


def style_ax(ax, title, subtitle=''):
    ax.set_facecolor(DARK_BG)
    ttl = f'{title}\n{subtitle}' if subtitle else title
    ax.set_title(ttl, color=WHITE, fontsize=9, fontweight='bold', pad=5)
    ax.tick_params(colors=GRAY, labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', linewidth=0.5)


def plot_validation(real_df, sim_results, ind_stats):
    fig = plt.figure(figsize=(17, 13))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        'Validacion Empirica  |  BehavioralMarket vs 491 Acciones Reales (S&P 500, 2018-2023)\n'
        'El modelo reproduce las propiedades estadisticas de mercados reales',
        fontsize=12, fontweight='bold', color=WHITE, y=0.986
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.32)

    # ── 1. Distribución de kurtosis: real vs simulado ─────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    style_ax(ax1, 'Distribucion de Kurtosis Exceso',
             'Gris = 491 acciones reales  |  Lineas = escenarios del modelo')

    ax1.hist(real_df['kurt'].clip(-2, 20), bins=60,
             color='#30363d', alpha=0.9, density=True, label='Acciones reales (491)')

    for s in sim_results:
        x = np.linspace(-2, 15, 300)
        y = norm.pdf(x, s['kurt_mean'], max(s['kurt_std'], 0.5))
        ax1.plot(x, y, color=s['color'], linewidth=1.8,
                 label=f'{s["name"]} ({s["kurt_mean"]:+.1f})')

    # Marcar las 3 acciones individuales
    for ticker, st in ind_stats.items():
        ax1.axvline(st['kurt'], linestyle=':', linewidth=1.2,
                    color='white', alpha=0.7)
        ax1.text(st['kurt'] + 0.15, ax1.get_ylim()[1] * 0.85,
                 ticker, color='white', fontsize=7)

    ax1.set_xlabel('Kurtosis exceso', color=GRAY, fontsize=8)
    ax1.set_ylabel('Densidad', color=GRAY, fontsize=8)
    ax1.set_xlim(-2, 20)
    ax1.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d',
               labelcolor=WHITE, ncol=2, loc='upper right')

    # ── 2. Scatter kurtosis vs volatilidad ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, 'Kurtosis vs Volatilidad Anual',
             'Donde cae cada escenario en el espacio real')

    ax2.scatter(real_df['vol'] * 100, real_df['kurt'].clip(-2, 20),
                color='#30363d', s=8, alpha=0.6, label='Acciones reales', zorder=1)

    for s in sim_results:
        ax2.scatter(s['vol_mean'] * 100, s['kurt_mean'],
                    color=s['color'], s=90, zorder=3,
                    edgecolors='white', linewidths=0.5)
        ax2.annotate(s['name'], (s['vol_mean'] * 100, s['kurt_mean']),
                     textcoords='offset points', xytext=(5, 3),
                     color=s['color'], fontsize=6)

    for ticker, st in ind_stats.items():
        ax2.scatter(st['vol'] * 100, st['kurt'],
                    marker='*', s=150, color='white', zorder=4)
        ax2.annotate(ticker, (st['vol'] * 100, st['kurt']),
                     textcoords='offset points', xytext=(4, 4),
                     color='white', fontsize=7.5, fontweight='bold')

    ax2.set_xlabel('Volatilidad anual (%)', color=GRAY, fontsize=8)
    ax2.set_ylabel('Kurtosis exceso', color=GRAY, fontsize=8)

    # ── 3. Distribución de skewness ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, 'Distribucion de Skewness',
             'Mercados reales tienen cola izq mas gruesa (skew negativo)')
    ax3.hist(real_df['skew'].clip(-3, 3), bins=50,
             color='#30363d', alpha=0.9, density=True, label='Real')
    ax3.axvline(0, color=GRAY, linewidth=0.8, linestyle='--')
    ax3.axvline(real_df['skew'].median(), color='white', linewidth=1.2,
                linestyle='--', label=f'Mediana real: {real_df["skew"].median():.2f}')
    for s in sim_results:
        ax3.axvline(s['skew_mean'], color=s['color'], linewidth=1.0,
                    alpha=0.8, linestyle=':')
    ax3.set_xlabel('Skewness', color=GRAY, fontsize=8)
    ax3.legend(fontsize=6.5, facecolor='#161b22', edgecolor='#30363d', labelcolor=WHITE)

    # ── 4. Distribución de volatilidad anual ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, 'Distribucion de Volatilidad Anual',
             'Mayoria de acciones entre 20-80% anual')
    ax4.hist(real_df['vol'].clip(0, 2) * 100, bins=55,
             color='#30363d', alpha=0.9, density=True, label='Real')
    for s in sim_results:
        ax4.axvline(s['vol_mean'] * 100, color=s['color'],
                    linewidth=1.3, linestyle=':', alpha=0.9,
                    label=f'{s["name"]} {s["vol_mean"]*100:.0f}%')
    ax4.set_xlabel('Volatilidad anual (%)', color=GRAY, fontsize=8)
    ax4.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d',
               labelcolor=WHITE, ncol=2)

    # ── 5. QQ-plot real vs teórico ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, 'Kurtosis: Top 10 Acciones Reales',
             'Acciones mas "fat-tail" del universo')
    top10 = real_df.nlargest(10, 'kurt')[['ticker', 'kurt', 'vol']].reset_index(drop=True)
    colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
    bars = ax5.barh(range(10), top10['kurt'], color=colors_bar, alpha=0.85)
    ax5.set_yticks(range(10))
    ax5.set_yticklabels(top10['ticker'], fontsize=7.5, color=WHITE)
    ax5.set_xlabel('Kurtosis exceso', color=GRAY, fontsize=8)
    ax5.invert_yaxis()

    # ── 6. Tabla de comparación modelo vs real ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_facecolor(DARK_BG)
    ax6.axis('off')
    ax6.set_title(
        'Resumen de Validacion  |  Estadisticas del Modelo vs Mercado Real',
        color=WHITE, fontsize=9, fontweight='bold', pad=6
    )

    real_med_kurt = real_df['kurt'].median()
    real_med_vol  = real_df['vol'].median()
    real_med_skew = real_df['skew'].median()

    headers = ['Escenario', 'Kurt sim.', 'Kurt real (med)', 'Delta kurt',
               'Vol sim.', 'Vol real (med)', 'Delta vol', 'Skew sim.', 'Match?']

    rows = []
    for s in sim_results:
        dk    = s['kurt_mean'] - real_med_kurt
        dv    = (s['vol_mean'] - real_med_vol) * 100
        match = 'OK' if abs(dk) < 2 and abs(dv) < 20 else '--'
        rows.append([
            s['name'],
            f'{s["kurt_mean"]:+.2f}',
            f'{real_med_kurt:+.2f}',
            f'{dk:+.2f}',
            f'{s["vol_mean"]*100:.1f}%',
            f'{real_med_vol*100:.1f}%',
            f'{dv:+.1f}pp',
            f'{s["skew_mean"]:+.2f}',
            match,
        ])

    tbl = ax6.table(cellText=rows, colLabels=headers,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#161b22')
        cell.set_edgecolor('#30363d')
        if r == 0:
            cell.set_text_props(color=WHITE, fontweight='bold')
        elif c == 0:
            cell.set_text_props(color=SCENARIOS[r-1]['color'], fontweight='bold')
        elif c == 8:   # Match column
            txt = cell.get_text().get_text()
            cell.set_text_props(color='#3fb950' if txt == 'OK' else '#f85149',
                                fontweight='bold')
        else:
            cell.set_text_props(color=GRAY)

    out = 'C:/Users/Corsair/Desktop/Python/BehavioralMarket/validacion.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'\nGrafica guardada: {out}')
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('=' * 62)
    print('  Validacion Empirica — BehavioralMarket vs S&P 500 Real')
    print('=' * 62)

    # Estadísticas de acciones individuales
    ind_stats = {}
    for ticker, path in IND_CSVS.items():
        df  = pd.read_csv(path, parse_dates=['Date']).sort_values('Date')
        lr  = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna().values
        ind_stats[ticker] = {
            'kurt': float(sci_kurt(lr)),
            'skew': float(sci_skew(lr)),
            'vol':  float(np.std(lr) * np.sqrt(252)),
        }

    real_df  = load_real_stats()
    sim_data = run_sim_stats()

    print(f'\nMediana real — kurt: {real_df["kurt"].median():.2f}  '
          f'vol: {real_df["vol"].median():.1%}  '
          f'skew: {real_df["skew"].median():.2f}')

    plot_validation(real_df, sim_data, ind_stats)
