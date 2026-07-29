import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

# Stocks individuales reales (Yahoo Finance)
tickers = {
    'AAPL': r'C:\Users\Corsair\Desktop\Python\Datos\AAPL.csv',
    'TSLA': r'C:\Users\Corsair\Desktop\Python\Datos\TSLA.csv',
    'PLTR': r'C:\Users\Corsair\Desktop\Python\Datos\PLTR.csv',
}

print('=== Stocks individuales (Yahoo Finance) ===')
for name, path in tickers.items():
    df = pd.read_csv(path, parse_dates=['Date']).sort_values('Date')
    lr = np.log(df['Adj Close'] / df['Adj Close'].shift(1)).dropna()
    print(f'\n{name}:')
    print(f'  Periodo : {df["Date"].min().date()} -> {df["Date"].max().date()}  ({len(df)} dias)')
    print(f'  Kurtosis: {kurtosis(lr):+.2f}   (mercados reales: 4-8)')
    print(f'  Skewness: {skew(lr):+.2f}')
    print(f'  Vol anual: {lr.std() * (252**0.5) * 100:.1f}%')
    print(f'  Max DD  : {((df["Adj Close"] / df["Adj Close"].cummax()) - 1).min() * 100:.1f}%')

# Dataset grande
print('\n=== stock_details_5_years.csv ===')
big = pd.read_csv(r'C:\Users\Corsair\Desktop\Python\Datos\stock_details_5_years.csv',
                  parse_dates=['Date'])
print(f'Shape   : {big.shape}')
print(f'Periodo : {big["Date"].min().date()} -> {big["Date"].max().date()}')
print(f'Empresas: {big["Company"].nunique()}  ({big["Company"].unique()[:8].tolist()} ...)')
