"""
fix_data.py
-----------
MultiStock_Training_v3.csv icindeki eksik/NaN OHLCV (Open, High, Low,
Close, Volume) verilerini yfinance'den yeniden cekip doldurur.

Cikti: MultiStock_Training_v3_fixed.csv

Kullanim:
  python fix_data.py
"""

import time

import pandas as pd
import yfinance as yf
from datetime import timedelta

# Veri setini oku
df = pd.read_csv("MultiStock_Training_v3.csv")
df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Zaman dilimi uyusmazliklarini onle

# Tarih formatini yfinance'e uymasi icin ayarla (sadece YYYY-MM-DD)
df['Date_str'] = df['Date'].dt.strftime('%Y-%m-%d')

symbols = df['symbol'].unique()
print(f"Bozuk OHLCV verileri yfinance uzerinden {len(symbols)} hisse icin bastan cekiliyor...")

for sym in symbols:
    mask = df['symbol'] == sym
    sym_df = df[mask]

    # Ilgili hisse icin en eski ve en yeni tarihi bul
    min_date = sym_df['Date'].min()
    max_date = sym_df['Date'].max() + timedelta(days=5)  # Son gunleri kacirmamak icin 5 gun ileri

    print(f"{sym} verisi indiriliyor ({min_date.date()} - {max_date.date()})...")

    # yfinance'den veriyi cek
    ticker = yf.Ticker(sym)
    hist = ticker.history(start=min_date.strftime('%Y-%m-%d'), end=max_date.strftime('%Y-%m-%d'))

    if hist.empty:
        print(f"HATA: {sym} icin veri cekilemedi!")
        continue

    hist = hist.reset_index()

    # yfinance bazen timezone bilgisi ekler, onu temizliyoruz
    if hist['Date'].dt.tz is not None:
        hist['Date'] = hist['Date'].dt.tz_localize(None)

    hist['Date_str'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist_clean = hist[['Date_str', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    hist_indexed = hist_clean.set_index('Date_str')

    # Eski bozuk/bos kolonlari taze yfinance verisiyle eslestirerek (map) tamamen yenile
    df.loc[mask, 'Open']   = df.loc[mask, 'Date_str'].map(hist_indexed['Open'])
    df.loc[mask, 'High']   = df.loc[mask, 'Date_str'].map(hist_indexed['High'])
    df.loc[mask, 'Low']    = df.loc[mask, 'Date_str'].map(hist_indexed['Low'])
    df.loc[mask, 'Close']  = df.loc[mask, 'Date_str'].map(hist_indexed['Close'])
    df.loc[mask, 'Volume'] = df.loc[mask, 'Date_str'].map(hist_indexed['Volume'])

    # yfinance'i rate-limit'e takilmamak icin istekler arasi kisa bekleme
    time.sleep(1)

# Gecici olarak actigimiz Date_str sutununu sil
df = df.drop(columns=['Date_str'])

# Yeni ve tertemiz veriyi kaydet
df.to_csv("MultiStock_Training_v3_fixed.csv", index=False)

remaining_nan = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum()
print("\nKalan NaN sayilari:")
print(remaining_nan)
print("\nIslem tamamlandi! MultiStock_Training_v3_fixed.csv olarak kaydedildi.")
