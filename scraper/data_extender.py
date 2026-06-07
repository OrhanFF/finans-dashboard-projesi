"""
data_extender.py
----------------
Mevcut MultiStock_Training.csv'yi 2023-2026 verileriyle genisletir.
Finnhub haberleri + yfinance fiyat verisi ceker.
Sadece teknik feature'lari ekler (FinBERT cok yavas, batch yaparız).

Cıktı: MultiStock_Training_extended.csv
"""

import os
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")
WATCHLIST   = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
                'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

EXISTING_CSV = "MultiStock_Training.csv"
OUTPUT_CSV   = "MultiStock_Training_extended.csv"

# Cekecegimiz aralik: mevcut verinin bittigi yerden bugun
# (eski veri ~2020-2022, yeni veri 2023-bugun)
FETCH_START = "2023-01-01"
FETCH_END   = datetime.now().strftime('%Y-%m-%d')


def fetch_news_sentiment_batch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Finnhub'dan gunluk sentiment skorlari ceker.
    FinBERT yerine basit kural tabanli: pozitif/negatif kelime sayımı.
    (Hız için — FinBERT ayrı script'te çalıştırılabilir)
    """
    if not FINNHUB_KEY:
        return pd.DataFrame(columns=['Date', 'Sentiment_Score', 'News_Count'])

    # Maks 1 yil aralik destekleniyor, parcalara böl
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt   = datetime.strptime(end,   '%Y-%m-%d')

    all_news = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=180), end_dt)
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    'symbol': symbol,
                    'from':   chunk_start.strftime('%Y-%m-%d'),
                    'to':     chunk_end.strftime('%Y-%m-%d'),
                    'token':  FINNHUB_KEY
                },
                timeout=10
            )
            chunk_news = resp.json()
            if isinstance(chunk_news, list):
                all_news.extend(chunk_news)
            time.sleep(0.5)
        except Exception as e:
            print(f"   Haber hatasi: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if not all_news:
        return pd.DataFrame(columns=['Date', 'Sentiment_Score', 'News_Count'])

    # FinBERT kullan (hafif mod - pipeline yuklu ise)
    try:
        from transformers import pipeline as hf_pipeline
        print(f"   FinBERT yukleniyor... ({len(all_news)} haber)")
        analyzer = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512
        )

        # Gunlük grupla
        daily = {}
        batch_texts = []
        batch_dates = []

        for item in all_news:
            headline = item.get('headline', '')
            summary  = item.get('summary', '')
            text = f"{headline}. {summary}".strip()
            if len(text) < 10:
                continue
            date_str = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
            batch_texts.append(text[:500])
            batch_dates.append(date_str)

        if batch_texts:
            results = analyzer(batch_texts, batch_size=16)
            for date_str, res in zip(batch_dates, results):
                score = res['score'] if res['label'] == 'positive' else (
                        -res['score'] if res['label'] == 'negative' else 0.0)
                if date_str not in daily:
                    daily[date_str] = []
                daily[date_str].append(score)

        records = [{'Date': d, 'Sentiment_Score': np.mean(s), 'News_Count': len(s)}
                   for d, s in daily.items()]
        return pd.DataFrame(records)

    except Exception as e:
        print(f"   FinBERT hatasi, 0 ile dolduruluyor: {e}")
        return pd.DataFrame(columns=['Date', 'Sentiment_Score', 'News_Count'])


def fetch_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """yfinance'den fiyat + teknik feature'lar."""
    # RSI/MA hesabi icin 90 gun oncesinden basla
    fetch_start = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=90)).strftime('%Y-%m-%d')

    raw = yf.download(symbol, start=fetch_start, end=end, progress=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()

    df['Daily_Return']   = close.pct_change()
    df['Volatility_14d'] = df['Daily_Return'].rolling(14).std() * np.sqrt(252)

    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    df['ATR_14d'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # VIX
    vix_raw = yf.download('^VIX', start=fetch_start, end=end, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = [c[0] for c in vix_raw.columns]
    df['VIX_Close'] = vix_raw['Close'] if not vix_raw.empty else 20.0

    # Target
    df['Next_Close'] = close.shift(-1)
    df['Target']     = (df['Next_Close'] > close).astype(int)

    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['Date']   = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['symbol'] = symbol

    # Sadece istenen tarih araligini don (warm-up kesiliyor)
    df = df[df['Date'] >= start].reset_index(drop=True)

    cols = ['symbol', 'Date', 'Daily_Return', 'Volatility_14d', 'ATR_14d', 'VIX_Close', 'Target']
    return df[cols]


def main():
    print("Veri Genisletici Basliyor")
    print(f"   Aralik: {FETCH_START} --> {FETCH_END}\n")

    # Mevcut veriyi yukle
    if os.path.exists(EXISTING_CSV):
        existing = pd.read_csv(EXISTING_CSV)
        print(f"Mevcut veri: {len(existing)} satir")
        print(f"   Tarih araligi: {existing['Date'].min()} - {existing['Date'].max()}")
        print(f"   VIX ortalama : {existing['VIX_Close'].mean():.1f}")
    else:
        existing = pd.DataFrame()
        print("Mevcut veri yok, sifirdan baslanıyor.")

    new_frames = []

    for symbol in WATCHLIST:
        print(f"\n{'='*50}")
        print(f"[{symbol}] isleniyor...")
        print(f"{'='*50}")

        # 1. Fiyat verisi
        price_df = fetch_price_data(symbol, FETCH_START, FETCH_END)
        if price_df.empty:
            print(f"   [{symbol}] fiyat verisi alinamadi, atlaniyor.")
            continue
        print(f"   [{symbol}] {len(price_df)} gun fiyat verisi alindi.")
        print(f"   VIX aralik: {price_df['VIX_Close'].min():.1f} - {price_df['VIX_Close'].max():.1f}")

        # 2. Haber sentiment
        print(f"   [{symbol}] Haber sentiment hesaplaniyor...")
        sent_df = fetch_news_sentiment_batch(symbol, FETCH_START, FETCH_END)

        if not sent_df.empty:
            merged = pd.merge(price_df, sent_df[['Date', 'Sentiment_Score']],
                              on='Date', how='left')
            merged['Sentiment_Score'] = merged['Sentiment_Score'].fillna(0.0)
            print(f"   [{symbol}] {(merged['Sentiment_Score'] != 0).sum()} haberli gun eslesti.")
        else:
            merged = price_df.copy()
            merged['Sentiment_Score'] = 0.0

        new_frames.append(merged)
        print(f"   [{symbol}] tamamlandi.")
        time.sleep(1)

    if not new_frames:
        print("Yeni veri alinamadi!")
        return

    new_df = pd.concat(new_frames, ignore_index=True)
    print(f"\nYeni veri: {len(new_df)} satir")
    print(f"   VIX ortalama: {new_df['VIX_Close'].mean():.1f}")
    print(f"   VIX min-max : {new_df['VIX_Close'].min():.1f} - {new_df['VIX_Close'].max():.1f}")

    # Birlestir
    if not existing.empty:
        # Mevcut veride olmayan tarihleri ekle (duplicate onleme)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['symbol', 'Date'], keep='last')
        combined = combined.sort_values(['symbol', 'Date']).reset_index(drop=True)
    else:
        combined = new_df

    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*55}")
    print(f"TAMAMLANDI!")
    print(f"   Toplam satir : {len(combined)}")
    print(f"   Tarih araligi: {combined['Date'].min()} - {combined['Date'].max()}")
    print(f"   VIX ortalama : {combined['VIX_Close'].mean():.1f}")
    print(f"   Dusuk VIX (<20) gun sayisi: {(combined['VIX_Close'] < 20).sum()}")
    print(f"   Dosya       : {OUTPUT_CSV}")
    print(f"\nSonraki adim: python model_trainer_v2.py (CSV_FILE = '{OUTPUT_CSV}')")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
