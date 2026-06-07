"""
finnhub_data_builder.py
-----------------------
Geçmiş 1 YILLIK (365 Gün) veriyi Finnhub ve Yahoo Finance'den çeker.
Haberlerin hem BAŞLIK hem de ÖZET (Summary) kısımlarını birleştirip analiz eder.
"""

import os
import time
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from transformers import pipeline
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY", "")

# Tam Kadro 10 Hissemiz
WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'JPM', 'JNJ', 'WMT', 'XOM', 'GOOGL']

# Finnhub Ücretsiz API maksimum 1 yıl geriye gitmeye izin verir.
DAYS_TO_FETCH = 365  

if not FINNHUB_KEY:
    raise ValueError("HATA: .env dosyasında FINNHUB_API_KEY bulunamadı!")

def build_premium_dataset():
    print("🚀 1. FinBERT NLP Modeli Yükleniyor...")
    # Metinler (Başlık+Özet) uzun olacağı için truncation=True çok önemli
    analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True, max_length=512)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    all_stocks_data = []

    for symbol in WATCHLIST:
        print(f"\n{'='*50}")
        print(f"⚙️ İŞLENİYOR: [{symbol}] - Son 1 Yıllık Veri Çekiliyor...")
        print(f"{'='*50}")
        
        # --- 1. FINNHUB'DAN HABERLERİ ÇEK ---
        url = "https://finnhub.io/api/v1/company-news"
        params = {
            'symbol': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'token': FINNHUB_KEY
        }
        
        print("   📰 Finnhub'dan 1 yıllık haber geçmişi indiriliyor...")
        try:
            resp = requests.get(url, params=params)
            news_data = resp.json()
            time.sleep(1.5) # API Banı yememek için 1.5 saniye dinlen
        except Exception as e:
            print(f"   ❌ Haber çekilemedi: {e}")
            continue

        if not news_data or not isinstance(news_data, list):
            print(f"   ⚠️ {symbol} için haber bulunamadı.")
            continue

        print(f"   ✅ {len(news_data)} adet haber bulundu. NLP Analizi başlıyor...")
        
        # Haberleri günlere göre grupla ve sentiment hesapla
        daily_scores = {}
        for idx, item in enumerate(news_data):
            if idx > 0 and idx % 200 == 0:
                print(f"      -> {idx} haber okundu ve puanlandı...")
                
            # KRİTİK NOKTA: Başlık ve Özeti birleştiriyoruz!
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            full_text = f"{headline}. {summary}".strip()
            
            if len(full_text) < 10: # Çok kısa/boş haberleri atla
                continue
            
            date_str = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
            
            result = analyzer(full_text)[0]
            score = result['score'] if result['label'] == 'positive' else (-result['score'] if result['label'] == 'negative' else 0.0)
            
            if date_str not in daily_scores:
                daily_scores[date_str] = []
            daily_scores[date_str].append(score)

        # Günlük ortalamaları al
        sentiment_records = []
        for d_str, scores in daily_scores.items():
            sentiment_records.append({'Date': d_str, 'Sentiment_Score': np.mean(scores)})
            
        sentiment_df = pd.DataFrame(sentiment_records)
        
        # --- 2. YFINANCE'DEN PİYASA VERİLERİNİ ÇEK ---
        print(f"   📈 {symbol} için Yahoo Finance teknik verileri çekiliyor...")
        
        start_yf = start_date - timedelta(days=30)
        raw = yf.download(symbol, start=start_yf.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if raw.empty:
            print(f"   ⚠️ {symbol} için fiyat verisi yok.")
            continue

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
            
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility_14d'] = df['Daily_Return'].rolling(14).std() * np.sqrt(252)
        
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        df['ATR_14d'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        
        vix = yf.download('^VIX', start=start_yf.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [c[0] for c in vix.columns]
        df['VIX_Close'] = vix['Close']
        
        df['Next_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        df['symbol'] = symbol
        
        # --- 3. VERİLERİ BİRLEŞTİR ---
        if sentiment_df.empty:
            merged = df.copy()
            merged['Sentiment_Score'] = 0.0
        else:
            merged = pd.merge(df, sentiment_df, on='Date', how='left')
            merged['Sentiment_Score'] = merged['Sentiment_Score'].fillna(0.0)
        
        final_cols = ['symbol', 'Date', 'Daily_Return', 'Volatility_14d', 'ATR_14d', 'VIX_Close', 'Sentiment_Score', 'Target']
        all_stocks_data.append(merged[final_cols])
        print(f"   🟢 [{symbol}] başarıyla veri setine eklendi!")

    print("\n🔗 4. Tüm Veriler Birleştiriliyor...")
    master_df = pd.concat(all_stocks_data, ignore_index=True)
    
    master_file_name = "Finnhub_Premium_Dataset.csv"
    master_df.to_csv(master_file_name, index=False)
    
    print(f"\n🏆 İŞLEM TAMAM! Yeni Premium Eğitim Seti '{master_file_name}' başarıyla yaratıldı.")
    print(f"Toplam Satır: {len(master_df)} (Her satır haberlerle çok daha dolu!)")
    
if __name__ == "__main__":
    build_premium_dataset()