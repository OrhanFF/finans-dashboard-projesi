"""
premium_dataset_builder.py
--------------------------
Kaggle CSV + Finnhub 1 yıllık haberleri birleştirir.
Her hisse için dengeli haber sayısı kullanır (JPM dengesizliği düzeltildi).
FinBERT ile sentiment analizi yapar (batch + GPU destekli).
Yeni model.pkl otomatik eğitilir.

Kullanım:
  python premium_dataset_builder.py
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
from transformers import pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

FINNHUB_KEY    = os.getenv("FINNHUB_API_KEY", "")
KAGGLE_CSV     = "raw_partner_headlines.csv"
OUTPUT_CSV     = "Premium_Dataset.csv"
CHECKPOINT_CSV = "checkpoint_sentiment.csv"
MODEL_FILE     = "model.pkl"
REPORT_FILE    = "model_report.txt"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

# Her hisseden alınacak maksimum haber sayısı
MAX_NEWS_PER_STOCK = 300

FEATURES   = ['Daily_Return', 'Volatility_14d', 'ATR_14d', 'VIX_Close', 'Sentiment_Score']
BATCH_SIZE = 64


# ─────────────────────────────────────────────
# ADIM 1a — KAGGLE'DAN DENGELİ HABER YÜKLE
# ─────────────────────────────────────────────

def load_kaggle_news(csv_path: str, watchlist: list) -> pd.DataFrame:
    print("📂 1a. Kaggle CSV yükleniyor...")

    if not os.path.exists(csv_path):
        print(f"   ⚠️  '{csv_path}' bulunamadı, Kaggle verisi atlanıyor.")
        return pd.DataFrame(columns=['Date', 'stock', 'text'])

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df[df['stock'].isin(watchlist)].copy()
    df['Date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df['text'] = df['headline'].astype(str)
    df = df[['Date', 'stock', 'text']].dropna()

    # Ham dağılımı göster
    print("   Kaggle ham dağılım:")
    for sym, cnt in df['stock'].value_counts().items():
        bar = '█' * min(int(cnt / 50), 40)
        print(f"   {sym:<6} {bar} {cnt}")

    # ── KRİTİK DÜZELTME: Her hisseden eşit miktarda al ──
    # JPM gibi outlier hisselerin modeli domine etmesini engelle
    balanced = (
        df.sort_values('Date', ascending=False)
          .groupby('stock')
          .head(MAX_NEWS_PER_STOCK)
          .reset_index(drop=True)
    )

    print(f"\n   Dengelenmiş dağılım (max {MAX_NEWS_PER_STOCK}/hisse):")
    for sym, cnt in balanced['stock'].value_counts().items():
        bar = '█' * min(int(cnt / 10), 40)
        print(f"   {sym:<6} {bar} {cnt}")

    print(f"\n   ✅ Kaggle: {len(balanced)} haber "
          f"({balanced['stock'].nunique()} hisse, "
          f"{balanced['Date'].min()} – {balanced['Date'].max()})")
    return balanced


# ─────────────────────────────────────────────
# ADIM 1b — FINNHUB'DAN SON 1 YILLIK HABERLER
# ─────────────────────────────────────────────

def load_finnhub_news(watchlist: list) -> pd.DataFrame:
    print("\n📂 1b. Finnhub'dan son 1 yıllık haberler çekiliyor...")

    if not FINNHUB_KEY:
        print("   ⚠️  FINNHUB_API_KEY bulunamadı.")
        return pd.DataFrame(columns=['Date', 'stock', 'text'])

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=365)
    all_news   = []

    for symbol in watchlist:
        print(f"   [{symbol}] çekiliyor...", end=" ")
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    'symbol': symbol,
                    'from':   start_date.strftime('%Y-%m-%d'),
                    'to':     end_date.strftime('%Y-%m-%d'),
                    'token':  FINNHUB_KEY
                },
                timeout=15
            )
            news_data = resp.json()
            time.sleep(1.2)

            count = 0
            for item in news_data:
                headline  = item.get('headline', '')
                summary   = item.get('summary', '')
                full_text = f"{headline}. {summary}".strip()
                if len(full_text) < 15:
                    continue
                date_str = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
                all_news.append({'Date': date_str, 'stock': symbol, 'text': full_text})
                count += 1

            print(f"{count} haber")
        except Exception as e:
            print(f"HATA: {e}")

    df = pd.DataFrame(all_news) if all_news else pd.DataFrame(columns=['Date', 'stock', 'text'])
    if not df.empty:
        print(f"   ✅ Finnhub: {len(df)} haber toplam")
    return df


# ─────────────────────────────────────────────
# ADIM 2 — BİRLEŞTİR + TEMİZLE
# ─────────────────────────────────────────────

def merge_sources(kaggle_df: pd.DataFrame, finnhub_df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔗 2. Kaynaklar birleştiriliyor...")

    combined = pd.concat([kaggle_df, finnhub_df], ignore_index=True)
    before   = len(combined)
    combined.drop_duplicates(subset=['stock', 'Date', 'text'], inplace=True)
    after    = len(combined)

    print(f"   {before} → {after} haber (duplicate temizlendi)")
    print(f"   Tarih: {combined['Date'].min()} – {combined['Date'].max()}")
    print(f"\n   Final dağılım:")
    for sym, cnt in combined['stock'].value_counts().items():
        bar = '█' * min(int(cnt / 10), 40)
        print(f"   {sym:<6} {bar} {cnt}")

    return combined


# ─────────────────────────────────────────────
# ADIM 3 — FINBERT BATCH SENTIMENT
# ─────────────────────────────────────────────

def run_finbert_batch(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n🧠 3. FinBERT Sentiment Analizi ({len(df)} haber)...")

    # Checkpoint kontrolü
    if os.path.exists(CHECKPOINT_CSV):
        print(f"   ♻️  Checkpoint bulundu, kaldığı yerden devam ediliyor.")
        checkpoint = pd.read_csv(CHECKPOINT_CSV)
        done_texts = set(checkpoint['text'].tolist())
        remaining  = df[~df['text'].isin(done_texts)].copy()
        done_df    = checkpoint
        print(f"   {len(done_df)} haber işlenmiş, {len(remaining)} kaldı.")
    else:
        remaining = df.copy()
        done_df   = pd.DataFrame()

    if remaining.empty:
        print("   ✅ Tüm haberler zaten işlenmiş.")
        result = pd.concat([done_df, pd.DataFrame()], ignore_index=True)
        return result

    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1

    print(f"   Cihaz: {'GPU 🚀' if device == 0 else 'CPU 🐢'}")

    analyzer = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=device,
        truncation=True,
        max_length=512
    )

    texts  = remaining['text'].astype(str).tolist()
    total  = len(texts)
    scores = []

    for i in range(0, total, BATCH_SIZE):
        batch   = texts[i:i + BATCH_SIZE]
        outputs = analyzer(batch)

        for r in outputs:
            if r['label'] == 'positive':
                scores.append(r['score'])
            elif r['label'] == 'negative':
                scores.append(-r['score'])
            else:
                scores.append(0.0)

        done = min(i + BATCH_SIZE, total)
        print(f"   → {done}/{total} ({done/total*100:.1f}%)", end='\r')

        # Her 300 haberde checkpoint kaydet
        if done % 300 == 0:
            temp = remaining.iloc[:len(scores)].copy()
            temp['Sentiment_Score'] = scores
            pd.concat([done_df, temp]).to_csv(CHECKPOINT_CSV, index=False)

    print(f"\n   ✅ Sentiment analizi tamamlandı.")
    remaining['Sentiment_Score'] = scores

    if os.path.exists(CHECKPOINT_CSV):
        os.remove(CHECKPOINT_CSV)

    if not done_df.empty:
        return pd.concat([done_df, remaining], ignore_index=True)
    return remaining


# ─────────────────────────────────────────────
# ADIM 4 — GÜNLÜK SENTIMENT ORTALAMALARI
# ─────────────────────────────────────────────

def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📅 4. Günlük sentiment ortalamaları hesaplanıyor...")
    daily = (
        df.groupby(['stock', 'Date'])['Sentiment_Score']
          .mean()
          .reset_index()
    )
    print(f"   ✅ {len(daily)} (hisse × gün) sentiment kaydı")
    return daily


# ─────────────────────────────────────────────
# ADIM 5 — FİYAT + TEKNİK İNDİKATÖRLER
# ─────────────────────────────────────────────

def fetch_stock_features(symbol: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(symbol, start=start, end=end, progress=False)
    if raw.empty or len(raw) < 15:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw[['High', 'Low', 'Close', 'Volume']].copy()
    df['Daily_Return']   = df['Close'].pct_change()
    df['Volatility_14d'] = df['Daily_Return'].rolling(14).std() * np.sqrt(252)

    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift()).abs()
    lc  = (df['Low']  - df['Close'].shift()).abs()
    df['ATR_14d'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    vix = yf.download('^VIX', start=start, end=end, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]
    df['VIX_Close'] = vix['Close']

    df['Next_Close'] = df['Close'].shift(-1)
    df['Target']     = (df['Next_Close'] > df['Close']).astype(int)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['Date']   = df['Date'].dt.strftime('%Y-%m-%d')
    df['symbol'] = symbol

    return df[['symbol', 'Date', 'Daily_Return', 'Volatility_14d', 'ATR_14d', 'VIX_Close', 'Target']]


def fetch_all_stocks(watchlist: list, start: str, end: str) -> pd.DataFrame:
    print("\n📈 5. Hisse fiyat + teknik veriler çekiliyor...")
    frames = []
    for sym in watchlist:
        print(f"   [{sym}]...", end=" ")
        try:
            f = fetch_stock_features(sym, start, end)
            if not f.empty:
                frames.append(f)
                print(f"{len(f)} gün")
            else:
                print("veri yok")
        except Exception as e:
            print(f"HATA: {e}")
    combined = pd.concat(frames, ignore_index=True)
    print(f"   ✅ Toplam {len(combined)} satır")
    return combined


# ─────────────────────────────────────────────
# ADIM 6 — BÜYÜK BİRLEŞTİRME
# ─────────────────────────────────────────────

def merge_all(stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔗 6. Fiyat + Sentiment birleştiriliyor...")

    merged = pd.merge(
        stock_df,
        sentiment_df,
        left_on=['symbol', 'Date'],
        right_on=['stock', 'Date'],
        how='left'
    ).drop(columns=['stock'], errors='ignore')

    merged['Sentiment_Score'] = merged['Sentiment_Score'].fillna(0.0)

    with_news = (merged['Sentiment_Score'] != 0.0).sum()
    total     = len(merged)

    print(f"   ✅ {total} satır, {merged['symbol'].nunique()} hisse")
    print(f"   📰 Haberli günler: {with_news} ({with_news/total*100:.1f}%)")
    print(f"   📭 Habersiz günler: {total-with_news} ({(total-with_news)/total*100:.1f}%)")

    return merged


# ─────────────────────────────────────────────
# ADIM 7 — MODEL EĞİTİMİ
# ─────────────────────────────────────────────

def train_and_save_model(df: pd.DataFrame):
    print(f"\n🤖 7. Random Forest modeli eğitiliyor...")

    X = df[FEATURES]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Düşüş', 'Yükseliş'])
    cm     = confusion_matrix(y_test, y_pred)

    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )

    print(f"\n{'='*50}")
    print(f"  YENİ MODEL SONUÇLARI")
    print(f"{'='*50}")
    print(f"  Test Accuracy : %{acc*100:.2f}")
    print(f"  Eğitim Seti   : {len(X_train)} gün")
    print(f"  Test Seti     : {len(X_test)} gün")
    print(f"\n{report}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"\n  Özellik Önemleri:")
    for feat, imp in importances:
        bar = '█' * int(imp * 40)
        print(f"    {feat:<20} {bar} {imp:.3f}")
    print(f"{'='*50}\n")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Test Accuracy: %{acc*100:.2f}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n\n")
        f.write("Özellik Önemleri:\n")
        for feat, imp in importances:
            f.write(f"  {feat}: {imp:.4f}\n")
        f.write(f"\nEğitim hisseleri: {', '.join(WATCHLIST)}\n")
        f.write(f"Haberli günler: {(df['Sentiment_Score'] != 0.0).sum()}/{len(df)}\n")

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'features': FEATURES}, f)

    print(f"   💾 '{MODEL_FILE}' güncellendi.")
    print(f"   📄 '{REPORT_FILE}' güncellendi.")
    return acc


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

def main():
    print("🚀 Premium Dataset Builder (Dengeli) Başlıyor...\n")
    start_time = datetime.now()

    kaggle_df   = load_kaggle_news(KAGGLE_CSV, WATCHLIST)
    finnhub_df  = load_finnhub_news(WATCHLIST)
    combined_df = merge_sources(kaggle_df, finnhub_df)

    if combined_df.empty:
        print("❌ Hiç haber verisi yok!")
        return

    combined_df     = run_finbert_batch(combined_df)
    sentiment_daily = aggregate_daily_sentiment(combined_df)

    start_str      = combined_df['Date'].min()
    end_str        = combined_df['Date'].max()
    start_extended = (pd.to_datetime(start_str) - pd.Timedelta(days=30)).strftime('%Y-%m-%d')

    stock_df = fetch_all_stocks(WATCHLIST, start_extended, end_str)
    final_df = merge_all(stock_df, sentiment_daily)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n   💾 '{OUTPUT_CSV}' kaydedildi ({len(final_df)} satır).")

    acc = train_and_save_model(final_df)

    elapsed = datetime.now() - start_time
    print(f"\n✅ TAMAMLANDI! ({elapsed.seconds // 60}dk {elapsed.seconds % 60}sn)")
    print(f"   → {OUTPUT_CSV}")
    print(f"   → {MODEL_FILE} (%{acc*100:.2f})")
    print(f"\n💡 ml_service.py'yi yeniden başlat:")
    print(f"   uvicorn ml_service:app --host 0.0.0.0 --port 8002 --reload")


if __name__ == "__main__":
    main()