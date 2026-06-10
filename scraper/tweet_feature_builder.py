"""
tweet_feature_builder.py
------------------------
MongoDB'deki analiz edilmiş tweetlerden günlük feature'lar üretir.
Çıktıyı mevcut MultiStock_Training_v2.csv ile birleştirir.

Üretilen feature'lar (günlük, hisse bazında):
  - Tweet_Sentiment_Avg   : Günlük ortalama sentiment skoru
  - Tweet_Sentiment_3d    : 3 günlük kayan sentiment ortalaması
  - Tweet_Toxic_Ratio     : Toksik tweet oranı (0.0 - 1.0)
  - Tweet_Volume          : Günlük tweet sayısı
  - Tweet_Volume_7d_Avg   : 7 günlük ortalama tweet hacmi
  - Tweet_Positive_Ratio  : Pozitif tweet oranı
  - Tweet_Negative_Ratio  : Negatif tweet oranı

Kullanım:
  python tweet_feature_builder.py
  python tweet_feature_builder.py --output Tweet_Features.csv   # Sadece feature CSV
  python tweet_feature_builder.py --merge                       # v2 CSV ile birleştir
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017/finans")
INPUT_CSV       = "MultiStock_Training_v2c.csv"
OUTPUT_FEATURES = "Tweet_Features.csv"
OUTPUT_MERGED   = "MultiStock_Training_v3.csv"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']


# ─────────────────────────────────────────────
# MONGODB'DEN VERİ ÇEK
# ─────────────────────────────────────────────

def load_tweets_from_mongo() -> pd.DataFrame:
    """
    MongoDB'deki analiz edilmiş tüm tweetleri çeker.
    is_analyzed=True olanları alır.
    """
    print("📦 MongoDB'den tweet verileri çekiliyor...")
    client     = MongoClient(MONGO_URI)
    db         = client.get_default_database()
    collection = db["tweets"]

    cursor = collection.find(
        {"is_analyzed": True},
        {
            "_id":            0,
            "symbol":         1,
            "tweet":          1,
            "timestamp":      1,
            "createdAt":      1,
            "is_toxic":       1,
            "toxicity_score": 1,
            "sentiment":      1,
            "sentiment_score":1,
        }
    )

    records = list(cursor)
    if not records:
        raise ValueError("❌ MongoDB'de analiz edilmiş tweet bulunamadı! "
                         "Önce tweet_collector.py ve tweet_analyzer.py çalıştırın.")

    df = pd.DataFrame(records)
    print(f"   ✅ {len(df)} tweet yüklendi ({df['symbol'].nunique()} hisse)")
    return df


# ─────────────────────────────────────────────
# TARİH ÇIKARMA
# ─────────────────────────────────────────────

def extract_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tweet'in tarihini YYYY-MM-DD formatına çevirir.
    Önce timestamp, yoksa createdAt kullanır.
    """
    def parse_date(row):
        # X'ten gelen ISO timestamp (2024-01-15T12:30:00.000Z)
        if pd.notna(row.get('timestamp')) and row['timestamp']:
            try:
                return pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d')
            except Exception:
                pass
        # MongoDB'nin kayıt tarihi (fallback)
        if pd.notna(row.get('createdAt')):
            try:
                return pd.to_datetime(row['createdAt']).strftime('%Y-%m-%d')
            except Exception:
                pass
        return None

    df['Date'] = df.apply(parse_date, axis=1)
    df = df.dropna(subset=['Date'])
    print(f"   📅 Tarih aralığı: {df['Date'].min()} — {df['Date'].max()}")
    return df


# ─────────────────────────────────────────────
# GÜNLÜK FEATURE HESAPLAMA
# ─────────────────────────────────────────────

def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Her (symbol, Date) çifti için günlük feature'ları hesaplar.
    """
    print("\n⚙️  Günlük tweet feature'ları hesaplanıyor...")

    all_features = []

    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy()
        sym_df = sym_df.sort_values('Date').reset_index(drop=True)

        # Günlük gruplama
        daily = sym_df.groupby('Date').agg(
            Tweet_Volume          = ('sentiment', 'count'),
            Tweet_Sentiment_Avg   = ('sentiment_score', 'mean'),
            Tweet_Toxic_Count     = ('is_toxic', 'sum'),
            Tweet_Positive_Count  = ('sentiment', lambda x: (x == 'positive').sum()),
            Tweet_Negative_Count  = ('sentiment', lambda x: (x == 'negative').sum()),
        ).reset_index()

        # Oranlar
        daily['Tweet_Toxic_Ratio']    = daily['Tweet_Toxic_Count']    / daily['Tweet_Volume']
        daily['Tweet_Positive_Ratio'] = daily['Tweet_Positive_Count'] / daily['Tweet_Volume']
        daily['Tweet_Negative_Ratio'] = daily['Tweet_Negative_Count'] / daily['Tweet_Volume']

        # Kayan ortalamalar
        daily = daily.sort_values('Date').reset_index(drop=True)
        daily['Tweet_Sentiment_3d']   = daily['Tweet_Sentiment_Avg'].rolling(3, min_periods=1).mean()
        daily['Tweet_Volume_7d_Avg']  = daily['Tweet_Volume'].rolling(7, min_periods=1).mean()

        # Gereksiz sütunları temizle
        daily.drop(['Tweet_Toxic_Count', 'Tweet_Positive_Count',
                    'Tweet_Negative_Count'], axis=1, inplace=True)

        daily['symbol'] = symbol
        all_features.append(daily)

        print(f"   [{symbol}] {len(daily)} günlük kayıt oluşturuldu.")

    result = pd.concat(all_features, ignore_index=True)

    # Sütun sırası
    cols = ['symbol', 'Date',
            'Tweet_Volume', 'Tweet_Volume_7d_Avg',
            'Tweet_Sentiment_Avg', 'Tweet_Sentiment_3d',
            'Tweet_Toxic_Ratio',
            'Tweet_Positive_Ratio', 'Tweet_Negative_Ratio']
    result = result[cols]

    print(f"\n   ✅ Toplam {len(result)} (hisse × gün) feature kaydı oluşturuldu.")
    return result


# ─────────────────────────────────────────────
# MEVCUT CSV İLE BİRLEŞTİRME
# ─────────────────────────────────────────────

def merge_with_training_data(tweet_features: pd.DataFrame) -> pd.DataFrame:
    """
    Tweet feature'larını MultiStock_Training_v2.csv ile birleştirir.
    Eşleşmeyen günler için tweet feature'ları 0.0 ile doldurulur.
    """
    print(f"\n🔗 '{INPUT_CSV}' ile birleştiriliyor...")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"❌ '{INPUT_CSV}' bulunamadı! "
            "Önce model_trainer_v2.py çalıştırın."
        )

    base_df = pd.read_csv(INPUT_CSV)
    base_df['Date'] = pd.to_datetime(base_df['Date']).dt.strftime('%Y-%m-%d')

    print(f"   Mevcut veri: {len(base_df)} satır, {base_df['symbol'].nunique()} hisse")
    print(f"   Tweet feature: {len(tweet_features)} satır")

    merged = pd.merge(
        base_df,
        tweet_features,
        on=['symbol', 'Date'],
        how='left'
    )

    # Eşleşmeyen günleri 0 ile doldur
    tweet_cols = [
        'Tweet_Volume', 'Tweet_Volume_7d_Avg',
        'Tweet_Sentiment_Avg', 'Tweet_Sentiment_3d',
        'Tweet_Toxic_Ratio',
        'Tweet_Positive_Ratio', 'Tweet_Negative_Ratio'
    ]
    for col in tweet_cols:
        merged[col] = merged[col].fillna(0.0)

    # Eşleşme istatistiği
    matched = (merged['Tweet_Volume'] > 0).sum()
    total   = len(merged)
    print(f"   Tweet eşleşmesi: {matched}/{total} gün (%{matched/total*100:.1f})")

    print(f"   ✅ Birleştirilmiş veri: {len(merged)} satır, {len(merged.columns)} sütun")
    return merged


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

def main(output_only: bool, merge: bool):
    # 1. Tweet verilerini yükle
    raw_df = load_tweets_from_mongo()

    # 2. Tarihleri çıkar
    raw_df = extract_date(raw_df)

    # 3. Günlük feature'ları hesapla
    tweet_features = build_daily_features(raw_df)

    # 4. Feature CSV'yi kaydet (her zaman)
    tweet_features.to_csv(OUTPUT_FEATURES, index=False)
    print(f"\n   💾 '{OUTPUT_FEATURES}' kaydedildi.")

    # 5. İstenirse mevcut eğitim verisiyle birleştir
    if merge or not output_only:
        try:
            merged_df = merge_with_training_data(tweet_features)
            merged_df.to_csv(OUTPUT_MERGED, index=False)
            print(f"   💾 '{OUTPUT_MERGED}' kaydedildi.")

            print(f"\n{'='*55}")
            print(f"🏆 TAMAMLANDI!")
            print(f"   → {OUTPUT_FEATURES}  (sadece tweet feature'ları)")
            print(f"   → {OUTPUT_MERGED}    (eğitime hazır birleşik veri)")
            print(f"\n💡 Sonraki adım: python model_trainer_v3.py")
            print(f"{'='*55}")

        except FileNotFoundError as e:
            print(f"\n⚠️  {e}")
            print(f"   Sadece '{OUTPUT_FEATURES}' kaydedildi.")
            print(f"   Birleştirmek için önce model_trainer_v2.py çalıştırın.")
    else:
        print(f"\n{'='*55}")
        print(f"🏆 TAMAMLANDI!")
        print(f"   → {OUTPUT_FEATURES}")
        print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tweet Feature Builder")
    parser.add_argument("--output", type=str, default=None,
                        help="Çıktı dosya adı (varsayılan: Tweet_Features.csv)")
    parser.add_argument("--merge", action="store_true",
                        help="Mevcut eğitim verisiyle birleştir")
    args = parser.parse_args()

    if args.output:
        OUTPUT_FEATURES = args.output

    main(output_only=not args.merge, merge=args.merge)