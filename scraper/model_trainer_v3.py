"""
model_trainer_v3.py
-------------------
Tweet feature'larını ekleyerek birleşik model eğitir.

Yenilikler (v2'ye göre):
  - Tweet_Sentiment_Avg   : Günlük X sentiment ortalaması
  - Tweet_Sentiment_3d    : 3 günlük kayan X sentiment
  - Tweet_Toxic_Ratio     : Toksik tweet oranı
  - Tweet_Volume_7d_Avg   : 7 günlük tweet hacim ortalaması
  - Tweet_Positive_Ratio  : Pozitif tweet oranı
  - Tweet_Negative_Ratio  : Negatif tweet oranı

Kullanım:
  python model_trainer_v3.py
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost kurulu değil, RF ve GB kullanılacak.")

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

INPUT_CSV   = "MultiStock_Training_v3.csv"
MODEL_FILE  = "model_v3.pkl"
REPORT_FILE = "model_report_v3.txt"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

# v2'deki tüm feature'lar + yeni tweet feature'ları
FEATURES = [
    # Teknik göstergeler (v2'den)
    'Daily_Return',
    'Volatility_14d',
    'ATR_14d',
    'VIX_Close',
    'RSI_14',
    'MA20_ratio',
    'Momentum_5d',
    # Haber sentiment (v2'den)
    'Sentiment_Decay',
    'Sentiment_3d_avg',
    'News_count_7d',
    # Tweet feature'ları (YENİ)
    'Tweet_Sentiment_Avg',
    'Tweet_Sentiment_3d',
    'Tweet_Toxic_Ratio',
    'Tweet_Volume_7d_Avg',
    'Tweet_Positive_Ratio',
    'Tweet_Negative_Ratio',
]


# ─────────────────────────────────────────────
# ADIM 1 — VERİ YÜKLE
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    print(f"📂 1. '{csv_path}' yükleniyor...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌ '{csv_path}' bulunamadı!\n"
            "Önce şu sırayla çalıştırın:\n"
            "  1. python tweet_collector.py\n"
            "  2. python tweet_analyzer.py\n"
            "  3. python tweet_feature_builder.py --merge\n"
        )

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)

    print(f"   ✅ {len(df)} satır, {df['symbol'].nunique()} hisse yüklendi.")

    # Tweet verisi olan günlerin oranı
    tweet_days = (df['Tweet_Volume_7d_Avg'] > 0).sum()
    print(f"   📊 Tweet verisi olan günler: {tweet_days}/{len(df)} (%{tweet_days/len(df)*100:.1f})")

    # Eksik feature kontrolü
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"❌ Eksik sütunlar: {missing}\n"
                         "tweet_feature_builder.py --merge çalıştırın.")

    return df


# ─────────────────────────────────────────────
# ADIM 2 — MODEL EĞİTİMİ
# ─────────────────────────────────────────────

def train_and_compare(df: pd.DataFrame):
    print("\n🤖 2. Modeller eğitiliyor...")

    X = df[FEATURES].fillna(0.0)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"   Eğitim: {len(X_train)} | Test: {len(X_test)}")

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    results   = {}
    best_acc  = 0
    best_name = None
    best_model= None

    for name, model in models.items():
        print(f"\n   [{name}] eğitiliyor...")
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        acc       = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        results[name] = {
            'model':    model,
            'accuracy': acc,
            'cv_mean':  cv_scores.mean(),
            'cv_std':   cv_scores.std(),
            'report':   classification_report(y_test, y_pred,
                            target_names=['Düşüş', 'Yükseliş']),
            'cm':       confusion_matrix(y_test, y_pred),
        }

        print(f"   → Test Accuracy : %{acc*100:.2f}")
        print(f"   → CV Accuracy   : %{cv_scores.mean()*100:.2f} ± %{cv_scores.std()*100:.2f}")

        if acc > best_acc:
            best_acc   = acc
            best_name  = name
            best_model = model

    return results, best_name, best_model, X_test, y_test


# ─────────────────────────────────────────────
# ADIM 3 — RAPOR
# ─────────────────────────────────────────────

def print_and_save_report(results, best_name, best_model):
    print(f"\n{'='*55}")
    print(f"  EN İYİ MODEL: {best_name}")
    print(f"{'='*55}")

    lines = [f"En iyi model: {best_name}\n"]

    for name, r in results.items():
        marker = " ★ EN İYİ" if name == best_name else ""
        block  = (
            f"\n{'─'*50}\n"
            f"Model: {name}{marker}\n"
            f"Test Accuracy : %{r['accuracy']*100:.2f}\n"
            f"CV  Accuracy  : %{r['cv_mean']*100:.2f} ± %{r['cv_std']*100:.2f}\n\n"
            f"{r['report']}\n"
            f"Confusion Matrix:\n{r['cm']}\n"
        )
        print(block)
        lines.append(block)

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = sorted(
            zip(FEATURES, best_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        imp_block = "\nÖzellik Önemleri (En İyi Model):\n"
        for feat, imp in importances:
            bar = '█' * int(imp * 50)
            imp_block += f"  {feat:<25} {bar} {imp:.4f}\n"
        print(imp_block)
        lines.append(imp_block)

        # Tweet feature'larının toplam katkısı
        tweet_feats = [f for f in FEATURES if f.startswith('Tweet_')]
        tweet_importance = sum(
            imp for feat, imp in importances
            if feat in tweet_feats
        )
        summary = (
            f"\nTweet feature'larının toplam katkısı: %{tweet_importance*100:.1f}\n"
            f"Teknik+Haber feature'ların katkısı  : %{(1-tweet_importance)*100:.1f}\n"
        )
        print(summary)
        lines.append(summary)

    lines.append(f"\nEğitim hisseleri: {', '.join(WATCHLIST)}")
    lines.append(f"\nKullanılan feature'lar ({len(FEATURES)} adet):\n  " +
                 "\n  ".join(FEATURES))
    lines.append("\nYenilikler (v3):\n"
                 "  - Tweet_Sentiment_Avg\n"
                 "  - Tweet_Sentiment_3d\n"
                 "  - Tweet_Toxic_Ratio\n"
                 "  - Tweet_Volume_7d_Avg\n"
                 "  - Tweet_Positive_Ratio\n"
                 "  - Tweet_Negative_Ratio")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"   📄 Rapor '{REPORT_FILE}' kaydedildi.")


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

def main():
    print("🚀 Model Trainer v3 Başlıyor")
    print("   Yenilik: Tweet sentiment + toksiklik + hacim feature'ları\n")

    # 1. Veri yükle
    df = load_data(INPUT_CSV)

    # 2. Modelleri eğit
    results, best_name, best_model, X_test, y_test = train_and_compare(df)

    # 3. Rapor
    print_and_save_report(results, best_name, best_model)

    # 4. Modeli kaydet
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model':      best_model,
            'features':   FEATURES,
            'model_name': f"{best_name} v3"
        }, f)
    print(f"   💾 Model '{MODEL_FILE}' kaydedildi.")

    print(f"\n✅ TAMAMLANDI!")
    print(f"   → {MODEL_FILE}")
    print(f"   → {REPORT_FILE}")
    print(f"\n💡 ml_service.py'de MODEL_PATH = '{MODEL_FILE}' yap ve yeniden başlat!")


if __name__ == "__main__":
    main()