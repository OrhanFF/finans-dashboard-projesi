"""
model_trainer_v2.py
-------------------
Yenilikler:
  1. Sentiment Decay  — haberin etkisi 3 gün boyunca azalarak devam eder
  2. Yeni teknik feature'lar — RSI_14, MA20_ratio, Momentum_5d
  3. Yeni sentiment feature'lar — Sentiment_3d_avg, News_count_7d
  4. XGBoost + Random Forest karşılaştırması — en iyi model kaydedilir

Kullanım:
  python model_trainer_v2.py
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost kurulu değil, sadece Random Forest kullanılacak.")
    print("   Kurmak için: pip install xgboost")

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

CSV_FILE    = "MultiStock_Training.csv"   # Mevcut eğitim verisi
OUTPUT_CSV  = "MultiStock_Training_v2.csv"
MODEL_FILE  = "model_v2.pkl"
REPORT_FILE = "model_report_v2.txt"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

# Sentiment decay katsayıları (gün 0 = haber günü)
DECAY_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}

FEATURES = [
    # Orijinal feature'lar
    'Daily_Return',
    'Volatility_14d',
    'ATR_14d',
    'VIX_Close',
    # Yeni teknik feature'lar
    'RSI_14',
    'MA20_ratio',
    'Momentum_5d',
    # Yeni sentiment feature'lar
    'Sentiment_Decay',     # Ana yenilik: decay uygulanmış sentiment
    'Sentiment_3d_avg',    # 3 günlük kayan ortalama
    'News_count_7d',       # Son 7 günde kaç haber çıktı
]


# ─────────────────────────────────────────────
# ADIM 1 — CSV Yükle
# ─────────────────────────────────────────────

def load_base_data(csv_path: str) -> pd.DataFrame:
    print("📂 1. Mevcut eğitim verisi yükleniyor...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)
    print(f"   ✅ {len(df)} satır, {df['symbol'].nunique()} hisse yüklendi.")
    haberli = (df['Sentiment_Score'] != 0).sum()
    print(f"   📰 Haberli günler: {haberli} / {len(df)} (%{haberli/len(df)*100:.1f})")
    return df


# ─────────────────────────────────────────────
# ADIM 2 — Sentiment Decay
# ─────────────────────────────────────────────

def apply_sentiment_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Haber olan günden itibaren etkiyi 3 gün boyunca azaltarak yayar.
    Gün 0: skor × 1.0
    Gün 1: skor × 0.7
    Gün 2: skor × 0.4
    Gün 3: skor × 0.1
    """
    print("\n📉 2. Sentiment Decay uygulanıyor...")
    
    df = df.copy()
    df['Sentiment_Decay'] = 0.0

    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        sym_df = df[mask].copy()
        decay_col = np.zeros(len(sym_df))

        for idx in range(len(sym_df)):
            raw_score = sym_df['Sentiment_Score'].iloc[idx]
            if raw_score != 0.0:
                for day_offset, weight in DECAY_WEIGHTS.items():
                    target_idx = idx + day_offset
                    if target_idx < len(sym_df):
                        # Daha güçlü bir sinyal varsa üzerine yazma
                        candidate = raw_score * weight
                        if abs(candidate) > abs(decay_col[target_idx]):
                            decay_col[target_idx] = candidate

        df.loc[mask, 'Sentiment_Decay'] = decay_col

    haberli_oncesi = (df['Sentiment_Score'] != 0).sum()
    haberli_sonrasi = (df['Sentiment_Decay'] != 0).sum()
    print(f"   Decay öncesi haberli gün: {haberli_oncesi}")
    print(f"   Decay sonrası dolu gün  : {haberli_sonrasi} (%{haberli_sonrasi/len(df)*100:.1f})")
    return df


# ─────────────────────────────────────────────
# ADIM 3 — Yeni Feature'lar
# ─────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚙️  3. Yeni teknik ve sentiment feature'lar hesaplanıyor...")
    
    result_frames = []

    for symbol in df['symbol'].unique():
        sym_df = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        close = sym_df['Close']

        # ── RSI_14 ──────────────────────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        sym_df['RSI_14'] = 100 - (100 / (1 + rs))

        # ── MA20_ratio ──────────────────────────────────────────────
        # Fiyatın 20 günlük ortalamaya göre yüzde sapması
        ma20 = close.rolling(20, min_periods=5).mean()
        sym_df['MA20_ratio'] = (close - ma20) / ma20

        # ── Momentum_5d ─────────────────────────────────────────────
        # 5 günlük fiyat değişim yüzdesi
        sym_df['Momentum_5d'] = close.pct_change(5)

        # ── Sentiment_3d_avg ────────────────────────────────────────
        sym_df['Sentiment_3d_avg'] = (
            sym_df['Sentiment_Decay'].rolling(3, min_periods=1).mean()
        )

        # ── News_count_7d ────────────────────────────────────────────
        # Son 7 günde haber çıkan gün sayısı (0 olmayan Sentiment_Score)
        has_news = (sym_df['Sentiment_Score'] != 0).astype(int)
        sym_df['News_count_7d'] = has_news.rolling(7, min_periods=1).sum()

        result_frames.append(sym_df)

    df_new = pd.concat(result_frames, ignore_index=True)
    df_new.dropna(subset=FEATURES, inplace=True)

    print(f"   ✅ Feature engineering tamamlandı. Kalan satır: {len(df_new)}")
    return df_new


# ─────────────────────────────────────────────
# ADIM 4 — Model Eğitimi
# ─────────────────────────────────────────────

def train_and_compare(df: pd.DataFrame):
    print("\n🤖 4. Modeller eğitiliyor ve karşılaştırılıyor...")

    X = df[FEATURES]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"   Eğitim: {len(X_train)} gün | Test: {len(X_test)} gün")

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
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    results = {}
    best_acc = 0
    best_name = None
    best_model = None

    for name, model in models.items():
        print(f"\n   [{name}] eğitiliyor...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'report': classification_report(y_test, y_pred,
                        target_names=['Düşüş', 'Yükseliş']),
            'cm': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred
        }

        print(f"   → Test Accuracy : %{acc*100:.2f}")
        print(f"   → CV Accuracy   : %{cv_scores.mean()*100:.2f} ± %{cv_scores.std()*100:.2f}")

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    return results, best_name, best_model, X_test, y_test


# ─────────────────────────────────────────────
# ADIM 5 — Rapor
# ─────────────────────────────────────────────

def print_and_save_report(results, best_name, best_model, df):
    print(f"\n{'='*55}")
    print(f"  SONUÇLAR — EN İYİ MODEL: {best_name}")
    print(f"{'='*55}")

    lines = []
    lines.append(f"En iyi model: {best_name}\n")

    for name, r in results.items():
        marker = " ★ EN İYİ" if name == best_name else ""
        block = (
            f"\n{'─'*50}\n"
            f"Model: {name}{marker}\n"
            f"Test Accuracy : %{r['accuracy']*100:.2f}\n"
            f"CV  Accuracy  : %{r['cv_mean']*100:.2f} ± %{r['cv_std']*100:.2f}\n\n"
            f"{r['report']}\n"
            f"Confusion Matrix:\n{r['cm']}\n"
        )
        print(block)
        lines.append(block)

    # Feature importance (en iyi model)
    if hasattr(best_model, 'feature_importances_'):
        importances = sorted(
            zip(FEATURES, best_model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        imp_block = "\nÖzellik Önemleri (En İyi Model):\n"
        for feat, imp in importances:
            bar = '█' * int(imp * 50)
            imp_block += f"  {feat:<22} {bar} {imp:.4f}\n"
        print(imp_block)
        lines.append(imp_block)

    lines.append(f"\nEğitim hisseleri: {', '.join(WATCHLIST)}")
    lines.append(f"\nKullanılan feature'lar: {', '.join(FEATURES)}")
    lines.append("\nYenilikler:\n  - Sentiment Decay (3 günlük)\n  - RSI_14\n  - MA20_ratio\n  - Momentum_5d\n  - Sentiment_3d_avg\n  - News_count_7d")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"   📄 Rapor '{REPORT_FILE}' kaydedildi.")


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

def main():
    print("🚀 Model Trainer v2 Başlıyor...\n")
    print(f"   Yenilikler: Sentiment Decay + RSI + MA20 + Momentum + XGBoost\n")

    # 1. Veri yükle
    df = load_base_data(CSV_FILE)

    # 2. Sentiment decay uygula
    df = apply_sentiment_decay(df)

    # 3. Feature engineering
    df = add_technical_features(df)

    # 4. CSV kaydet
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n   💾 '{OUTPUT_CSV}' kaydedildi ({len(df)} satır, {len(FEATURES)} feature).")

    # 5. Modelleri eğit ve karşılaştır
    results, best_name, best_model, X_test, y_test = train_and_compare(df)

    # 6. Rapor
    print_and_save_report(results, best_name, best_model, df)

    # 7. En iyi modeli kaydet
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': best_model, 'features': FEATURES, 'model_name': best_name}, f)
    print(f"   💾 En iyi model '{MODEL_FILE}' olarak kaydedildi.")

    print(f"\n✅ TAMAMLANDI!")
    print(f"   → {OUTPUT_CSV}")
    print(f"   → {MODEL_FILE}  ({best_name})")
    print(f"   → {REPORT_FILE}")
    print(f"\n💡 ml_service.py'de MODEL_PATH = 'model_v2.pkl' yap ve yeniden başlat!")


if __name__ == "__main__":
    main()