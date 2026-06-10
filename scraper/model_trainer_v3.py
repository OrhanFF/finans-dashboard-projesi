"""
model_trainer_v3.py
-------------------
Tüm sistemin en gelişmiş, nihai versiyonu (v3).
Önceki V2C sürümündeki teknik ve haber optimizasyonlarına,
analiz edilmiş Twitter (X) sentiment ve hacim verilerini ekler.

Yenilikler (v2c -> v3):
  - Tweet_Volume & Tweet_Volume_7d_Avg
  - Tweet_Sentiment_Avg & Tweet_Sentiment_3d
  - Tweet_Toxic_Ratio (Spam/Troll filtreli)
  - Tweet_Positive_Ratio & Tweet_Negative_Ratio

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

INPUT_CSV   = "MultiStock_Training_v3.csv"
MODEL_FILE  = "model_v3.pkl"
REPORT_FILE = "model_report_v3.txt"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

FEATURES = [
    # Teknik göstergeler (v2c'den)
    'Daily_Return',
    'Volatility_14d',
    'ATR_14d',
    'VIX_log',
    'RSI_14',
    'MA20_ratio',
    'Momentum_5d',
    
    # Haber sentiment (v2c'den)
    'Sentiment_Decay',
    'Sentiment_3d_avg',
    'News_count_7d',
    
    # Tweet feature'ları (YENİ v3)
    'Tweet_Volume',
    'Tweet_Volume_7d_Avg',
    'Tweet_Sentiment_Avg',
    'Tweet_Sentiment_3d',
    'Tweet_Toxic_Ratio',
    'Tweet_Positive_Ratio',
    'Tweet_Negative_Ratio',
]


# ─────────────────────────────────────────────
# ADIM 1 — VERİ YÜKLE
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    print(f"📂 1. '{csv_path}' yükleniyor...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ '{csv_path}' bulunamadı! Lütfen önce build_tweet_features.bat dosyasını çalıştırın.")

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)
    
    # Nan değerleri temizle
    df.dropna(subset=FEATURES, inplace=True)

    print(f"   ✅ {len(df)} satır, {df['symbol'].nunique()} hisse yüklendi.")
    print(f"   📅 Tarih: {df['Date'].min().date()} - {df['Date'].max().date()}")

    dusus    = (df['Target'] == 0).sum()
    yukselis = (df['Target'] == 1).sum()
    print(f"   📉 Sınıf dağılımı: DÜŞÜŞ {dusus} (%{dusus/len(df)*100:.1f}) | YÜKSELİŞ {yukselis} (%{yukselis/len(df)*100:.1f})")

    # Tweet verisi olan günlerin oranı
    tweet_days = (df['Tweet_Volume'] > 0).sum()
    print(f"   🐦 Tweet verisi olan günler: {tweet_days}/{len(df)} (%{tweet_days/len(df)*100:.1f})")

    return df


# ─────────────────────────────────────────────
# ADIM 2 — WALK-FORWARD CV
# ─────────────────────────────────────────────

def walk_forward_cv(pipeline, df, n_splits=5):
    """
    Hisse bazında zaman-ileri CV: her hissenin kendi zaman ekseninde
    TimeSeriesSplit uygulanır, fold'lar tüm hisselerden birleştirilir.
    Böylece bir fold'da bazı hisselerin tamamen eğitimde/testte kalması
    (sembol bazlı sızıntı) önlenir.
    """
    scores = []
    symbol_splits = {}
    for symbol, sym_df in df.groupby('symbol'):
        idx = sym_df.index.to_numpy()
        tscv = TimeSeriesSplit(n_splits=n_splits)
        symbol_splits[symbol] = [(idx[tr], idx[te]) for tr, te in tscv.split(idx)]

    X_all = df[FEATURES].fillna(0.0).values
    y_all = df['Target'].values
    idx_to_pos = {idx_val: pos for pos, idx_val in enumerate(df.index)}

    for fold in range(n_splits):
        train_idx = np.concatenate([symbol_splits[s][fold][0] for s in symbol_splits])
        test_idx  = np.concatenate([symbol_splits[s][fold][1] for s in symbol_splits])
        train_pos = [idx_to_pos[i] for i in train_idx]
        test_pos  = [idx_to_pos[i] for i in test_idx]

        pipeline.fit(X_all[train_pos], y_all[train_pos])
        acc = accuracy_score(y_all[test_pos], pipeline.predict(X_all[test_pos]))
        scores.append(acc)
        print(f"      Fold {fold+1}: {acc:.3f}")
    return {'mean': np.mean(scores), 'std': np.std(scores), 'scores': scores}


def per_symbol_split(df, train_frac=0.8):
    """
    Her hissenin kendi zaman ekseninde ilk %80'i eğitim, son %20'si test
    olacak şekilde böler. Böylece her hisse hem eğitimde hem testte yer
    alır ve test verisi gerçekten "gelecek" günleri temsil eder.
    """
    train_mask = pd.Series(False, index=df.index)
    for symbol, sym_df in df.groupby('symbol'):
        n = len(sym_df)
        split = int(n * train_frac)
        train_mask.loc[sym_df.index[:split]] = True
    return train_mask.values


# ─────────────────────────────────────────────
# ADIM 3 — MODEL EĞİTİMİ
# ─────────────────────────────────────────────

def train(df: pd.DataFrame):
    print("\n🤖 2. Modeller eğitiliyor (Hisse Bazında Zamansal Split %80/%20)...")
    
    X_raw = df[FEATURES].fillna(0.0).values
    y     = df['Target'].values

    # Hisse bazında zamansal (kronolojik) split (%80 Eğitim, %20 Test)
    # Her hissenin ilk %80'i eğitime, son %20'si teste gider.
    train_mask = per_symbol_split(df, train_frac=0.8)
    X_tr, X_te = X_raw[train_mask], X_raw[~train_mask]
    y_tr, y_te = y[train_mask],     y[~train_mask]
    print(f"   Eğitim: {len(X_tr)} | Test: {len(X_te)}")

    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10,
            min_samples_leaf=5, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, max_features='sqrt',
            subsample=0.8, random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric='logloss',
            use_label_encoder=False,
            random_state=42, n_jobs=-1
        )

    results   = {}
    best_acc  = 0
    best_name = None
    best_pipe = None

    for name, clf in classifiers.items():
        print(f"\n   [{name}]")
        # Pipeline: Önce veriyi standartlaştır (0 ortalama, 1 varyans), sonra modele ver
        pipe = Pipeline([('scaler', StandardScaler()), ('model', clf)])

        cv = walk_forward_cv(pipe, df, n_splits=5)
        print(f"   CV: %{cv['mean']*100:.2f} ± %{cv['std']*100:.2f}")

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)

        results[name] = {
            'pipeline': pipe, 
            'accuracy': acc,
            'cv_mean':  cv['mean'], 
            'cv_std':   cv['std'],
            'report':   classification_report(y_te, y_pred, target_names=['Düşüş', 'Yükseliş']),
            'cm':       confusion_matrix(y_te, y_pred),
        }
        print(f"   Test Accuracy: %{acc*100:.2f}")

        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    return results, best_name, best_pipe


# ─────────────────────────────────────────────
# ADIM 4 — RAPOR VE KAYIT
# ─────────────────────────────────────────────

def save_report(results, best_name, best_pipe):
    print(f"\n{'='*55}")
    print(f"  🏆 EN İYİ MODEL: {best_name}  (%{results[best_name]['accuracy']*100:.2f})")
    print(f"{'='*55}")

    lines = [f"Model: {best_name} v3\n",
             f"Veri: {INPUT_CSV} (Teknik + Haber + Twitter Sentiment)\n"]

    for name, r in results.items():
        mark = " [★ EN İYİ]" if name == best_name else ""
        b = (f"\n{'─'*50}\n{name}{mark}\n"
             f"Test: %{r['accuracy']*100:.2f} | "
             f"CV: %{r['cv_mean']*100:.2f} +/- %{r['cv_std']*100:.2f}\n\n"
             f"{r['report']}\nCM:\n{r['cm']}\n")
        print(b)
        lines.append(b)

    # Feature importance
    clf = best_pipe.named_steps['model']
    if hasattr(clf, 'feature_importances_'):
        imps = sorted(zip(FEATURES, clf.feature_importances_),
                      key=lambda x: x[1], reverse=True)
        block = "\nÖzellik Önemleri:\n"
        for f, imp in imps:
            bar = '█' * int(imp * 50)
            block += f"  {f:<25} {bar} {imp:.4f}\n"
        print(block)
        lines.append(block)

        # Kategorik katkıları hesapla
        tech_feats  = [f for f in FEATURES if f not in ['Sentiment_Decay', 'Sentiment_3d_avg', 'News_count_7d'] and not f.startswith('Tweet_')]
        news_feats  = ['Sentiment_Decay', 'Sentiment_3d_avg', 'News_count_7d']
        tweet_feats = [f for f in FEATURES if f.startswith('Tweet_')]

        tech_imp  = sum(i for f, i in imps if f in tech_feats)
        news_imp  = sum(i for f, i in imps if f in news_feats)
        tweet_imp = sum(i for f, i in imps if f in tweet_feats)

        summary = (
            f"\n📊 Veri Kaynaklarının Karar Alma Etkisi (Ağırlığı):\n"
            f"   Teknik Analiz (Fiyat/Hacim)  : %{tech_imp*100:.1f}\n"
            f"   Finnhub (Haberler)           : %{news_imp*100:.1f}\n"
            f"   Twitter (Sosyal Medya)       : %{tweet_imp*100:.1f}\n"
        )
        print(summary)
        lines.append(summary)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n   📄 Rapor '{REPORT_FILE}' olarak kaydedildi.")


def main():
    print("🚀 Model Trainer v3 Başlıyor")
    print("   Nihai Sürüm: Teknik Analiz + Haber Duygusu + Twitter Duygusu\n")

    # 1. Yükle
    df = load_data(INPUT_CSV)

    # 2. Eğit
    results, best_name, best_pipe = train(df)

    # 3. Rapor
    save_report(results, best_name, best_pipe)

    # 4. Kaydet
    with open(MODEL_FILE, 'wb') as f:
        from datetime import datetime
        pickle.dump({
            'pipeline':      best_pipe,
            'features':      FEATURES,
            'model_name':    f"{best_name} v3",
            'uses_pipeline': True,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'feature_count': len(FEATURES),
        }, f)
    print(f"   💾 Nihai Model '{MODEL_FILE}' olarak kaydedildi.")

    print(f"\n✅ İŞLEM TAMAMLANDI!")
    print(f"   → En gelişmiş sürüm olan v3 başarıyla eğitildi ve kaydedildi.")
    print(f"\n💡 Canlıya almak için ml_service.py içindeki MODEL_PATH değerini '{MODEL_FILE}' olarak güncelleyin!")


if __name__ == "__main__":
    main()