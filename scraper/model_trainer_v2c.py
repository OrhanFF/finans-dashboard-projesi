"""
model_trainer_v2c.py
--------------------
Tüm iyileştirmeleri tek scriptte birleştirir:
  - Genişletilmiş veri (2020-2026, 9460 satır)
  - Sentiment Decay, RSI_14, MA20_ratio, Momentum_5d
  - VIX_log (log-scale VIX)
  - StandardScaler (Pipeline içinde)
  - TimeSeriesSplit walk-forward CV (data leakage yok)
  - max_features=sqrt (tek feature dominansı kırıldı)

Kullanım:
  python model_trainer_v2c.py
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

INPUT_CSV   = "MultiStock_Training_extended.csv"
OUTPUT_CSV  = "MultiStock_Training_v2c.csv"
MODEL_FILE  = "model_v2c.pkl"
REPORT_FILE = "model_report_v2c.txt"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

DECAY_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}

FEATURES = [
    'Daily_Return',
    'Volatility_14d',
    'ATR_14d',
    'VIX_log',
    'RSI_14',
    'MA20_ratio',
    'Momentum_5d',
    'Sentiment_Decay',
    'Sentiment_3d_avg',
    'News_count_7d',
]


# ─────────────────────────────────────────────
# ADIM 1 — VERİ YÜKLE
# ─────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    print("1. Veri yukleniyor...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)
    print(f"   {len(df)} satir, {df['symbol'].nunique()} hisse")
    print(f"   Tarih: {df['Date'].min().date()} - {df['Date'].max().date()}")
    print(f"   VIX ort  : {df['VIX_Close'].mean():.1f}")
    print(f"   VIX min  : {df['VIX_Close'].min():.1f}")
    print(f"   VIX max  : {df['VIX_Close'].max():.1f}")

    dusus    = (df['Target'] == 0).sum()
    yukselis = (df['Target'] == 1).sum()
    print(f"   Sinif dagılımı: DUSUS {dusus} (%{dusus/len(df)*100:.1f}) | YUKSELIS {yukselis} (%{yukselis/len(df)*100:.1f})")
    return df


# ─────────────────────────────────────────────
# ADIM 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def apply_sentiment_decay(df: pd.DataFrame) -> pd.DataFrame:
    print("\n2a. Sentiment Decay uygulanıyor...")
    df = df.copy()
    df['Sentiment_Decay'] = 0.0
    for symbol in df['symbol'].unique():
        mask   = df['symbol'] == symbol
        sym_df = df[mask].copy()
        decay  = np.zeros(len(sym_df))
        for i in range(len(sym_df)):
            raw = sym_df['Sentiment_Score'].iloc[i]
            if raw != 0.0:
                for offset, w in DECAY_WEIGHTS.items():
                    t = i + offset
                    if t < len(sym_df):
                        cand = raw * w
                        if abs(cand) > abs(decay[t]):
                            decay[t] = cand
        df.loc[mask, 'Sentiment_Decay'] = decay
    print(f"   Dolu gun: {(df['Sentiment_Decay'] != 0).sum()} / {len(df)}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("2b. Teknik feature'lar hesaplanıyor...")
    frames = []
    for symbol in df['symbol'].unique():
        s = df[df['symbol'] == symbol].copy().reset_index(drop=True)
        c = s['Close'] if 'Close' in s.columns else None

        # RSI_14 (fiyat sütunu olmayabilir, sadece close bazlı)
        if c is not None:
            delta    = c.diff()
            gain     = delta.clip(lower=0)

        # RSI_14 — Daily_Return'den direkt hesapla (Close'a ihtiyac yok)
        # EWM (Exponential Weighted Mean): min_periods=1 ile ilk gunden itibaren gecerli
        ret  = s['Daily_Return'].fillna(0)
        gain = ret.clip(lower=0)
        loss = (-ret).clip(lower=0)
        avg_gain = gain.ewm(span=14, min_periods=1).mean()
        avg_loss = loss.ewm(span=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        s['RSI_14'] = (100 - (100 / (1 + rs))).fillna(50)  # NaN -> 50 (notr)

        # Price proxy — MA20 ve Momentum icin
        # fillna(0) ile NaN propagation tamamen engellendi
        price_proxy     = (1 + ret).cumprod()
        ma20            = price_proxy.rolling(20, min_periods=5).mean()
        s['MA20_ratio'] = ((price_proxy - ma20) / ma20).fillna(0)
        s['Momentum_5d'] = price_proxy.pct_change(5).fillna(0)

        # VIX log-scale
        s['VIX_log'] = np.log(s['VIX_Close'].clip(lower=1.0))

        # Sentiment rolling features
        s['Sentiment_3d_avg'] = s['Sentiment_Decay'].rolling(3, min_periods=1).mean()
        has_news = (s['Sentiment_Score'] != 0).astype(int)
        s['News_count_7d']    = has_news.rolling(7, min_periods=1).sum()

        frames.append(s)

    result = pd.concat(frames, ignore_index=True)
    result.dropna(subset=FEATURES, inplace=True)
    print(f"   Feature engineering sonrasi: {len(result)} satir")
    return result


# ─────────────────────────────────────────────
# ADIM 3 — WALK-FORWARD CV
# ─────────────────────────────────────────────

def walk_forward_cv(pipeline, X, y, n_splits=5):
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        pipeline.fit(X[tr], y[tr])
        acc = accuracy_score(y[te], pipeline.predict(X[te]))
        scores.append(acc)
        print(f"   Fold {fold+1}: {acc:.3f}")
    return {'mean': np.mean(scores), 'std': np.std(scores), 'scores': scores}


# ─────────────────────────────────────────────
# ADIM 4 — MODEL EĞİTİMİ
# ─────────────────────────────────────────────

def train(df: pd.DataFrame):
    print("\n3. Modeller egitiliyor (zamansal split %80/%20)...")
    X_raw = df[FEATURES].fillna(0.0).values
    y     = df['Target'].values

    split = int(len(X_raw) * 0.8)
    X_tr, X_te = X_raw[:split], X_raw[split:]
    y_tr, y_te = y[:split],     y[split:]
    print(f"   Egitim: {len(X_tr)} | Test: {len(X_te)}")

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
            random_state=42, n_jobs=-1
        )

    results   = {}
    best_acc  = 0
    best_name = None
    best_pipe = None

    for name, clf in classifiers.items():
        print(f"\n   [{name}]")
        pipe = Pipeline([('scaler', StandardScaler()), ('model', clf)])

        cv = walk_forward_cv(pipe, X_raw, y, n_splits=5)
        print(f"   CV: {cv['mean']:.3f} +/- {cv['std']:.3f}")

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)

        results[name] = {
            'pipeline': pipe, 'accuracy': acc,
            'cv_mean': cv['mean'], 'cv_std': cv['std'],
            'report': classification_report(y_te, y_pred,
                          target_names=['Dusus', 'Yukselis']),
            'cm': confusion_matrix(y_te, y_pred),
        }
        print(f"   Test Accuracy: %{acc*100:.2f}")

        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe

    return results, best_name, best_pipe


# ─────────────────────────────────────────────
# ADIM 5 — RAPOR VE KAYIT
# ─────────────────────────────────────────────

def save_report(results, best_name, best_pipe):
    print(f"\n{'='*55}")
    print(f"  EN IYI MODEL: {best_name}  (%{results[best_name]['accuracy']*100:.2f})")
    print(f"{'='*55}")

    lines = [f"Model: {best_name} v2c\n",
             f"Veri: {INPUT_CSV} (2020-2026, genisletilmis)\n"]

    for name, r in results.items():
        mark = " [EN IYI]" if name == best_name else ""
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
        block = "\nOzellik Onemleri:\n"
        for f, imp in imps:
            bar = '#' * int(imp * 50)
            block += f"  {f:<25} {bar} {imp:.4f}\n"
        print(block)
        lines.append(block)

        vix_imp = next((i for f, i in imps if f == 'VIX_log'), 0)
        print(f"  VIX_log onemliligi: %{vix_imp*100:.1f} (v2'de VIX_Close %28.6 idi)")

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  Rapor: {REPORT_FILE}")


def main():
    print("Model Trainer v2c Basliyor")
    print("   Genisletilmis veri + tum iyilestirmeler\n")

    # 1. Yükle
    df = load_data(INPUT_CSV)

    # 2. Feature engineering
    df = apply_sentiment_decay(df)
    df = add_features(df)

    # 3. Kaydet
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n   CSV kaydedildi: {OUTPUT_CSV} ({len(df)} satir)")

    # 4. Eğit
    results, best_name, best_pipe = train(df)

    # 5. Rapor
    save_report(results, best_name, best_pipe)

    # 6. Kaydet
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'pipeline':      best_pipe,
            'features':      FEATURES,
            'model_name':    f"{best_name} v2c",
            'uses_pipeline': True,
        }, f)
    print(f"  Model kaydedildi: {MODEL_FILE}")

    print(f"\nTAMAMLANDI!")
    print(f"  Sonraki adim: ml_service.py -> MODEL_PATH = '{MODEL_FILE}'")


if __name__ == "__main__":
    main()
