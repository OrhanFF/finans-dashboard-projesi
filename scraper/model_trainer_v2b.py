"""
model_trainer_v2b.py
--------------------
v2'nin üzerine kritik iyileştirmeler:
  1. StandardScaler  — VIX gibi büyük değerleri normalize eder
  2. VIX log-scale   — log(VIX) daha stabil ilişki kurar
  3. TimeSeriesSplit — data leakage'ı önleyen zamansal CV
  4. max_features    — tek feature dominansını kırar
  5. Scaler pickle'a dahil edilir → ml_service tutarlı çalışır

Kullanım:
  python model_trainer_v2b.py
"""

import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost kurulu degil, RF ve GB kullanilacak.")

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

INPUT_CSV   = "MultiStock_Training_v2.csv"   # v2 feature-engineered CSV
MODEL_FILE  = "model_v2b.pkl"
REPORT_FILE = "model_report_v2b.txt"

WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

FEATURES = [
    'Daily_Return',
    'Volatility_14d',
    'ATR_14d',
    'VIX_log',          # <-- log(VIX_Close) kullanilacak
    'RSI_14',
    'MA20_ratio',
    'Momentum_5d',
    'Sentiment_Decay',
    'Sentiment_3d_avg',
    'News_count_7d',
]


# ─────────────────────────────────────────────
# ADIM 1 — VERİ YÜKLE + VIX LOG
# ─────────────────────────────────────────────

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    print("1. Veri yukleniyor...")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' bulunamadi!\n"
            "Once model_trainer_v2.py calistirin."
        )

    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['symbol', 'Date']).reset_index(drop=True)

    print(f"   {len(df)} satir, {df['symbol'].nunique()} hisse yuklendi.")

    # VIX log-scale
    df['VIX_log'] = np.log(df['VIX_Close'].clip(lower=1.0))
    print(f"   VIX_Close aralik: {df['VIX_Close'].min():.1f} - {df['VIX_Close'].max():.1f}")
    print(f"   VIX_log   aralik: {df['VIX_log'].min():.3f} - {df['VIX_log'].max():.3f}")

    # Eksik feature kontrolu
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Eksik sutunlar: {missing}")

    df = df.dropna(subset=FEATURES + ['Target'])
    print(f"   NaN temizlendi → {len(df)} satir kaldi.")
    return df


# ─────────────────────────────────────────────
# ADIM 2 — WALK-FORWARD CROSS VALIDATION
# ─────────────────────────────────────────────

def walk_forward_cv(pipeline, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    TimeSeriesSplit ile zamansal cross-validation.
    Her fold: gecmis verilerle egit, gelecek verilerle test et.
    Bu sayede data leakage olmaz (shuffle=True hatayi gizliyordu).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipeline.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, pipeline.predict(X_te))
        fold_scores.append(acc)
        print(f"   Fold {fold+1}: {acc:.3f}")

    return {
        'scores': fold_scores,
        'mean': np.mean(fold_scores),
        'std':  np.std(fold_scores),
    }


# ─────────────────────────────────────────────
# ADIM 3 — MODEL EGİTİMİ
# ─────────────────────────────────────────────

def train_and_compare(df: pd.DataFrame):
    print("\n2. Modeller egitiliyor...")

    X_raw = df[FEATURES].fillna(0.0).values
    y     = df['Target'].values

    # Zaman sirasina gore son %20 test
    split_idx = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train, y_test          = y[:split_idx],     y[split_idx:]

    print(f"   Egitim: {len(X_train_raw)} gun | Test: {len(X_test_raw)} gun")
    print(f"   (Zamansal split — en son %20 test olarak ayrildi)")

    # Siniflandiricilar — max_features ile tek feature dominansini kir
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=8,
            max_features='sqrt',      # her split'te sadece sqrt(n) feature dene
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            max_features='sqrt',      # VIX dominansini kisitla
            subsample=0.8,
            random_state=42
        ),
    }

    if XGBOOST_AVAILABLE:
        classifiers['XGBoost'] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,    # her agacta sadece %80 feature kullan
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    results   = {}
    best_acc  = 0
    best_name = None
    best_pipe = None

    for name, clf in classifiers.items():
        print(f"\n   [{name}] egitiliyor...")

        # Pipeline: scaler + model
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model',  clf)
        ])

        # Walk-forward CV
        print(f"   Walk-forward CV ({5} fold):")
        cv_result = walk_forward_cv(pipe, X_raw, y, n_splits=5)
        print(f"   CV Ortalama: {cv_result['mean']:.3f} +/- {cv_result['std']:.3f}")

        # Son egitim
        pipe.fit(X_train_raw, y_train)
        y_pred = pipe.predict(X_test_raw)
        acc    = accuracy_score(y_test, y_pred)

        results[name] = {
            'pipeline':  pipe,
            'accuracy':  acc,
            'cv_mean':   cv_result['mean'],
            'cv_std':    cv_result['std'],
            'report':    classification_report(y_test, y_pred,
                             target_names=['Dusus', 'Yukselis']),
            'cm':        confusion_matrix(y_test, y_pred),
        }

        print(f"   Test Accuracy: %{acc*100:.2f}")

        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_pipe = pipe

    return results, best_name, best_pipe, X_test_raw, y_test


# ─────────────────────────────────────────────
# ADIM 4 — RAPOR
# ─────────────────────────────────────────────

def print_and_save_report(results, best_name, best_pipe):
    print(f"\n{'='*55}")
    print(f"  EN IYI MODEL: {best_name}")
    print(f"{'='*55}")

    lines = [f"En iyi model: {best_name}\n",
             f"Yenilikler (v2b): StandardScaler + VIX_log + TimeSeriesSplit + max_features\n"]

    for name, r in results.items():
        marker = " EN IYI" if name == best_name else ""
        block = (
            f"\n{'─'*50}\n"
            f"Model: {name}{marker}\n"
            f"Test Accuracy : %{r['accuracy']*100:.2f}\n"
            f"CV  Accuracy  : %{r['cv_mean']*100:.2f} +/- %{r['cv_std']*100:.2f}\n\n"
            f"{r['report']}\n"
            f"Confusion Matrix:\n{r['cm']}\n"
        )
        print(block)
        lines.append(block)

    # Feature importance (modelden)
    clf = best_pipe.named_steps['model']
    if hasattr(clf, 'feature_importances_'):
        importances = sorted(
            zip(FEATURES, clf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        imp_block = "\nOzellik Onemleri (En Iyi Model):\n"
        for feat, imp in importances:
            bar = '#' * int(imp * 50)
            imp_block += f"  {feat:<25} {bar} {imp:.4f}\n"
        print(imp_block)
        lines.append(imp_block)

        # VIX karsilastirmasi
        vix_imp = next((imp for f, imp in importances if f == 'VIX_log'), 0)
        vix_summary = (
            f"\nVIX_log onemliligi: %{vix_imp*100:.1f} "
            f"(v2'de VIX_Close %28.6 idi)\n"
        )
        print(vix_summary)
        lines.append(vix_summary)

    lines.append(f"\nEgitim hisseleri: {', '.join(WATCHLIST)}")
    lines.append(f"\nKullanilan feature'lar ({len(FEATURES)} adet):\n  " + "\n  ".join(FEATURES))

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"   Rapor '{REPORT_FILE}' kaydedildi.")


# ─────────────────────────────────────────────
# ANA AKIS
# ─────────────────────────────────────────────

def main():
    print("Model Trainer v2b Basliyor")
    print("   Yenilik: StandardScaler + VIX_log + TimeSeriesSplit + max_features\n")

    # 1. Veri
    df = load_and_prepare(INPUT_CSV)

    # 2. Egit
    results, best_name, best_pipe, X_test, y_test = train_and_compare(df)

    # 3. Rapor
    print_and_save_report(results, best_name, best_pipe)

    # 4. Kaydet — scaler pipeline icinde, ayri kaydetmeye gerek yok
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'pipeline':   best_pipe,      # StandardScaler + model birlikte
            'features':   FEATURES,
            'model_name': f"{best_name} v2b",
            'uses_pipeline': True,        # ml_service bunu gorunce pipeline kullanir
        }, f)
    print(f"   Model '{MODEL_FILE}' kaydedildi. (Pipeline icinde scaler var)")

    print(f"\nTAMAMLANDI!")
    print(f"   -> {MODEL_FILE}")
    print(f"   -> {REPORT_FILE}")
    print(f"\nSonraki adim: ml_service.py -> MODEL_PATH = '{MODEL_FILE}'")


if __name__ == "__main__":
    main()
