"""
ml_service.py
-------------
ML tahmin servisi. :8002 portunda çalışır.
Node.js backend bu servise istek atar, tahmin alır.

Kullanım:
  uvicorn ml_service:app --host 0.0.0.0 --port 8002 --reload

Endpoint:
  GET /predict?symbol=AAPL
  GET /health
"""

import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# MODEL YÜKLEMESİ
# ─────────────────────────────────────────────

MODEL_PATH   = "model_v2c.pkl"         # v2c: 9460 satir + EWM RSI + VIX_log + TimeSeriesSplit
FINNHUB_KEY  = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_BASE = "https://finnhub.io/api/v1"

SUPPORTED_SYMBOLS = {
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
    'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL'
}

print("Model yukleniyor...")
try:
    with open(MODEL_PATH, 'rb') as f:
        saved = pickle.load(f)

    USES_PIPELINE = saved.get('uses_pipeline', False)
    if USES_PIPELINE:
        MODEL = saved['pipeline']
    else:
        MODEL = saved['model']

    FEATURES   = saved['features']
    MODEL_NAME = saved.get('model_name', 'Unknown')
    print(f"Model hazir. ({MODEL_NAME}) | Pipeline: {USES_PIPELINE}")
    print(f"   Ozellikler: {FEATURES}")
except FileNotFoundError:
    raise RuntimeError(f"'{MODEL_PATH}' bulunamadi! Once model_trainer_v2c.py calistirin.")

# ─────────────────────────────────────────────
# FINBERT SINGLETON — sadece bir kere yuklenir
# ─────────────────────────────────────────────

FINBERT_ANALYZER = None

def get_finbert():
    """FinBERT pipeline'ini ilk cagirida yukler, sonra cache'den doner."""
    global FINBERT_ANALYZER
    if FINBERT_ANALYZER is None:
        print("FinBERT yukleniyor (ilk istek, bir kere yapilir)...")
        try:
            from transformers import pipeline as hf_pipeline
            FINBERT_ANALYZER = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512
            )
            print("FinBERT hazir. Sonraki istekler aninda gelecek.")
        except Exception as e:
            print(f"FinBERT yuklenemedi: {e}")
    return FINBERT_ANALYZER

app = FastAPI(
    title="Hisse Tahmin ML Servisi v2",
    description="Feature Engineering + Sentiment Decay ile fiyat yön tahmini",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# TEKNİK FEATURE HESAPLAMA
# ─────────────────────────────────────────────

def get_technical_features(symbol: str) -> dict:
    """
    yfinance'den son 60 günlük veri çeker.
    Tüm yeni feature'ları hesaplar:
    Daily_Return, Volatility_14d, ATR_14d, VIX_Close,
    RSI_14, MA20_ratio, Momentum_5d
    """
    end   = datetime.now()
    start = end - timedelta(days=90)  # RSI ve MA20 için yeterli geçmiş

    raw = yf.download(
        symbol,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False
    )

    if raw.empty or len(raw) < 22:
        raise HTTPException(
            status_code=404,
            detail=f"'{symbol}' için yeterli fiyat verisi bulunamadı."
        )

    # MultiIndex temizleme
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
    except Exception:
        pass

    close = raw['Close'].squeeze()
    high  = raw['High'].squeeze()
    low   = raw['Low'].squeeze()

    # ── Daily_Return ────────────────────────────────────────────
    daily_return = float(close.pct_change().iloc[-1])

    # ── Volatility_14d ──────────────────────────────────────────
    volatility_14d = float(close.pct_change().rolling(14).std().iloc[-1] * np.sqrt(252))

    # ── ATR_14d ─────────────────────────────────────────────────
    hl  = high - low
    hc  = (high - close.shift()).abs()
    lc  = (low  - close.shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_14d = float(tr.rolling(14).mean().iloc[-1])

    # ── VIX ─────────────────────────────────────────────────────
    vix_raw = yf.download(
        '^VIX',
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        progress=False
    )
    try:
        if isinstance(vix_raw.columns, pd.MultiIndex):
            vix_raw.columns = [c[0] for c in vix_raw.columns]
    except Exception:
        pass
    vix_close = float(vix_raw['Close'].iloc[-1]) if not vix_raw.empty else 20.0
    vix_log   = float(np.log(max(vix_close, 1.0)))

    # ── RSI_14 ──────────────────────────────────────────────────
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14   = float((100 - (100 / (1 + rs))).iloc[-1])

    # ── MA20_ratio ──────────────────────────────────────────────
    ma20      = close.rolling(20, min_periods=5).mean()
    ma20_ratio = float(((close - ma20) / ma20).iloc[-1])

    # ── Momentum_5d ─────────────────────────────────────────────
    momentum_5d = float(close.pct_change(5).iloc[-1])

    return {
        'current_price':  float(close.iloc[-1]),
        'Daily_Return':   daily_return,
        'Volatility_14d': volatility_14d,
        'ATR_14d':        atr_14d,
        'VIX_Close':      vix_close,
        'VIX_log':        vix_log,
        'RSI_14':         rsi_14,
        'MA20_ratio':     ma20_ratio,
        'Momentum_5d':    momentum_5d,
    }


# ─────────────────────────────────────────────
# SENTIMENT HESAPLAMA (Finnhub + FinBERT)
# ─────────────────────────────────────────────

_sentiment_cache: dict = {}

def get_sentiment_features(symbol: str) -> dict:
    """
    Bugünkü sentiment skorunu döner (önbellekli).
    Decay ve 3d_avg için tek günlük tahmin yaptığımızdan
    Sentiment_Decay = bugünkü skor
    Sentiment_3d_avg = önbellekteki son 3 günün ortalaması
    News_count_7d = son 7 günde haber olan gün sayısı
    """
    today_str = datetime.now().strftime('%Y-%m-%d')
    cached = _sentiment_cache.get(symbol)

    if cached and cached['date'] == today_str:
        print(f"   [Cache] {symbol} sentiment cached: {cached['score']:.3f}")
        return cached

    print(f"   [FinBERT] {symbol} sentiment computing...")
    score = _compute_sentiment(symbol)

    # Geçmiş skorları tut (3 günlük ortalama için)
    history = _sentiment_cache.get(f"{symbol}_history", [])
    history.append({'date': today_str, 'score': score})
    history = history[-7:]  # Son 7 günü sakla

    # 3 günlük ortalama
    recent_scores = [h['score'] for h in history[-3:]]
    sentiment_3d_avg = float(np.mean(recent_scores)) if recent_scores else 0.0

    # Son 7 günde haber olan gün sayısı
    news_count_7d = float(sum(1 for h in history if h['score'] != 0.0))

    result = {
        'date':              today_str,
        'score':             score,
        'Sentiment_Decay':   score,           # Bugün haber varsa direkt skor
        'Sentiment_3d_avg':  sentiment_3d_avg,
        'News_count_7d':     news_count_7d,
    }

    _sentiment_cache[symbol] = result
    _sentiment_cache[f"{symbol}_history"] = history
    print(f"   [FinBERT] {symbol} score: {score:.3f} | 3d_avg: {sentiment_3d_avg:.3f} | 7d_count: {news_count_7d}")
    return result


def _compute_sentiment(symbol: str) -> float:
    """Finnhub'dan haberleri çekip FinBERT ile analiz eder."""
    if not FINNHUB_KEY:
        return 0.0

    today    = datetime.now()
    week_ago = today - timedelta(days=7)

    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                'symbol': symbol,
                'from':   week_ago.strftime('%Y-%m-%d'),
                'to':     today.strftime('%Y-%m-%d'),
                'token':  FINNHUB_KEY
            },
            timeout=10
        )
        news = resp.json()
    except Exception:
        return 0.0

    if not news:
        return 0.0

    headlines = [
        f"{n.get('headline', '')}. {n.get('summary', '')}"
        for n in news[:20]
        if n.get('headline')
    ]
    if not headlines:
        return 0.0

    try:
        analyzer = get_finbert()
        if analyzer is None:
            return 0.0
        results = analyzer(headlines)
        scores = []
        for r in results:
            if r['label'] == 'positive':
                scores.append(r['score'])
            elif r['label'] == 'negative':
                scores.append(-r['score'])
            else:
                scores.append(0.0)
        return float(np.mean(scores))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
# RESPONSE MODELİ
# ─────────────────────────────────────────────

class PredictionResponse(BaseModel):
    symbol:           str
    prediction:       str
    prediction_int:   int
    confidence:       float
    sentiment_score:  float
    daily_return:     float
    volatility_14d:   float
    vix_close:        float
    current_price:    float
    rsi_14:           float
    ma20_ratio:       float
    momentum_5d:      float
    is_supported:     bool
    model_name:       str
    message:          str


# ─────────────────────────────────────────────
# ENDPOINT'LER
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":            "ok",
        "model_loaded":      MODEL is not None,
        "model_name":        MODEL_NAME,
        "features":          FEATURES,
        "supported_symbols": sorted(SUPPORTED_SYMBOLS),
        "timestamp":         datetime.now().isoformat()
    }


@app.get("/predict", response_model=PredictionResponse)
def predict(symbol: str):
    symbol       = symbol.upper().strip()
    is_supported = symbol in SUPPORTED_SYMBOLS

    message = (
        "Tahmin yapıldı."
        if is_supported
        else f"'{symbol}' model eğitim setinde yok. Tahmin genel piyasa örüntülerine dayalıdır."
    )

    # 1. Teknik feature'lar
    try:
        tech = get_technical_features(symbol)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 2. Sentiment feature'lar (önbellekli)
    sent = get_sentiment_features(symbol)

    # 3. Feature vektoru — model egitimindeki sirayla birebir ayni
    feature_map = {
        'Daily_Return':    tech['Daily_Return'],
        'Volatility_14d':  tech['Volatility_14d'],
        'ATR_14d':         tech['ATR_14d'],
        'VIX_Close':       tech['VIX_Close'],   # geriye donuk uyumluluk
        'VIX_log':         tech['VIX_log'],     # v2b modeli bunu kullanir
        'RSI_14':          tech['RSI_14'],
        'MA20_ratio':      tech['MA20_ratio'],
        'Momentum_5d':     tech['Momentum_5d'],
        'Sentiment_Decay': sent['Sentiment_Decay'],
        'Sentiment_3d_avg': sent['Sentiment_3d_avg'],
        'News_count_7d':   sent['News_count_7d'],
    }

    X = np.array([[feature_map[f] for f in FEATURES]])

    # 4. Tahmin
    pred_int   = int(MODEL.predict(X)[0])
    pred_prob  = MODEL.predict_proba(X)[0]
    confidence = float(max(pred_prob))
    pred_label = "YUKSELIS" if pred_int == 1 else "DUSUS"

    print(f"[{symbol}] Tahmin: {pred_label} "
          f"(Guven: %{confidence*100:.1f}, "
          f"RSI: {tech['RSI_14']:.1f}, "
          f"Sentiment: {sent['score']:.3f})")

    return PredictionResponse(
        symbol=symbol,
        prediction=pred_label,
        prediction_int=pred_int,
        confidence=round(confidence, 4),
        sentiment_score=round(sent['score'], 4),
        daily_return=round(tech['Daily_Return'], 4),
        volatility_14d=round(tech['Volatility_14d'], 4),
        vix_close=round(tech['VIX_Close'], 2),
        current_price=round(tech['current_price'], 2),
        rsi_14=round(tech['RSI_14'], 2),
        ma20_ratio=round(tech['MA20_ratio'], 4),
        momentum_5d=round(tech['Momentum_5d'], 4),
        is_supported=is_supported,
        model_name=MODEL_NAME,
        message=message
    )