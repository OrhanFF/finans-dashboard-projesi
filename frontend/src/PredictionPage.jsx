import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = "http://localhost:3001/api";

const SUPPORTED_SYMBOLS = [
  'AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
  'JPM', 'JNJ', 'WMT', 'XOM', 'GOOGL'
];

function getVixComment(vix) {
  if (vix < 15) return { text: 'Düşük — piyasa sakin',    color: '#10b981' };
  if (vix < 25) return { text: 'Orta — normal volatilite', color: '#f59e0b' };
  return             { text: 'Yüksek — piyasa gergin',    color: '#ef4444' };
}

function getSentimentLabel(score) {
  if (score >  0.2) return { text: 'Pozitif', color: '#10b981' };
  if (score < -0.2) return { text: 'Negatif', color: '#ef4444' };
  return             { text: 'Nötr',    color: '#8b949e' };
}

function getRsiLabel(rsi) {
  if (rsi < 30) return { text: 'Aşırı Satım', color: '#10b981', hint: 'Toparlanma sinyali' };
  if (rsi > 70) return { text: 'Aşırı Alım',  color: '#ef4444', hint: 'Düzeltme riski' };
  return             { text: 'Normal',       color: '#8b949e', hint: 'Nötr bölge' };
}

function getMomentumLabel(mom) {
  if (mom >  0.03) return { text: 'Güçlü Yükseliş', color: '#10b981' };
  if (mom >  0)    return { text: 'Hafif Yükseliş',  color: '#34d399' };
  if (mom > -0.03) return { text: 'Hafif Düşüş',     color: '#f59e0b' };
  return               { text: 'Güçlü Düşüş',     color: '#ef4444' };
}

function getMa20Label(ratio) {
  if (ratio >  0.05) return { text: 'MA Üzerinde',  color: '#10b981', hint: `+${(ratio*100).toFixed(1)}%` };
  if (ratio < -0.05) return { text: 'MA Altında',   color: '#ef4444', hint: `${(ratio*100).toFixed(1)}%` };
  return              { text: 'MA Yakınında', color: '#f59e0b', hint: `${(ratio*100).toFixed(1)}%` };
}

function PredictionPage({ searchTerm }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const cleanSymbol = searchTerm
    ? searchTerm.trim().replace(/^#/, '').toUpperCase()
    : '';

  useEffect(() => {
    if (!cleanSymbol) { setData(null); return; }

    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);
      setData(null);
      try {
        const res = await axios.get(`${API_URL}/predict`, {
          params: { symbol: cleanSymbol }
        });
        setData(res.data);
      } catch (err) {
        if (err.response?.status === 503) {
          setError('ML servisi çalışmıyor. Terminalde şunu çalıştırın: uvicorn ml_service:app --port 8002');
        } else {
          setError('Tahmin alınamadı. Lütfen tekrar deneyin.');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, [cleanSymbol]);

  if (loading) {
    return (
      <div className="loading-indicator" role="status">
        <p>AI tahmini hesaplanıyor... (FinBERT haberleri analiz ediyor)</p>
      </div>
    );
  }

  if (error) {
    return <div className="error-message" role="alert"><p>{error}</p></div>;
  }

  if (!data) return null;

  const isUp          = data.prediction_int === 1;
  const dirClass      = isUp ? 'up' : 'down';
  const vixInfo       = getVixComment(data.vix_close);
  const sentInfo      = getSentimentLabel(data.sentiment_score);
  const rsiInfo       = getRsiLabel(data.rsi_14);
  const momInfo       = getMomentumLabel(data.momentum_5d);
  const ma20Info      = getMa20Label(data.ma20_ratio);
  const confidencePct = Math.round(data.confidence * 100);
  const dailyRetPct   = (data.daily_return * 100).toFixed(2);
  const volPct        = (data.volatility_14d * 100).toFixed(1);
  const sentBarWidth  = Math.abs(Math.max(-1, Math.min(1, data.sentiment_score))) * 50;
  const sentIsPos     = data.sentiment_score >= 0;
  const rsiPct        = Math.min(100, Math.max(0, data.rsi_14));

  return (
    <section className="results-list">

      {/* ── ANA TAHMİN KARTI ── */}
      <article className={`prediction-main-card ${dirClass}`}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <p className="prediction-label">AI Fiyat Tahmini · {data.symbol}</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap', marginTop: '6px' }}>
              <p className={`prediction-result ${dirClass}`}>
                {isUp ? '▲ Yükseliş' : '▼ Düşüş'}
              </p>
              <span className={`prediction-badge ${data.is_supported ? 'supported' : 'general'}`}>
                {data.is_supported ? '✓ Desteklenen Hisse' : '⚠ Genel Tahmin'}
              </span>
            </div>
            <p className="prediction-subtitle">Modelin yarınki kapanış yönü beklentisi</p>
          </div>

          <div style={{ textAlign: 'right' }}>
            <p className="prediction-label">Anlık Fiyat</p>
            <p className="prediction-price">${data.current_price.toFixed(2)}</p>
            <p className="prediction-daily-return" style={{ color: parseFloat(dailyRetPct) >= 0 ? '#10b981' : '#ef4444' }}>
              {parseFloat(dailyRetPct) >= 0 ? '+' : ''}{dailyRetPct}% bugün
            </p>
          </div>
        </div>

        {/* Güven Skoru */}
        <div className="confidence-section">
          <div className="confidence-header">
            <span className="prediction-label" style={{ margin: 0 }}>Güven Skoru</span>
            <span style={{ fontSize: '14px', fontWeight: '600', color: '#f5f5f5' }}>%{confidencePct}</span>
          </div>
          <div className="confidence-bar-bg">
            <div
              className={`confidence-bar-fill ${dirClass}`}
              style={{ width: `${confidencePct}%` }}
            />
          </div>
          <p className="confidence-hint">
            %50 = rastgele tahmin · %60+ = anlamlı sinyal · Model: {data.model_name || 'Gradient Boosting'}
          </p>
        </div>
      </article>

      {/* ── UYARI ── */}
      <div className="prediction-warning">
        ⚠ Bu tahmin geçmiş piyasa verilerine dayalı bir makine öğrenmesi modelidir. Yatırım kararı için kullanmayın.
      </div>

      {/* ── TEKNİK İNDİKATÖRLER BAŞLIĞI ── */}
      <p className="prediction-section-title">Teknik İndikatörler</p>

      {/* ── TEKNİK METRİK KARTLAR ── */}
      <div className="prediction-metric-grid">

        {/* RSI */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">RSI (14 Gün)</p>
          <p className="metric-value-lg" style={{ color: rsiInfo.color }}>{data.rsi_14.toFixed(1)}</p>
          <div className="rsi-bar-bg">
            <div className="rsi-zone rsi-zone-oversold" />
            <div className="rsi-zone rsi-zone-normal" />
            <div className="rsi-zone rsi-zone-overbought" />
            <div className="rsi-needle" style={{ left: `${rsiPct}%` }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px' }}>
            <span className="metric-hint">30</span>
            <span className="metric-hint" style={{ color: rsiInfo.color }}>{rsiInfo.text}</span>
            <span className="metric-hint">70</span>
          </div>
          <p className="metric-hint" style={{ marginTop: '4px' }}>{rsiInfo.hint}</p>
        </article>

        {/* MA20 Ratio */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">20G Ortalama Sapma</p>
          <p className="metric-value-lg" style={{ color: ma20Info.color }}>{ma20Info.hint}</p>
          <p className="metric-hint" style={{ color: ma20Info.color, marginTop: '6px' }}>{ma20Info.text}</p>
          <p className="metric-hint" style={{ marginTop: '4px' }}>Fiyat / 20 günlük MA</p>
        </article>

        {/* Momentum */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">5G Momentum</p>
          <p className="metric-value-lg" style={{ color: momInfo.color }}>
            {data.momentum_5d >= 0 ? '+' : ''}{(data.momentum_5d * 100).toFixed(2)}%
          </p>
          <p className="metric-hint" style={{ color: momInfo.color, marginTop: '6px' }}>{momInfo.text}</p>
          <p className="metric-hint" style={{ marginTop: '4px' }}>5 günlük fiyat değişimi</p>
        </article>

        {/* Volatilite */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">14G Volatilite</p>
          <p className="metric-value-lg" style={{ color: '#f5f5f5' }}>%{volPct}</p>
          <p className="metric-hint" style={{ marginTop: '6px' }}>Yıllıklandırılmış risk</p>
        </article>

      </div>

      {/* ── SENTIMENT & PİYASA BAŞLIĞI ── */}
      <p className="prediction-section-title">Sentiment & Piyasa</p>

      <div className="prediction-metric-grid">

        {/* Sentiment */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">Haber Sentiment</p>
          <p className="metric-value-lg" style={{ color: sentInfo.color }}>{sentInfo.text}</p>
          <div className="sentiment-bar-bg">
            <div className="sentiment-bar-center" />
            <div
              className="sentiment-bar-fill"
              style={{
                width: `${sentBarWidth}%`,
                left: sentIsPos ? '50%' : `${50 - sentBarWidth}%`,
                background: sentInfo.color,
              }}
            />
          </div>
          <p className="metric-hint" style={{ marginTop: '6px' }}>Skor: {data.sentiment_score.toFixed(3)}</p>
        </article>

        {/* VIX */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">VIX Korku Endeksi</p>
          <p className="metric-value-lg" style={{ color: vixInfo.color }}>{data.vix_close.toFixed(2)}</p>
          <p className="metric-hint" style={{ color: vixInfo.color, marginTop: '6px' }}>{vixInfo.text}</p>
        </article>

        {/* Günlük Getiri */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">Günlük Getiri</p>
          <p className="metric-value-lg" style={{ color: parseFloat(dailyRetPct) >= 0 ? '#10b981' : '#ef4444' }}>
            {parseFloat(dailyRetPct) >= 0 ? '+' : ''}{dailyRetPct}%
          </p>
          <p className="metric-hint" style={{ marginTop: '6px' }}>Son kapanış değişimi</p>
        </article>

      </div>

      {/* ── X (TWITTER) ANALİZİ BAŞLIĞI ── */}
      <p className="prediction-section-title">X (Twitter) Analizi</p>

      <div className="prediction-metric-grid">
        {/* Tweet Hacmi */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">Tweet Hacmi</p>
          <p className="metric-value-lg" style={{ color: '#8b5cf6' }}>{data.tweet_volume ? data.tweet_volume.toFixed(0) : '0'}</p>
          <p className="metric-hint" style={{ marginTop: '6px' }}>Son 24 saatteki X etkileşimi</p>
        </article>

        {/* Tweet Sentiment */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">Tweet Sentiment</p>
          <p className="metric-value-lg" style={{ color: (data.tweet_sentiment_avg || 0) > 0 ? '#10b981' : (data.tweet_sentiment_avg || 0) < 0 ? '#ef4444' : '#8b949e' }}>
            {(data.tweet_sentiment_avg || 0) > 0 ? 'Pozitif' : (data.tweet_sentiment_avg || 0) < 0 ? 'Negatif' : 'Nötr'}
          </p>
          <p className="metric-hint" style={{ marginTop: '6px' }}>Skor: {(data.tweet_sentiment_avg || 0).toFixed(3)}</p>
        </article>

        {/* Toksik Oranı */}
        <article className="prediction-metric-card">
          <p className="metric-label-sm">Toksik Tweet Oranı</p>
          <p className="metric-value-lg" style={{ color: (data.tweet_toxic_ratio || 0) > 0.1 ? '#ef4444' : '#10b981' }}>
            %{((data.tweet_toxic_ratio || 0) * 100).toFixed(1)}
          </p>
          <p className="metric-hint" style={{ marginTop: '6px' }}>Manipülasyon ve FUD riski</p>
        </article>
      </div>

      {/* ── MODEL BİLGİSİ ── */}
      <article className="model-info-card">
        <p className="prediction-label">Model Bilgisi</p>
        <div className="model-info-grid">
          {[
            ['Algoritma',       data.model_name || 'Gradient Boosting'],
            ['Eğitim Seti',     '10 hisse · 750 gün'],
            ['Test Accuracy',   '%68.00'],
            ['CV Accuracy',     '%66.00'],
            ['Özellik Sayısı',  '17 indikatör'],
            ['Model Versiyonu', 'v3.0'],
          ].map(([label, value]) => (
            <div key={label} className="model-info-item">
              <p className="model-info-item-label">{label}</p>
              <p className="model-info-item-value">{value}</p>
            </div>
          ))}
        </div>

        {/* Desteklenen hisseler */}
        <div className="supported-symbols-section">
          <p className="supported-symbols-label">Eğitilen Hisseler</p>
          <div className="supported-symbols-list">
            {SUPPORTED_SYMBOLS.map(sym => (
              <span
                key={sym}
                className={`symbol-pill ${sym === cleanSymbol ? 'active' : ''}`}
              >
                {sym}
              </span>
            ))}
          </div>
        </div>

        {/* Yenilikler */}
        <div className="model-v2-badge-row">
          {['X Sentiment', 'Toxic BERT', 'Tweet Volume', 'Sentiment Decay', 'RSI-14', 'Momentum'].map(tag => (
            <span key={tag} className="model-v2-badge">{tag}</span>
          ))}
        </div>
      </article>

    </section>
  );
}

export default PredictionPage;