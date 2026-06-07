import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import NewsCard from './NewsCard.jsx';
import NewsIntensity from './NewsIntensity.jsx';

const BACKEND_URL = 'http://localhost:3001/api';

const EVENT_OPTIONS = [
  { value: 'all',   label: '🔍 Tümü' },
  { value: '📊 Bilanço / Rapor',         label: '📊 Bilanço / Rapor' },
  { value: '👔 Yönetim / CEO',            label: '👔 Yönetim / CEO' },
  { value: '💰 Temettü / Geri Alım',      label: '💰 Temettü / Geri Alım' },
  { value: '🤝 Birleşme & Satın Alma',    label: '🤝 Birleşme & Satın Alma' },
  { value: '⚖️ Yasal / Dava',             label: '⚖️ Yasal / Dava' },
  { value: '🚀 Yeni Ürün / Duyuru',       label: '🚀 Yeni Ürün / Duyuru' },
  { value: '📈 Analist Tavsiyesi',         label: '📈 Analist Tavsiyesi' },
  { value: '📉 Fiyat Hareketi',           label: '📉 Fiyat Hareketi' },
  { value: 'none', label: '🏷️ Etiketsiz' },
];

function NewsPage({ searchTerm }) {
  const [news, setNews]       = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const [eventFilter, setEventFilter] = useState('all');

  const cleanSearchTerm = searchTerm
    ? searchTerm.trim().replace(/^#/, '').toUpperCase()
    : '';

  useEffect(() => {
    if (!cleanSearchTerm) { setNews([]); return; }

    const fetchDbNews = async () => {
      setLoading(true);
      setError(null);
      setNews([]);
      setEventFilter('all');

      try {
        const response = await axios.get(`${BACKEND_URL}/db-news`, {
          params: { symbol: cleanSearchTerm }
        });

        if (response.data && response.data.length > 0) {
          setNews(response.data);
        } else {
          setError(`'${cleanSearchTerm}' için veritabanında kayıtlı haber bulunamadı. Backend taraması yaptınız mı?`);
        }
      } catch (err) {
        console.error("Haber çekme hatası:", err);
        setError("Haberler sunucudan çekilirken bir hata oluştu. Server çalışıyor mu?");
      } finally {
        setLoading(false);
      }
    };

    fetchDbNews();
  }, [cleanSearchTerm]);

  const filteredNews = useMemo(() => {
    if (eventFilter === 'all') return news;
    if (eventFilter === 'none') return news.filter(a => !a.event_type);
    return news.filter(a => a.event_type === eventFilter);
  }, [news, eventFilter]);

  const eventCounts = useMemo(() => {
    const counts = { all: news.length, none: 0 };
    news.forEach(a => {
      if (!a.event_type) {
        counts['none'] = (counts['none'] || 0) + 1;
      } else {
        counts[a.event_type] = (counts[a.event_type] || 0) + 1;
      }
    });
    return counts;
  }, [news]);

  if (loading) {
    return <div className='loading-indicator' role="status"><p>Haberler veritabanından yükleniyor...</p></div>;
  }

  return (
    <section className="results-list">
      {cleanSearchTerm && <NewsIntensity symbol={cleanSearchTerm} />}

      {error && <div className='no-results' role="alert"><p>{error}</p></div>}

      {!error && news.length > 0 && (
        <>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '12px',
            marginTop: '20px',
            marginBottom: '15px',
            borderBottom: '1px solid #3a414b',
            paddingBottom: '12px'
          }}>
            <h3 style={{ margin: 0, fontSize: '1.2rem' }}>
              {cleanSearchTerm} İçin Son Haberler & Olaylar
            </h3>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '0.85rem', color: '#94a3b8', whiteSpace: 'nowrap' }}>
                Etikete Göre:
              </label>
              <select
                value={eventFilter}
                onChange={e => setEventFilter(e.target.value)}
                className="mini-select"
                style={{ minWidth: '200px' }}
              >
                {EVENT_OPTIONS.map(opt => {
                  const count = eventCounts[opt.value] ?? 0;
                  if (opt.value !== 'all' && opt.value !== 'none' && count === 0) return null;
                  return (
                    <option key={opt.value} value={opt.value}>
                      {opt.label} ({opt.value === 'all' ? news.length : (eventCounts[opt.value] ?? 0)})
                    </option>
                  );
                })}
              </select>

              {eventFilter !== 'all' && (
                <button
                  onClick={() => setEventFilter('all')}
                  style={{
                    background: 'none',
                    border: '1px solid #8a4cff',
                    color: '#a78bfa',
                    borderRadius: '12px',
                    padding: '3px 10px',
                    fontSize: '0.8rem',
                    cursor: 'pointer',
                  }}
                >
                  ✕ Filtreyi Kaldır
                </button>
              )}
            </div>
          </div>

          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginBottom: '12px', marginTop: 0 }}>
            {filteredNews.length} haber gösteriliyor
            {eventFilter !== 'all' && ` · "${EVENT_OPTIONS.find(o => o.value === eventFilter)?.label}" filtresi aktif`}
          </p>

          {filteredNews.length > 0 ? (
            <div className="news-grid">
              {filteredNews.map(article => (
                <NewsCard key={article._id || article.url} article={article} />
              ))}
            </div>
          ) : (
            <div className='no-results'>
              <p>Bu etikette haber bulunamadı.</p>
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default NewsPage;