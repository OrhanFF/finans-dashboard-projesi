import React, { useState } from 'react';
import axios from 'axios';

const formatUnixTimestamp = (timestamp) => {
  const date = new Date(timestamp * 1000); 
  return date.toLocaleDateString('tr-TR', {
    year: 'numeric', month: 'long', day: 'numeric',
  });
};

function NewsCard({ article }) {
  const [aiSummary, setAiSummary] = useState(article.ai_summary || ""); 
  const [isSummarizing, setIsSummarizing] = useState(false);

  const content = article.summary || article.headline;
  const displaySummary = content.length > 200 ? `${content.substring(0, 200)}...` : content;

  // Event Rengi Seçici
  const getEventColor = (eventType) => {
    if (!eventType) return '#8a4cff';
    if (eventType.includes("Bilanço")) return '#10b981'; // Yeşil
    if (eventType.includes("Dava")) return '#ff5555'; // Kırmızı
    if (eventType.includes("CEO")) return '#f59e0b'; // Turuncu
    if (eventType.includes("Birleşme")) return '#3b82f6'; // Mavi
    if (eventType.includes("Analist")) return '#8b5cf6'; // Mor
    if (eventType.includes("Fiyat")) return '#0ea5e9'; // Açık Mavi
    if (eventType.includes("Ürün")) return '#ec4899'; // Pembe
    return '#64748b'; // Gri
  };

  const eventColor = getEventColor(article.event_type);

  const handleSummarize = async (e) => {
    e.preventDefault();
    e.stopPropagation(); 
    
    if (aiSummary || isSummarizing) return;

    setIsSummarizing(true);
    try {
      const response = await axios.post("http://localhost:3001/api/summarize", {
        text: content
      });

      if (response.data && response.data[0]) {
        setAiSummary(response.data[0].summary_text);
      }
    } catch (err) {
      setAiSummary("Özet alınamadı.");
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <div className="news-card-link" style={{ marginBottom: '20px' }}>
      <article className="tweet-card news-card" style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        
        {/* ÜST KISIM: Resim ve İçerik */}
        <div style={{ display: 'flex', gap: '15px', alignItems: 'flex-start' }}>
            {article.image && (
            <div className="news-image-container" style={{ width: '100px', height: '100px', flexShrink: 0 }}>
                <img src={article.image} alt="News" style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '8px' }} />
            </div>
            )}
            
            <div className="news-content-wrapper" style={{ flexGrow: 1 }}>
                <header className="tweet-card-header" style={{ marginBottom: '8px' }}>
                    <div style={{display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap'}}>
                        <strong className="tweet-user news-source" style={{ color: '#0ea5e9' }}>{article.source}</strong>
                        
                        {/* EVENT ROZETİ */}
                        {article.event_type && (
                            <span style={{
                                backgroundColor: `${eventColor}22`, 
                                color: eventColor,
                                border: `1px solid ${eventColor}`,
                                fontSize: '0.7rem',
                                padding: '2px 8px',
                                borderRadius: '12px',
                                fontWeight: 'bold',
                                whiteSpace: 'nowrap'
                            }}>
                                {article.event_type}
                            </span>
                        )}
                    </div>
                    <span className="tweet-date news-date" style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                        {formatUnixTimestamp(article.datetime)}
                    </span>
                </header>
                
                <h3 className="news-headline" style={{ margin: '0 0 8px 0', fontSize: '1.1rem', lineHeight: '1.4' }}>
                    <a href={article.url} target="_blank" rel="noopener noreferrer" style={{color:'inherit', textDecoration:'none'}}>
                    {article.headline}
                    </a>
                </h3>
                
                <p className="news-summary" style={{ fontSize: '0.9rem', color: '#cbd5e1', margin: 0 }}>
                    {displaySummary}
                </p>
            </div>
        </div>

        {/* ORTA KISIM: AI ÖZETİ */}
        {aiSummary && (
            <div className="ai-summary-box" style={{ 
                marginTop: '5px', 
                padding: '15px', 
                backgroundColor: '#111827', 
                borderLeft: '4px solid #8a4cff', 
                borderRadius: '6px', 
                fontSize: '0.95rem',
                color: '#e5e7eb', 
                lineHeight: '1.6',
                boxShadow: 'inset 0 0 10px rgba(0,0,0,0.5)' 
            }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', color: '#a78bfa' }}>
                    <span>✨</span>
                    <strong>Yapay Zeka Özeti:</strong>
                </div>
                <p style={{ margin: 0 }}>
                    {aiSummary}
                </p>
            </div>
        )}

        {/* ALT KISIM: Footer (DÜZELTİLDİ: width 100%) */}
        <div className="news-footer" style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            width: '100%', // BU SATIR EKLENDİ - ARTIK TAM OTURACAK
            marginTop: '10px',
            paddingTop: '15px',
            borderTop: '1px solid #374151' 
        }}>
            {/* SOL KÖŞE */}
            <a href={article.url} target="_blank" rel="noopener noreferrer" 
               className="read-more-text" 
               style={{ 
                   color: '#8a4cff', 
                   textDecoration: 'none', 
                   fontSize: '0.9rem', 
                   fontWeight: 'bold',
                   transition: 'color 0.2s'
               }}
               onMouseOver={(e) => e.target.style.color = '#a78bfa'}
               onMouseOut={(e) => e.target.style.color = '#8a4cff'}
            >
              Haberi Oku →
            </a>
            
            {/* SAĞ KÖŞE */}
            <button 
              className="summarize-btn"
              onClick={handleSummarize}
              disabled={isSummarizing || (aiSummary !== "" && aiSummary !== article.ai_summary)}
              style={{
                  padding: '8px 20px',
                  backgroundColor: isSummarizing ? '#4b5563' : aiSummary ? '#10b981' : '#8a4cff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '20px', 
                  cursor: isSummarizing ? 'not-allowed' : 'pointer',
                  fontSize: '0.85rem',
                  fontWeight: '600',
                  boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
                  transition: 'transform 0.2s'
              }}
              onMouseDown={(e) => e.target.style.transform = 'scale(0.95)'}
              onMouseUp={(e) => e.target.style.transform = 'scale(1)'}
            >
              {isSummarizing ? '⏳ İşleniyor...' : aiSummary ? '✓ Hazır' : '✨ Özetle'}
            </button>
        </div>

      </article>
    </div>
  );
}

export default NewsCard;