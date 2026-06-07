import React, { useState, useEffect } from 'react';
import axios from 'axios';

const FINNHUB_API_KEY = 'cu37qopr01qure9c388gcu37qopr01qure9c3890';
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';

function AnalystRecommendations({ data }) {
  const { strongBuy, buy, hold, sell, strongSell, period } = data;
  
  const totalAnalysts = strongBuy + buy + hold + sell + strongSell;

  if (totalAnalysts === 0) {
    return (
        <div className='no-results' role="alert">
            <p>Bu hisse senedi için analist tavsiyesi bulunamadı.</p>
        </div>
    );
  }

  const categories = [
    { label: 'Güçlü Al', value: strongBuy, className: 'bar-strong-buy' },
    { label: 'Al', value: buy, className: 'bar-buy' },
    { label: 'Tut', value: hold, className: 'bar-hold' },
    { label: 'Sat', value: sell, className: 'bar-sell' },
    { label: 'Güçlü Sat', value: strongSell, className: 'bar-strong-sell' }
  ];

  return (
    <article className="tweet-card recommendation-card">
      <header className="tweet-card-header">
        <strong className="tweet-user">Analist Tavsiyeleri</strong>
        <span className="tweet-date">Son Güncelleme: {period}</span>
      </header>
      
      <p className="recommendation-total">
        Toplam {totalAnalysts} Analist Değerlendirmesi
      </p>
      
      <div className="recommendation-chart">
        {categories.map(cat => (
          <div className="chart-row" key={cat.label}>
            <span className="chart-label">{cat.label}</span>
            <div className="chart-bar-container">
              <div 
                className={`chart-bar ${cat.className}`}
                style={{ width: `${(cat.value / totalAnalysts) * 100}%` }}
              ></div>
            </div>
            <span className="chart-value">{cat.value}</span>
          </div>
        ))}
      </div>
    </article>
  );
}


function AnalystPage({ searchTerm }) {

  const [recommendationData, setRecommendationData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const cleanSearchTerm = searchTerm ? searchTerm.trim().replace(/^#/, '') : '';

    if (!cleanSearchTerm) {
      setRecommendationData(null);
      return;
    }

    const fetchRecommendations = async () => {
      setLoading(true);
      setError(null);
      setRecommendationData(null);

      const params = {
        symbol: cleanSearchTerm,
        token: FINNHUB_API_KEY
      };

      try {
        const response = await axios.get(`${FINNHUB_BASE_URL}/stock/recommendation`, { params });

        if (response.data && response.data.length > 0) {
          setRecommendationData(response.data[0]);
        } else {
          setError(`'${cleanSearchTerm}' için analist tavsiyesi bulunamadı.`);
        }
      
      } catch (err) {
        console.error(err);
        if (err.response && err.response.status === 401) {
             setError("Finnhub API anahtarı geçersiz veya eksik.");
        } else if (err.response && err.response.status === 429) {
             setError("Finnhub API limitine ulaştınız. Lütfen bir süre bekleyin.");
        } else {
             setError("Tavsiye verileri çekilirken bir hata oluştu.");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
    
  }, [searchTerm]);

  if (loading) {
    return <div className='loading-indicator' role="status"><p>Analist tavsiyeleri yükleniyor...</p></div>;
  }
  
  if (error) {
    return <div className='no-results' role="alert"><p>{error}</p></div>;
  }

  return (
    <section className="results-list">
        {recommendationData && <AnalystRecommendations data={recommendationData} />}
    </section>
  );
}

export default AnalystPage;