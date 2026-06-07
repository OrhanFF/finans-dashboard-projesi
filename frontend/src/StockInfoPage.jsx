import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ALPHA_VANTAGE_API_KEY = 'TYB57AXXNIEYR0QS'; 
const FINNHUB_API_KEY = 'cu37qopr01qure9c388gcu37qopr01qure9c3890';

const AV_API_BASE_URL = 'https://www.alphavantage.co/query';
const FINNHUB_BASE_URL = 'https://finnhub.io/api/v1';


function CompanyProfile({ data }) {
  const description = data.Description?.length > 300 
    ? `${data.Description.substring(0, 300)}...` 
    : data.Description;

  return (
    <article className="tweet-card company-overview">
      <header className="tweet-card-header">
        <strong className="tweet-user">{data.Name} ({data.Symbol})</strong>
        <span className="tweet-date">{data.Industry}</span>
      </header>
      {description && <p className="company-description">{description}</p>}
      <div className="company-details-grid grid-col-3">
        <div><strong>Borsa:</strong> {data.Exchange}</div>
        <div><strong>Ülke:</strong> {data.Country}</div>
        <div><strong>Para Birimi:</strong> {data.Currency}</div>
      </div>
    </article>
  );
}

function FinancialMetrics({ data }) {
  const metric = data.metric || {};
  const formatValue = (value, type = 'string', locale = 'tr-TR') => {
    if (value === 'None' || value === null || value === undefined || value === 0) return 'N/A';
    if (type === 'number') return parseFloat(value).toFixed(2);
    if (type === 'currency') return (parseFloat(value) * 1000000).toLocaleString(locale) + ' USD';
    return value;
  };

  return (
    <article className="tweet-card company-overview">
      <header className="tweet-card-header">
        <strong className="tweet-user">{data.symbol} - Finansal Rasyolar</strong>
      </header>
      <div className="company-details-grid grid-col-4">
        <div>
            <strong>Piyasa Değeri:</strong> 
            {formatValue(metric.marketCapitalization, 'currency')}
        </div>
        <div>
            <strong>F/K Oranı:</strong> 
            {formatValue(metric.peNormalizedAnnual, 'number')}
        </div>
        <div>
            <strong>EPS (Yıllık):</strong> 
            {formatValue(metric.epsNormalizedAnnual, 'number')}
        </div>
        <div>
            <strong>Temettü Verimi:</strong> 
            {formatValue(metric.dividendYieldIndicatedAnnual, 'number')}%
        </div>
        <div>
            <strong>52 Haftalık Zirve:</strong> 
            {formatValue(metric['52WeekHigh'], 'number')}
        </div>
        <div>
            <strong>52 Haftalık Dip:</strong> 
            {formatValue(metric['52WeekLow'], 'number')}
        </div>
         <div>
            <strong>Beta:</strong> 
            {formatValue(metric.beta, 'number')}
        </div>
        <div>
            <strong>H. Fiyat (Ort):</strong> 
            {formatValue(metric.targetPrice, 'number')}
        </div>
      </div>
    </article>
  );
}

function QuoteDetails({ data }) {
  const price = parseFloat(data.c);
  const change = parseFloat(data.d);
  const changePercent = parseFloat(data.dp);
  const isPositive = change >= 0;

  return (
    <article className="tweet-card quote-details">
      <header className="tweet-card-header">
        <strong className="tweet-user">Anlık Fiyat Bilgisi</strong>
      </header>
      <div className="quote-main-price">
        {price.toFixed(2)} <small>USD</small>
        <span className="quote-change" style={{ color: isPositive ? 'var(--success-color)' : 'var(--error-color)' }}>
          {isPositive ? '▲' : '▼'} {change.toFixed(2)} ({changePercent.toFixed(2)}%)
        </span>
      </div>
      <div className="quote-details-grid">
        <div className="quote-detail-item">
          <strong>Açılış:</strong> {parseFloat(data.o).toFixed(2)}
        </div>
        <div className="quote-detail-item">
          <strong>Günlük Yüksek:</strong> {parseFloat(data.h).toFixed(2)}
        </div>
        <div className="quote-detail-item">
          <strong>Günlük Düşük:</strong> {parseFloat(data.l).toFixed(2)}
        </div>
        <div className="quote-detail-item">
          <strong>Önceki Kapanış:</strong> {parseFloat(data.pc).toFixed(2)}
        </div>
      </div>
    </article>
  );
}

function StockChart({ chartUrl, symbol }) {
  return (
    <article className="tweet-card stock-chart">
      <header className="tweet-card-header">
        <strong className="tweet-user">{symbol} - 30 Günlük Fiyat Grafiği</strong>
      </header>
      <div className="chart-container">
        <img src={chartUrl} alt={`${symbol} hisse senedi grafiği`} style={{ width: '100%', height: 'auto' }} />
      </div>
    </article>
  );
}

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));


function StockInfoPage({ searchTerm }) {
  
  const [overviewData, setOverviewData] = useState(null);
  const [quoteData, setQuoteData] = useState(null);
  const [metricData, setMetricData] = useState(null);
  const [chartUrl, setChartUrl] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const cleanSearchTerm = searchTerm ? searchTerm.trim().replace(/^#/, '').toUpperCase() : '';

    if (!cleanSearchTerm) {
      setError("Lütfen bir hisse senedi sembolü arayın (örn: AAPL, MSFT).");
      setOverviewData(null);
      setQuoteData(null);
      setMetricData(null);
      setChartUrl(null);
      return;
    }

    const fetchStockData = async () => {
      setLoading(true);
      setError(null);
      setOverviewData(null);
      setQuoteData(null);
      setMetricData(null);
      setChartUrl(null);

      const paramsAVOverview = { function: 'OVERVIEW', symbol: cleanSearchTerm, apikey: ALPHA_VANTAGE_API_KEY };
      const paramsAVChart = { function: 'TIME_SERIES_DAILY', symbol: cleanSearchTerm, outputsize: 'compact', apikey: ALPHA_VANTAGE_API_KEY };
      const paramsFinnhubQuote = { symbol: cleanSearchTerm, token: FINNHUB_API_KEY };
      const paramsFinnhubMetric = { symbol: cleanSearchTerm, metric: 'all', token: FINNHUB_API_KEY };

      let hasData = false;

      try {
        const [quoteResult, metricResult] = await Promise.allSettled([
          axios.get(`${FINNHUB_BASE_URL}/quote`, { params: paramsFinnhubQuote }),
          axios.get(`${FINNHUB_BASE_URL}/stock/metric`, { params: paramsFinnhubMetric })
        ]);

        if (quoteResult.status === 'fulfilled' && quoteResult.value.data && quoteResult.value.data.c !== 0) {
          setQuoteData(quoteResult.value.data);
          hasData = true;
        }
        if (metricResult.status === 'fulfilled' && metricResult.value.data && Object.keys(metricResult.value.data.metric).length > 0) {
          setMetricData(metricResult.value.data);
          hasData = true;
        }

        try {
          const overviewResponse = await axios.get(AV_API_BASE_URL, { params: paramsAVOverview });
          if (overviewResponse.data && !overviewResponse.data.Note) {
            if (Object.keys(overviewResponse.data).length > 0 && overviewResponse.data.Symbol) {
              setOverviewData(overviewResponse.data);
              hasData = true;
            }
          } else if (overviewResponse.data.Note) {
            console.warn("AV Limiti (Overview):", overviewResponse.data.Note);
            setError("Alpha Vantage API limitine takıldınız. Lütfen 1 dakika bekleyin.");
            setLoading(false); 
            return; 
          }
        } catch (avOverviewError) {
          console.error("AV Overview Hatası:", avOverviewError);
          setError("Şirket profili yüklenemedi.");
        }
        
        await wait(1200); 

        try {
          const chartResponse = await axios.get(AV_API_BASE_URL, { params: paramsAVChart });
          if (chartResponse.data && chartResponse.data['Time Series (Daily)']) {
            const timeSeries = chartResponse.data['Time Series (Daily)'];
            const entries = Object.entries(timeSeries).slice(0, 30).reverse();
            const labels = entries.map(([date]) => date.slice(5));
            const dataPoints = entries.map(([, values]) => parseFloat(values['4. close']));
            const chartDataStr = dataPoints.join(',');
            const min = Math.min(...dataPoints);
            const max = Math.max(...dataPoints);
            const chartLabelsStr = labels.filter((_, i) => i % 6 === 0).join('|');
            const axisColor = "f5f5f5";
            const url = `https://image-charts.com/chart?cht=lc&chs=700x300&chd=t:${chartDataStr}&chds=${min},${max}&chxt=x,y&chxl=0:|${chartLabelsStr}&chco=8a4cff&chls=2.0&chf=bg,s,1f2329&chxs=0,${axisColor}|1,${axisColor}`;
            setChartUrl(url); 
            hasData = true;
          } else if (chartResponse.data.Note) {
            console.warn("AV Limiti (Chart):", chartResponse.data.Note);
            if (!error) {
                 setError("Alpha Vantage API limiti nedeniyle grafik yüklenemedi. Diğer veriler gösteriliyor.");
            }
          }
        } catch (avChartError) {
          console.error("AV Chart Hatası:", avChartError);
        }

        if (!hasData && !error) {
          setError(`'${cleanSearchTerm}' için hisse senedi bilgisi bulunamadı.`);
        }

      } catch (err) {
        console.error("Genel Hata:", err);
        setError("API'lere bağlanırken genel bir hata oluştu.");
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
    
  }, [searchTerm]);

  if (loading) {
    return <div className='loading-indicator' role="status"><p>Hisse bilgileri yükleniyor...</p></div>;
  }
  
  if (error) {
    return <div className='no-results' role="alert"><p>{error}</p></div>;
  }

  return (
    <div className="stock-info-container">
      
      {overviewData && <CompanyProfile data={overviewData} />}
      
      {quoteData && <QuoteDetails data={quoteData} />}

      {metricData && <FinancialMetrics data={metricData} />}
      
      {chartUrl && <StockChart chartUrl={chartUrl} symbol={searchTerm.trim().replace(/^#/, '').toUpperCase()} />}
      
      {!overviewData && !quoteData && !chartUrl && !metricData && !loading && !error && (
          <div className='no-results'>
              <p>"{searchTerm}" için veri bulunamadı.</p>
          </div>
      )}
    </div>
  );
}

export default StockInfoPage;