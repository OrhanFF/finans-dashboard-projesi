import React, { useState, useEffect } from 'react';
import axios from 'axios';

function NewsIntensity({ symbol }) {
  const [range, setRange] = useState('1week'); 
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!symbol) return;
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await axios.get(`http://localhost:3001/api/news-intensity`, {
          params: { symbol, range }
        });
        setData(res.data);
      } catch (err) {
        console.error("Hata:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [symbol, range]);

  const getChartUrl = () => {
    if (data.length === 0) return "";
    
    const counts = data.map(d => d.count).join(',');
    
    // --- GÖRSEL TEMİZLİK ---
    const labels = data.map((d, index) => {
        // Eğer 24 saat modundaysak
        if (range === '1day') {
            // Sadece her 3. saati yaz, diğerlerini boş bırak (Kalabalığı önler)
            // Örn: 12:00, 15:00, 18:00 görünür
            if (index % 3 !== 0) return ''; 
            
            // ":00" kısmını at, sadece saati yaz (14:00 -> 14)
            return d.label.split(':')[0];
        }
        // Hafta modundaysak hepsini göster (zaten 7 tane var)
        return d.label;
    }).join('|'); 
    
    const max = Math.max(...data.map(d => d.count)) + 1; // Tavan değeri biraz yükselt

    // chbh=a,10,5 -> Çubukları otomatik genişlet ama aralarında 5px boşluk bırak
    return `https://image-charts.com/chart?cht=bvg&chs=700x200&chd=t:${counts}&chds=0,${max}&chxt=x,y&chxl=0:|${labels}&chco=8a4cff&chf=bg,s,1f2329&chxs=0,f5f5f5,11|1,f5f5f5,10&chbh=a,10,5&chg=0,20`; 
  };

  return (
    <div className="tweet-card news-intensity-card" style={{ marginBottom: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
        <strong className="tweet-user">📈 Haber Yoğunluğu</strong>
        <div className="range-selector">
          <button onClick={() => setRange('1day')} className={`mini-btn ${range === '1day' ? 'active' : ''}`}>24 Saat</button>
          <button onClick={() => setRange('1week')} className={`mini-btn ${range === '1week' ? 'active' : ''}`}>1 Hafta</button>
        </div>
      </div>
      
      <div style={{ minHeight: '180px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1f2329', borderRadius: '8px' }}>
        {loading ? (
          <p style={{color: '#8a4cff'}}>Veriler analiz ediliyor...</p>
        ) : data.length > 0 ? (
          <img src={getChartUrl()} alt="Haber Yoğunluğu Grafiği" style={{ width: '100%', height: 'auto' }} />
        ) : (
          <p style={{color: '#8899a6'}}>Seçilen aralıkta henüz haber girişi yok.</p>
        )}
      </div>
    </div>
  );
}

export default NewsIntensity;