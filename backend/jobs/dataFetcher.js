const axios = require('axios');
const News = require('../models/News'); // Bir üst klasöre çıkıp models'e girer

// --- AYARLAR ---
const FINNHUB_API_KEY = "cu37qopr01qure9c388gcu37qopr01qure9c3890";
const FINNHUB_BASE_URL = "https://finnhub.io/api/v1";

const WATCHLIST = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
             'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL'
];

const formatDate = (date) => date.toISOString().split('T')[0];
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// --- 🔥 GÜNCEL EVENT DETECTOR (DAHA HASSAS) ---
function detectEvent(headline, summary) {
    const text = `${headline} ${summary}`.toLowerCase();

    // 1. YÖNETİM & CEO
    if (text.includes("ceo") || text.includes("cfo") || text.includes("resigns") || text.includes("appointed") || text.includes("steps down") || text.includes("executive")) {
        return "👔 Yönetim / CEO";
    }

    // 2. BİLANÇO & RAKAMLAR
    if (text.includes("earnings") || text.includes("revenue") || text.includes("profit") || text.includes("quarterly") || text.includes("results") || text.includes("forecast") || text.includes("sales")) {
        return "📊 Bilanço / Rapor";
    }

    // 3. TEMETTÜ & GERİ ALIM
    if (text.includes("dividend") || text.includes("buyback") || text.includes("repurchase")) {
        return "💰 Temettü / Geri Alım";
    }

    // 4. BİRLEŞME & SATIN ALMA
    if (text.includes("acquisition") || text.includes("merger") || text.includes("acquire") || text.includes("deal") || text.includes("bought")) {
        return "🤝 Birleşme & Satın Alma";
    }

    // 5. YASAL & DAVA
    if (text.includes("lawsuit") || text.includes("sues") || text.includes("court") || text.includes("legal") || text.includes("settlement") || text.includes("antitrust")) {
        return "⚖️ Yasal / Dava";
    }

    // 6. YENİ ÜRÜN & TEKNOLOJİ
    if (text.includes("launch") || text.includes("unveil") || text.includes("new product") || text.includes("release") || text.includes("announces")) {
        return "🚀 Yeni Ürün / Duyuru";
    }
    
    // 7. ANALİST TAVSİYESİ
    if (text.includes("upgrade") || text.includes("downgrade") || text.includes("price target") || text.includes("rating") || text.includes("buy rating") || text.includes("sell rating")) {
        return "📈 Analist Tavsiyesi";
    }

    // 8. (YENİ) FİYAT HAREKETİ - Bunu ekledik ki boş geçmesin
    if (text.includes("rise") || text.includes("fall") || text.includes("jump") || text.includes("drop") || text.includes("surge") || text.includes("plunge") || text.includes("gain") || text.includes("loss") || text.includes("high") || text.includes("low")) {
        return "📉 Fiyat Hareketi";
    }

    return null; 
}

// --- ANA FONKSİYON ---
const fetchAndSaveData = async () => {
  console.log('🔄 [JOB] Hassas Event Detector Devrede...', new Date().toLocaleString());

  const toDate = new Date();
  const fromDate = new Date();
  fromDate.setDate(toDate.getDate() - 7); 

  for (const symbol of WATCHLIST) {
    try {
      await sleep(1500); // Biraz hızlandırdım

      const response = await axios.get(`${FINNHUB_BASE_URL}/company-news`, {
          params: { 
              symbol: symbol, 
              from: formatDate(fromDate), 
              to: formatDate(toDate), 
              token: FINNHUB_API_KEY 
          }
      });

      const newsList = response.data;
      let savedCount = 0;

      for (const newsItem of newsList) {
        try {
            if (!newsItem.headline || !newsItem.url) continue;

            const detectedEvent = detectEvent(newsItem.headline, newsItem.summary);
            
            // --- LOGLAMA (Sistemin çalıştığını buradan anlarsın) ---
            if (detectedEvent) {
                console.log(`   🔥 ${symbol} Olayı: ${detectedEvent}`);
            }
            // -----------------------------------------------------

            const newNews = new News({
                symbol: symbol,
                category: newsItem.category,
                datetime: newsItem.datetime,
                headline: newsItem.headline,
                image: newsItem.image,
                source: newsItem.source,
                summary: newsItem.summary,
                url: newsItem.url,
                event_type: detectedEvent 
            });

            await newNews.save();
            savedCount++;
        } catch (error) {
            if (error.code !== 11000) console.error(`Hata: ${error.message}`);
        }
      }

      if(savedCount > 0) console.log(`[+] ${symbol}: ${savedCount} haber işlendi.`);
      
    } catch (error) {
       console.error(`❌ [API HATA] ${symbol}:`, error.message);
    }
  }
  console.log('✅ [JOB] Tüm işlemler tamamlandı.');
};

module.exports = fetchAndSaveData;