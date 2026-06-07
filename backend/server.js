require('dotenv').config();
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const connectDB = require('./config/db');
const News = require('./models/News');
const fetchAndSaveData = require('./jobs/dataFetcher');

const app = express();
const PORT = process.env.PORT || 3001;

connectDB();
app.use(cors());
app.use(express.json());

const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY;
const FINNHUB_BASE_URL = "https://finnhub.io/api/v1";
const SCRAPER_API_URL  = "http://127.0.0.1:8001";
const ML_SERVICE_URL   = "http://127.0.0.1:8002";
const HF_TOKEN         = process.env.HF_TOKEN;

const formatDate = (date) => date.toISOString().split('T')[0];

app.get('/api/baslat', (req, res) => {
  fetchAndSaveData();
  res.send(`<h1>İşlem Başlatıldı!</h1>`);
});

app.get('/api/temizle', async (req, res) => {
  try {
    await News.deleteMany({});
    res.send(`<h1>Veritabanı Temizlendi!</h1>`);
  } catch (error) {
    res.status(500).send('Hata oluştu.');
  }
});

app.post('/api/summarize', async (req, res) => {
  const { text } = req.body;
  if (!text) return res.status(400).json({ error: 'Metin yok.' });
  const cleanText = text.replace(/(\r\n|\n|\r)/gm, " ");
  try {
    const response = await axios.post(
      "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6",
      { inputs: cleanText },
      { headers: { Authorization: `Bearer ${HF_TOKEN}` }, timeout: 8000 }
    );
    if (response.data && response.data[0]) return res.json(response.data);
  } catch (error) {
    console.log("⚠️ AI Meşgul, manuel özet devreye giriyor.");
  }
  const manuelOzet = cleanText.length > 200 ? cleanText.substring(0, 200) + "..." : cleanText;
  res.json([{ summary_text: manuelOzet }]);
});

app.get('/api/search', async (req, res) => {
  const { q } = req.query;
  if (!q) return res.status(400).json({ error: 'Arama terimi gerekli' });
  try {
    const response = await axios.get(`${SCRAPER_API_URL}/scrape`, { params: { ticker: q } });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: 'Scraper servisine ulaşılamadı' });
  }
});

app.get('/api/news-intensity', async (req, res) => {
  const { symbol, range } = req.query;
  const to = new Date();
  let from = new Date();
  let stats = {};

  if (range === '1day') {
    from.setHours(to.getHours() - 24);
    for (let i = 23; i >= 0; i--) {
      const d = new Date();
      d.setHours(d.getHours() - i);
      stats[`${d.getHours()}:00`] = 0;
    }
  } else {
    from.setDate(to.getDate() - 7);
    for (let i = 6; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      stats[d.toLocaleDateString('tr-TR', { weekday: 'short' })] = 0;
    }
  }

  try {
    const response = await axios.get(`${FINNHUB_BASE_URL}/company-news`, {
      params: { symbol, from: formatDate(from), to: formatDate(to), token: FINNHUB_API_KEY }
    });
    response.data.forEach(news => {
      const date = new Date(news.datetime * 1000);
      const key = range === '1day'
        ? `${date.getHours()}:00`
        : date.toLocaleDateString('tr-TR', { weekday: 'short' });
      if (stats.hasOwnProperty(key)) stats[key]++;
    });
    res.json(Object.keys(stats).map(label => ({ label, count: stats[label] })));
  } catch (error) {
    res.status(500).json({ error: 'Veri alınamadı' });
  }
});

app.get('/api/db-news', async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error: 'Sembol gerekli' });
  try {
    const news = await News.find({ symbol: symbol.toUpperCase() }).sort({ datetime: -1 });
    res.json(news);
  } catch (error) {
    res.status(500).json({ error: 'Veritabanı hatası' });
  }
});

// ─────────────────────────────────────────────
// YENİ: ML TAHMİN ENDPOINT'İ
// ─────────────────────────────────────────────

app.get('/api/predict', async (req, res) => {
  const { symbol } = req.query;
  if (!symbol) return res.status(400).json({ error: 'Sembol gerekli' });

  try {
    const response = await axios.get(`${ML_SERVICE_URL}/predict`, {
      params: { symbol: symbol.toUpperCase() },
      timeout: 30000
    });
    res.json(response.data);
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({
        error: 'ML servisi çalışmıyor.',
        detail: 'Şu komutu çalıştırın: uvicorn ml_service:app --port 8002'
      });
    }
    console.error("ML Servis Hatası:", error.message);
    res.status(500).json({ error: 'Tahmin alınamadı.' });
  }
});

app.listen(PORT, () => {
  console.log(`\n🚀 Server:    http://localhost:${PORT}`);
  console.log(`🤖 Tahmin:    http://localhost:${PORT}/api/predict?symbol=AAPL`);
  console.log(`🧹 Temizle:   http://localhost:${PORT}/api/temizle`);
  console.log(`👉 Başlat:    http://localhost:${PORT}/api/baslat`);
});