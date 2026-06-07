const mongoose = require('mongoose');

const NewsSchema = new mongoose.Schema({
  // index: true ekledik -> Arama hızı 100 kat artar
  symbol: { type: String, required: true, index: true }, 
  
  category: { type: String }, 
  
  // Sıralama yaparken (sort: -1) burası kullanılıyor, index şart
  datetime: { type: Number, index: true }, 
  
  headline: { type: String, required: true },
  image: { type: String }, 
  source: { type: String },
  summary: { type: String },
  
  // URL unique (benzersiz) olmalı, aynı haber iki kere kaydedilmesin
  url: { type: String, required: true, unique: true }, 
  
  ai_summary: { type: String }, 
  event_type: { type: String, default: null } 

}, { timestamps: true });

// BİLEŞİK İNDEX (Compound Index) - İleri Seviye Optimizasyon
// "AAPL hissesinin en yeni haberlerini getir" sorgusu için muazzam hız sağlar.
NewsSchema.index({ symbol: 1, datetime: -1 });

module.exports = mongoose.model('News', NewsSchema);