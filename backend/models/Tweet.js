const mongoose = require('mongoose');

const TweetSchema = new mongoose.Schema({

  // Hisse sembolü — AAPL, TSLA vs.
  symbol: { type: String, required: true, index: true },

  // Tweet içeriği
  user:      { type: String, required: true },
  tweet:     { type: String, required: true },
  timestamp: { type: String },

  // Analiz sonuçları (tweet_analyzer.py tarafından doldurulur)
  is_toxic:       { type: Boolean, default: null },
  toxicity_score: { type: Number,  default: null },
  sentiment:      { type: String,  default: null }, // "positive" | "negative" | "neutral"
  sentiment_score:{ type: Number,  default: null },
  is_analyzed:    { type: Boolean, default: false, index: true },

}, { timestamps: true });

// Bileşik index: sembol + tarih sıralaması için
TweetSchema.index({ symbol: 1, createdAt: -1 });

// Aynı tweet iki kere kaydedilmesin
TweetSchema.index({ user: 1, tweet: 1 }, { unique: true });

module.exports = mongoose.model('Tweet', TweetSchema);