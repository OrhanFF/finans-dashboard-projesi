"""
tweet_analyzer.py
-----------------
MongoDB'deki ham tweetleri batch halinde analiz eder.
  - toxic-bert  → is_toxic, toxicity_score
  - FinBERT     → sentiment, sentiment_score

Kullanım:
  python tweet_analyzer.py                    # Tüm analiz edilmemiş tweetler
  python tweet_analyzer.py --symbol AAPL      # Sadece bir hisse
  python tweet_analyzer.py --batch 64         # Batch boyutu
"""

import argparse
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from transformers import pipeline

load_dotenv()

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017/finans")
DEFAULT_BATCH    = 64
TOXICITY_THRESHOLD = 0.60
TOXIC_LABELS     = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}


# ─────────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────────

def get_collection():
    client = MongoClient(MONGO_URI)
    db     = client.get_default_database()
    return db["tweets"]


# ─────────────────────────────────────────────
# ANALİZ FONKSİYONLARI
# ─────────────────────────────────────────────

def run_toxicity(analyzer, texts: list) -> list:
    """
    Batch toxicity analizi.
    Her metin için {"is_toxic": bool, "toxicity_score": float} döner.
    """
    results = []
    try:
        outputs = analyzer(texts)
        for output in outputs:
            toxic_scores = [
                r["score"]
                for r in output
                if r["label"].lower() in TOXIC_LABELS
            ]
            max_score = max(toxic_scores) if toxic_scores else 0.0
            results.append({
                "is_toxic":       max_score >= TOXICITY_THRESHOLD,
                "toxicity_score": round(max_score, 4)
            })
    except Exception as e:
        print(f"   ⚠️ Toxicity hatası: {e}")
        results = [{"is_toxic": False, "toxicity_score": 0.0}] * len(texts)
    return results


def run_sentiment(analyzer, texts: list) -> list:
    """
    Batch FinBERT sentiment analizi.
    Her metin için {"sentiment": str, "sentiment_score": float} döner.
    """
    results = []
    try:
        outputs = analyzer(texts)
        for output in outputs:
            label = output["label"]   # positive / negative / neutral
            score = output["score"]

            if label == "positive":
                sent_score =  round(score, 4)
            elif label == "negative":
                sent_score = -round(score, 4)
            else:
                sent_score = 0.0

            results.append({
                "sentiment":       label,
                "sentiment_score": sent_score
            })
    except Exception as e:
        print(f"   ⚠️ Sentiment hatası: {e}")
        results = [{"sentiment": "neutral", "sentiment_score": 0.0}] * len(texts)
    return results


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

def main(symbol: str | None, batch_size: int):
    print("📦 MongoDB bağlantısı kuruluyor...")
    collection = get_collection()

    # Analiz edilmemiş tweetleri çek
    query = {"is_analyzed": False}
    if symbol:
        query["symbol"] = symbol.upper()

    total = collection.count_documents(query)
    if total == 0:
        print("✅ Analiz edilecek tweet bulunamadı. Tüm tweetler zaten analiz edilmiş.")
        return

    print(f"📊 {total} tweet analiz edilecek.")
    if symbol:
        print(f"   Hisse filtresi: {symbol.upper()}")

    # Modelleri yükle
    print("\n🧠 Modeller yükleniyor...")

    print("   [1/2] toxic-bert yükleniyor...")
    toxicity_analyzer = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        truncation=True,
        max_length=512,
        top_k=None
    )
    print("   ✅ toxic-bert hazır.")

    print("   [2/2] FinBERT yükleniyor...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512
    )
    print("   ✅ FinBERT hazır.\n")

    # Batch halinde analiz et
    processed = 0
    cursor = collection.find(query).batch_size(batch_size)
    batch_docs  = []
    batch_texts = []

    def flush_batch():
        nonlocal processed

        if not batch_docs:
            return

        texts = batch_texts[:]

        tox_results  = run_toxicity(toxicity_analyzer, texts)
        sent_results = run_sentiment(sentiment_analyzer, texts)

        for doc, tox, sent in zip(batch_docs, tox_results, sent_results):
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {
                    "is_toxic":        tox["is_toxic"],
                    "toxicity_score":  tox["toxicity_score"],
                    "sentiment":       sent["sentiment"],
                    "sentiment_score": sent["sentiment_score"],
                    "is_analyzed":     True,
                    "analyzed_at":     datetime.now(timezone.utc),
                }}
            )
            processed += 1

        pct = processed / total * 100
        print(f"   → {processed}/{total} (%{pct:.1f}) tamamlandı...", end="\r")

        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        batch_docs.append(doc)
        batch_texts.append(doc.get("tweet", ""))

        if len(batch_docs) >= batch_size:
            flush_batch()

    # Kalan tweetleri işle
    flush_batch()

    print(f"\n\n{'='*50}")
    print(f"🏆 ANALİZ TAMAMLANDI!")
    print(f"   İşlenen tweet : {processed}")

    # Özet istatistik
    toxic_count = collection.count_documents(
        {**({"symbol": symbol.upper()} if symbol else {}), "is_toxic": True}
    )
    pos_count = collection.count_documents(
        {**({"symbol": symbol.upper()} if symbol else {}), "sentiment": "positive"}
    )
    neg_count = collection.count_documents(
        {**({"symbol": symbol.upper()} if symbol else {}), "sentiment": "negative"}
    )

    print(f"\n   📊 Özet İstatistik:")
    print(f"   Toksik tweet  : {toxic_count}")
    print(f"   Pozitif       : {pos_count}")
    print(f"   Negatif       : {neg_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tweet Analiz Aracı")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Sadece tek hisse analiz et (örn: AAPL)")
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                        help=f"Batch boyutu (varsayılan: {DEFAULT_BATCH})")
    args = parser.parse_args()

    main(args.symbol, args.batch)