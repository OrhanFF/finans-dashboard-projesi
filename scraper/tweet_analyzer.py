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
from pymongo import MongoClient, UpdateOne
from transformers import pipeline

load_dotenv()

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────

MONGO_URI          = os.getenv("MONGO_URI", "mongodb://localhost:27017/finans")
DEFAULT_BATCH      = 64
TOXICITY_THRESHOLD = 0.60
TOXIC_LABELS       = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}


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
        print(f"\n   ⚠️ Toxicity hatası: {e}")
        results = [{"is_toxic": False, "toxicity_score": 0.0}] * len(texts)
    return results


def run_sentiment(analyzer, texts: list) -> list:
    results = []
    try:
        outputs = analyzer(texts)
        for output in outputs:
            label = output["label"]
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
        print(f"\n   ⚠️ Sentiment hatası: {e}")
        results = [{"sentiment": "neutral", "sentiment_score": 0.0}] * len(texts)
    return results


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

def main(symbol: str | None, batch_size: int):
    print("📦 MongoDB bağlantısı kuruluyor...")
    collection = get_collection()

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

    # ── Donanım seç ──────────────────────────────────────────────
    print("\n🧠 Modeller yükleniyor...")
    import torch
    device_id = 0 if torch.cuda.is_available() else -1
    device_label = "GPU (CUDA)" if device_id == 0 else "CPU"
    print(f"   [!] Kullanılan donanım: {device_label}")

    print("   [1/2] toxic-bert yükleniyor...")
    toxicity_analyzer = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        truncation=True,
        max_length=512,
        top_k=None,
        device=device_id
    )
    print("   ✅ toxic-bert hazır.")

    print("   [2/2] FinBERT yükleniyor...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        truncation=True,
        max_length=512,
        device=device_id
    )
    print("   ✅ FinBERT hazır.\n")

    # ── Batch analiz ─────────────────────────────────────────────
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="Analiz", unit="tweet")
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("   (tqdm kurulu değil, sade ilerleme gösterilecek)")

    processed   = 0
    cursor      = collection.find(query).batch_size(batch_size)
    batch_docs  = []
    batch_texts = []

    def flush_batch():
        nonlocal processed
        if not batch_docs:
            return

        tox_results  = run_toxicity(toxicity_analyzer,  batch_texts[:])
        sent_results = run_sentiment(sentiment_analyzer, batch_texts[:])

        operations = [
            UpdateOne(
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
            for doc, tox, sent in zip(batch_docs, tox_results, sent_results)
        ]

        if operations:
            collection.bulk_write(operations)
            processed += len(operations)
            if use_tqdm:
                pbar.update(len(operations))
            else:
                pct = processed / total * 100
                print(f"   → {processed}/{total} (%{pct:.1f}) tamamlandı...", end="\r")

        batch_docs.clear()
        batch_texts.clear()

    for doc in cursor:
        batch_docs.append(doc)
        batch_texts.append(doc.get("tweet", ""))
        if len(batch_docs) >= batch_size:
            flush_batch()

    flush_batch()

    if use_tqdm:
        pbar.close()

    # ── Özet ─────────────────────────────────────────────────────
    sym_filter = {"symbol": symbol.upper()} if symbol else {}

    toxic_count = collection.count_documents({**sym_filter, "is_toxic": True})
    pos_count   = collection.count_documents({**sym_filter, "sentiment": "positive"})
    neg_count   = collection.count_documents({**sym_filter, "sentiment": "negative"})
    neu_count   = collection.count_documents({**sym_filter, "sentiment": "neutral"})

    print(f"\n\n{'='*50}")
    print(f"🏆 ANALİZ TAMAMLANDI!")
    print(f"   İşlenen tweet : {processed}")
    print(f"\n   📊 Özet İstatistik:")
    if processed > 0:
        print(f"   Toksik tweet  : {toxic_count}  (%{toxic_count/processed*100:.1f})")
        print(f"   Pozitif       : {pos_count}  (%{pos_count/processed*100:.1f})")
        print(f"   Negatif       : {neg_count}  (%{neg_count/processed*100:.1f})")
        print(f"   Nötr          : {neu_count}  (%{neu_count/processed*100:.1f})")
    print(f"{'='*50}")
    print(f"\n💡 Sonraki adım: python tweet_feature_builder.py --merge")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tweet Analiz Aracı")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Sadece tek hisse analiz et (örn: AAPL)")
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH,
                        help=f"Batch boyutu (varsayılan: {DEFAULT_BATCH})")
    args = parser.parse_args()

    main(args.symbol, args.batch)