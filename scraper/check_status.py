from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/finans")

def check_status():
    client = MongoClient(MONGO_URI)
    db = client.get_default_database()
    col = db["tweets"]
    
    symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN', 'JPM', 'JNJ', 'WMT', 'XOM', 'GOOGL']
    
    print("\n--- MONGODB TWEET DURUMU ---")
    for sym in symbols:
        count = col.count_documents({"symbol": sym})
        status = "BİTTİ" if count > 2000 else ("YARIM KALDI" if count > 0 else "BAŞLAMADI")
        print(f"{sym:<6}: {count:>5} tweet  [{status}]")
    print("----------------------------\n")

if __name__ == "__main__":
    check_status()
