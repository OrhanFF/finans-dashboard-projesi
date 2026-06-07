import requests
import json
import time

# Tum hisseler - cache yok, hepsi taze cekiliyor
symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'JPM', 'AMZN', 'JNJ', 'WMT', 'XOM']
results = []

print("=" * 65)
print("TAHMIN TESTI - v2c Model (9460 satir egitim verisi)")
print("=" * 65)

# Health check
try:
    h = requests.get('http://127.0.0.1:8002/health', timeout=5).json()
    print(f"Model    : {h['model_name']}")
    print(f"Features : {len(h['features'])} adet")
    print("=" * 65)
except Exception as e:
    print(f"ML servisi calismiyor: {e}")
    exit()

# Tahminler
for sym in symbols:
    t0 = time.time()
    try:
        r = requests.get('http://127.0.0.1:8002/predict',
                         params={'symbol': sym}, timeout=120)
        elapsed = round(time.time() - t0, 1)
        d = r.json()
        d['elapsed_s'] = elapsed
        results.append(d)

        pred   = d['prediction']
        conf   = d['confidence']
        price  = d['current_price']
        rsi    = d['rsi_14']
        sent   = d['sentiment_score']
        mom    = d['momentum_5d']
        icon   = "^" if d['prediction_int'] == 1 else "v"
        print(f"{sym:5} {icon} {pred:8} | %{conf*100:4.1f} guvен | "
              f"${price:.0f} | RSI:{rsi:.0f} | sent:{sent:+.3f} | mom:{mom:+.4f} | {elapsed}s")
    except Exception as e:
        print(f"{sym:5}: HATA - {e}")

print()
print("=" * 65)
yukselis = [r for r in results if r.get('prediction_int') == 1]
dusus    = [r for r in results if r.get('prediction_int') == 0]
print(f"YUKSELIS : {len(yukselis)} hisse -> {', '.join(r['symbol'] for r in yukselis)}")
print(f"DUSUS    : {len(dusus)} hisse -> {', '.join(r['symbol'] for r in dusus)}")
print()
if results:
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    print(f"Ort. Guven    : %{avg_conf*100:.1f}")
    print(f"Ort. RSI      : {sum(r['rsi_14'] for r in results)/len(results):.1f}")
    print(f"Ort. Sentiment: {sum(r['sentiment_score'] for r in results)/len(results):+.4f}")
print("=" * 65)
