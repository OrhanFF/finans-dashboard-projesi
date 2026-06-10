import requests

try:
    print("Testing /predict?symbol=AAPL")
    r = requests.get('http://localhost:8002/predict?symbol=AAPL')
    print("STATUS:", r.status_code)
    print("BODY:", r.text)
except Exception as e:
    print("ERROR:", str(e))
