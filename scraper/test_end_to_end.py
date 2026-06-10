import ml_service

try:
    print("Testing /predict?symbol=AAPL")
    res = ml_service.predict("AAPL")
    print("SUCCESS!")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
