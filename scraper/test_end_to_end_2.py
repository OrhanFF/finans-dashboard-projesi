import ml_service

try:
    print("Testing /predict?symbol=TSLA")
    res = ml_service.predict("TSLA")
    print("SUCCESS!")
    print(res)
except Exception as e:
    import traceback
    traceback.print_exc()
