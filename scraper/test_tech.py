import ml_service

try:
    print("Testing get_technical_features('AAPL')")
    tech = ml_service.get_technical_features('AAPL')
    print("SUCCESS!")
    print(tech)
except Exception as e:
    import traceback
    traceback.print_exc()

try:
    print("Testing get_sentiment_features('AAPL')")
    sent = ml_service.get_sentiment_features('AAPL')
    print("SUCCESS!")
    print(sent)
except Exception as e:
    import traceback
    traceback.print_exc()
