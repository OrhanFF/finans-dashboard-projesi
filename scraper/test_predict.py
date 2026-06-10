import pickle
import numpy as np

with open('model_v3.pkl', 'rb') as f:
    saved = pickle.load(f)

MODEL = saved['pipeline']
FEATURES = saved['features']

print("FEATURES:", FEATURES)

feature_map = {}
for f in FEATURES:
    feature_map[f] = 0.0

X = np.array([[feature_map[f] for f in FEATURES]])
print("X shape:", X.shape)

try:
    pred = MODEL.predict(X)
    print("Pred:", pred)
except Exception as e:
    import traceback
    traceback.print_exc()
