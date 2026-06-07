"""
Egitim verisini analiz eder:
- Sinif dagılımı (Target 0 vs 1)
- VIX degerine gore DUSUS/YUKSELIS dagılımı
- Suanki VIX=15.4 egitimde hangi sinıfa düşüyor?
- Her feature'ın ortalama degeri sinıfa gore
"""
import pandas as pd
import numpy as np

df = pd.read_csv("MultiStock_Training_v2.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['VIX_log'] = np.log(df['VIX_Close'].clip(lower=1.0))

print("=" * 60)
print("EGITIM VERISI ANALIZI")
print("=" * 60)

# 1. Sinif dagılımı
total = len(df)
dusus    = (df['Target'] == 0).sum()
yukselis = (df['Target'] == 1).sum()
print(f"\n1. SINIF DAGILI:")
print(f"   DUSUS    (0): {dusus:4d}  (%{dusus/total*100:.1f})")
print(f"   YUKSELIS (1): {yukselis:4d}  (%{yukselis/total*100:.1f})")
print(f"   TOPLAM      : {total}")

# 2. VIX aralik analizi
print(f"\n2. VIX ARALIK:")
print(f"   Egitim min : {df['VIX_Close'].min():.1f}")
print(f"   Egitim max : {df['VIX_Close'].max():.1f}")
print(f"   Egitim ort : {df['VIX_Close'].mean():.1f}")
print(f"   Guncel VIX : 15.4  (log={np.log(15.4):.3f})")

# 3. Dusuk VIX (<20) altında sinif dagılımı
low_vix = df[df['VIX_Close'] < 20]
lv_d = (low_vix['Target'] == 0).sum()
lv_y = (low_vix['Target'] == 1).sum()
print(f"\n3. DUSUK VIX (<20) --- {len(low_vix)} gun:")
print(f"   DUSUS    : {lv_d} (%{lv_d/len(low_vix)*100:.1f})")
print(f"   YUKSELIS : {lv_y} (%{lv_y/len(low_vix)*100:.1f})")
print(f"   => Model dusuk VIX'i {('DUSUS' if lv_d > lv_y else 'YUKSELIS')} ile ilikilendiriyor!")

# 4. VIX araligına gore dagılım
print(f"\n4. VIX ARALIKLARINDA YUKSELIS ORANI:")
bins = [0, 15, 20, 25, 30, 40, 50, 100]
labels = ['<15','15-20','20-25','25-30','30-40','40-50','>50']
df['vix_bin'] = pd.cut(df['VIX_Close'], bins=bins, labels=labels)
grouped = df.groupby('vix_bin', observed=True)['Target'].agg(['mean','count'])
grouped.columns = ['yukselis_orani', 'gun_sayisi']
for idx, row in grouped.iterrows():
    bar = '#' * int(row['yukselis_orani'] * 20)
    print(f"   VIX {str(idx):6}: {bar:<20} %{row['yukselis_orani']*100:.1f}  ({int(row['gun_sayisi'])} gun)")

# 5. Suanki degerler egitimde nerede?
print(f"\n5. SUANKI DEGERLER EGITIMDE NEREDE?")
features_current = {
    'VIX_Close': 15.4,
    'RSI_14':    65.85,   # AAPL
    'Momentum_5d': -0.004,
    'Daily_Return': 0.003,
    'MA20_ratio': 0.026,
}
for feat, cur_val in features_current.items():
    if feat not in df.columns:
        continue
    pct = (df[feat] < cur_val).mean() * 100
    mean_val = df[feat].mean()
    print(f"   {feat:<18}: suanki={cur_val:.3f} | egitim_ort={mean_val:.3f} | egitimin %{pct:.0f}'inden dusuk")

# 6. Feature ortalama - sinifa gore
print(f"\n6. FEATURE ORTALAMA (DUSUS vs YUKSELIS):")
features = ['VIX_Close', 'Daily_Return', 'RSI_14', 'MA20_ratio', 'Momentum_5d', 'Sentiment_Decay']
for f in features:
    if f not in df.columns:
        continue
    d_avg = df[df['Target']==0][f].mean()
    y_avg = df[df['Target']==1][f].mean()
    diff = y_avg - d_avg
    sign = "^" if diff > 0 else "v"
    print(f"   {f:<20}: DUSUS={d_avg:7.3f} | YUKSELIS={y_avg:7.3f} | fark={diff:+.3f} {sign}")
