@echo off
chcp 65001 > nul
echo ========================================
echo   TWEETLERI MODEL VERISIYLE BIRLESTIR
echo ========================================
echo.
echo Bu islem MongoDB'deki analiz edilmis tweetlerin
echo gunluk ortalamalarini hesaplayip (Sentiment, Hacim, Toksik Orani),
echo mevcut MultiStock_Training_v2.csv ile birlestirir
echo ve nihai "MultiStock_Training_v3.csv" dosyasini uretir.
echo.

cd /d d:\finans-dashboard-projesi\scraper
python tweet_feature_builder.py --merge

echo.
pause
