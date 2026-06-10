@echo off
chcp 65001 > nul
echo ========================================
echo   TWEET DUYGU ANALIZI (SENTIMENT)
echo ========================================
echo.
echo Bu islem MongoDB'deki toplanmis 24.000 tweeti
echo tek tek Yapay Zeka (FinBERT) ile okuyup, pozitif veya
echo negatif olduguna karar verecektir.
echo Bu islem bilgisayarinizin hizina gore birkac saat surebilir.
echo.

cd /d d:\finans-dashboard-projesi\scraper
python tweet_analyzer.py --batch 32

echo.
echo Analiz Tamamlandi!
pause
