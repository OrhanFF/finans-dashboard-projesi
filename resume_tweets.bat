@echo off
chcp 65001 > nul
echo ========================================
echo   KALAN HISSELERI TOPLAMAYA DEVAM ET
echo ========================================
echo.
echo Bu script, AAPL, TSLA ve NVDA'yi atlayarak 
echo kalan hisseleri (MSFT, AMZN, JPM, JNJ, WMT, XOM, GOOGL) toplar.
echo Temizlik yapilmaz, eski veriler korunur.
echo.

cd /d d:\finans-dashboard-projesi\scraper
python tweet_collector.py --symbol GOOGL

echo.
echo Islem Tamamlandi!
pause
