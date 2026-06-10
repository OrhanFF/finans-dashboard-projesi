@echo off
chcp 65001 > nul
echo ========================================
echo   TWITTER (X) VERI TOPLAYICI (36 AY)
echo ========================================
echo.
echo Islem basliyor. Bu script onceki eksik tweetleri temizler,
echo ardindan son 3 yilin en populer tweetlerini toplamaya baslar.
echo Lutfen islem bitene kadar bu pencereyi kapatmayin.
echo.

cd /d d:\finans-dashboard-projesi\scraper
python tweet_collector.py --clean

echo.
echo Islem Tamamlandi!
pause
