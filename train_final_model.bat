@echo off
chcp 65001 > nul
echo ========================================
echo   YAPAY ZEKA MODEL EĞİTİMİ (V3 NİHAİ)
echo ========================================
echo.
echo Bu islem; Teknik Analiz, Haber Duygusu ve Twitter Duygusunu
echo harmanlayarak en gelismis, nihai Yapay Zeka (V3) modelini egitecektir.
echo Egitim basliyor...
echo.

cd /d d:\finans-dashboard-projesi\scraper
python model_trainer_v3.py

echo.
pause
