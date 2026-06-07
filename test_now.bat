@echo off
echo ========================================
echo   V2C MODEL TAHMIN TESTI
echo ========================================
echo.
echo ML servisi hazir mi kontrol ediliyor...
set PYTHONUTF8=1
python -c "import requests; r=requests.get('http://127.0.0.1:8002/health', timeout=5); print('OK:', r.json()['model_name'])" 2>nul
if errorlevel 1 (
    echo HATA: ML servisi calismiyor!
    echo Once start_all.bat calistirin.
    pause
    exit /b
)
echo.
echo Tahminler aliniyor...
python scraper\test_predictions.py
echo.
pause
