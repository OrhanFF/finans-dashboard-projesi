@echo off
chcp 65001 > nul
echo ========================================
echo   FINANS DASHBOARD - TUM SERVISLER
echo ========================================
echo.

echo [1/3] ML Servisi baslatiliyor (port 8002)...
start "ML Service - Port 8002" cmd /k "chcp 65001 > nul && cd /d d:\finans-dashboard-projesi\scraper && uvicorn ml_service:app --host 0.0.0.0 --port 8002 --reload"

timeout /t 3 /nobreak > nul

echo [2/3] Backend baslatiliyor (port 3001)...
start "Backend - Port 3001" cmd /k "cd /d d:\finans-dashboard-projesi\backend && npm start"

timeout /t 3 /nobreak > nul

echo [3/3] Frontend baslatiliyor (port 5173)...
start "Frontend - Port 5173" cmd /k "cd /d d:\finans-dashboard-projesi\frontend && npm run dev"

echo.
echo ========================================
echo Servisler hazir (birka saniye bekle):
echo   ML Service : http://localhost:8002/health
echo   Backend    : http://localhost:3001
echo   Frontend   : http://localhost:5173
echo ========================================
echo.
pause
