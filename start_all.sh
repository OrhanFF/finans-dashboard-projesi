#!/bin/bash

echo "========================================="
echo "  Gündem Borsa AI - Mac Başlatıcı (start_all.sh) "
echo "========================================="
echo "Servisler sırasıyla ayağa kaldırılıyor..."
echo ""

# 1. Frontend (Vite) - Port 5173
echo "[1/4] Frontend başlatılıyor (port 5173)..."
cd frontend
npm run dev &
cd ..
sleep 3

# 2. Node Backend - Port 3001
echo "[2/4] Node Backend başlatılıyor (port 3001)..."
cd backend
npm run dev &
cd ..
sleep 3

# 3. Python ML Service - Port 8002
echo "[3/4] ML Service başlatılıyor (port 8002)..."
cd scraper
source venv/bin/activate
uvicorn ml_service:app --host 0.0.0.0 --port 8002 &
cd ..
sleep 3

# 4. Python Scraper API - Port 8001
echo "[4/4] Scraper API başlatılıyor (port 8001)..."
cd scraper
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 &
cd ..

echo ""
echo "========================================="
echo "Bütün servisler arka planda çalışıyor!"
echo "Tarayıcınızdan http://localhost:5173 adresine gidebilirsiniz."
echo "Tüm servisleri durdurmak için bu pencereyi kapatın veya 'killall node' ve 'killall python' komutlarını kullanın."
echo "========================================="

# Betiğin kapanmasını engelle
wait
