@echo off
chcp 65001 > nul
echo =========================================
echo  PYTORCH GPU (CUDA) SURUMU YUKLENIYOR...
echo =========================================
echo.
echo 1) Eski (Sadece CPU destekli) PyTorch siliniyor...
pip uninstall -y torch torchvision torchaudio

echo.
echo 2) Yeni (Ekran karti destekli) PyTorch indiriliyor...
echo (Not: Bu indirme islemi dosya boyutundan dolayi internetinizin hizina gore 5-10 dakika surebilir, lutfen bekleyin.)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Kurulum Tamamlandi! GPU artik aktif.
echo Simdi tekrar analyze_tweets.bat dosyasini calistirabilirsiniz.
pause
