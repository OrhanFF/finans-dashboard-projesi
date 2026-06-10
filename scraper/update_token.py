import asyncio
import os
import re
from playwright.async_api import async_playwright

async def main():
    print("=========================================")
    print("  TWITTER (X) YENI TOKEN ALICI")
    print("=========================================")
    print("Tarayici aciliyor...")
    print("Lutfen acilan pencereden X hesabiniza giris yapin.")
    print("Siz giris yapip ana sayfayi gorene kadar bu pencere kapanmayacak (sure: 5 dakika).")
    print("=========================================")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        await page.goto("https://x.com/login")
        
        # Kullanici /home (ana sayfa) adresine yonlenene kadar bekle
        try:
            await page.wait_for_url("**/home", timeout=300000) # 5 dakika bekle
        except Exception:
            print("\nHATA: Giris yapilamadi veya zaman asimina ugradi.")
            await browser.close()
            return

        print("\n✅ Giris basarili! Arka planda guvenlik anahtari (auth_token) aliniyor...")
        await asyncio.sleep(2) # Cerezlerin oturmasi icin kisa bir bekleme
        
        cookies = await context.cookies()
        auth_token = None
        for c in cookies:
            if c['name'] == 'auth_token':
                auth_token = c['value']
                break
                
        if auth_token:
            print(f"🔑 Yeni Token Bulundu: {auth_token[:5]}...{auth_token[-5:]}")
            
            # .env dosyasini bul ve guncelle
            env_path = "d:\\finans-dashboard-projesi\\scraper\\.env"
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Mevcut tokeni degistir
                if "X_AUTH_TOKEN" in content:
                    content = re.sub(r'X_AUTH_TOKEN=.*', f'X_AUTH_TOKEN="{auth_token}"', content)
                else:
                    content += f'\nX_AUTH_TOKEN="{auth_token}"\n'
                    
                with open(env_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print("✅ .env dosyasi basariyla GUNCELLEDI!")
            else:
                print("HATA: .env dosyasi bulunamadi!")
        else:
            print("HATA: auth_token cerezlerde bulunamadi.")
            
        await browser.close()
        print("Pencere kapatildi. Artik collect_tweets.bat dosyasini tekrar calistirabilirsiniz!")

if __name__ == "__main__":
    asyncio.run(main())
