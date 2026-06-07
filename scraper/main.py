import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page
)
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from transformers import pipeline as hf_pipeline

load_dotenv()

# ─────────────────────────────────────────────
# TOXICITY AYARLARI
# ─────────────────────────────────────────────

# Modelin toksik olarak etiketlediği sınıflar
TOXIC_LABELS = {"toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"}

# Bu eşiğin üzerindeki skor → toksik kabul edilir (%60)
TOXICITY_THRESHOLD = 0.60


async def block_trackers(route):
    """
    X.com'un bot algılamasını tetikleyebilecek
    yaygın izleyici ve reklam domain'lerini engeller.
    """
    url = route.request.url
    if (
        route.request.resource_type == "image" or
        "googleads" in url or
        "analytics" in url or
        "googletagmanager" in url or
        "doubleclick.net" in url
    ):
        await route.abort()
    else:
        await route.continue_()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Sunucu başlıyor... Playwright başlatılıyor...")

    auth_token = os.getenv("X_AUTH_TOKEN")
    if not auth_token:
        raise ValueError("X_AUTH_TOKEN .env dosyasında bulunamadı. Lütfen Adım 1-2-3'ü kontrol edin.")

    # ── toxic-bert bir kere yükleniyor ──────────────────────────
    print("🧠 toxic-bert modeli yükleniyor... (ilk seferde ~400MB indirir)")
    try:
        toxicity_analyzer = hf_pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            truncation=True,
            max_length=512,
            top_k=None          # Tüm etiket skorlarını döndür
        )
        app.state.toxicity_analyzer = toxicity_analyzer
        print("✅ toxic-bert hazır.")
    except Exception as e:
        print(f"⚠️ toxic-bert yüklenemedi: {e}")
        app.state.toxicity_analyzer = None
    # ────────────────────────────────────────────────────────────

    p = await async_playwright().start()

    browser = await p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled"]
    )

    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    )

    print("Oturum cookie'si tarayıcıya ekleniyor...")
    await context.add_cookies([
        {
            "name": "auth_token",
            "value": auth_token,
            "domain": ".x.com",
            "path": "/",
            "secure": True,
            "httpOnly": True,
            "sameSite": "None"
        }
    ])
    print("Cookie eklendi.")

    page_for_test = None
    try:
        print("Girişin başarılı olup olmadığını test etmek için X.com ana sayfasına gidiliyor...")
        page_for_test = await context.new_page()
        await page_for_test.route("**/*", block_trackers)

        await page_for_test.goto("https://x.com/home", wait_until="networkidle", timeout=60000)
        await page_for_test.wait_for_selector('a[aria-label="Ara ve keşfet"]', timeout=15000)

        print("Cookie ile giriş başarılı! Sunucu hazır.")
        await page_for_test.close()

    except Exception as e:
        print(f"Cookie ile giriş yapılamadı. ANA HATA: {e}")
        print("Muhtemel Sebepler:")
        print("1. 'X_AUTH_TOKEN' .env dosyasında yanlış veya eksik.")
        print("2. 'auth_token' süresi dolmuş olabilir (Tekrar almayı deneyin).")
        print("3. X.com yine de bot olduğunu anladı.")

        if page_for_test:
            await page_for_test.screenshot(path="cookie_login_error.png")
            print("Ekran görüntüsü 'cookie_login_error.png' olarak kaydedildi.")

        await context.close()
        await browser.close()
        await p.stop()
        raise

    app.state.playwright = p
    app.state.browser = browser
    app.state.context = context

    yield

    print("Sunucu kapanıyor... Tarayıcı ve Playwright kapatılıyor.")
    await app.state.context.close()
    await app.state.browser.close()
    await app.state.playwright.stop()
    print("Kapatma işlemi tamamlandı.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Merhaba": "Scraper Servisi (Gerçek Veri Aktif - Cookie Modu + Toxicity Analizi)"}


# ─────────────────────────────────────────────
# TOXİCİTY HESAPLAMA
# ─────────────────────────────────────────────

def analyze_toxicity(analyzer, text: str) -> dict:
    """
    Verilen tweet metnini toxic-bert ile analiz eder.

    Döndürür:
        is_toxic      (bool)  — TOXICITY_THRESHOLD üzerindeyse True
        toxicity_score (float) — en yüksek toksik etiket skoru (0.0 – 1.0)
    """
    if analyzer is None:
        return {"is_toxic": False, "toxicity_score": 0.0}

    try:
        # top_k=None → tüm etiketlerin skorlarını liste olarak verir
        results = analyzer(text)[0]  # [{"label": "toxic", "score": 0.92}, ...]

        # Sadece toksik etiketlere ait skorları al
        toxic_scores = [
            r["score"]
            for r in results
            if r["label"].lower() in TOXIC_LABELS
        ]

        max_toxic_score = max(toxic_scores) if toxic_scores else 0.0
        is_toxic = max_toxic_score >= TOXICITY_THRESHOLD

        return {
            "is_toxic": is_toxic,
            "toxicity_score": round(max_toxic_score, 4)
        }

    except Exception as e:
        print(f"⚠️ Toxicity analizi hatası: {e}")
        return {"is_toxic": False, "toxicity_score": 0.0}


# ─────────────────────────────────────────────
# TWEET ÇEKME
# ─────────────────────────────────────────────

async def get_tweets_from_x(
    context: BrowserContext,
    hashtag: str,
    toxicity_analyzer,
    limit: int = 15
):
    """
    Mevcut tarayıcı bağlamını kullanarak tweet çeker,
    her tweet'i toxic-bert ile analiz eder.
    """
    tweets_data = []
    seen_texts = set()
    page = None

    try:
        page = await context.new_page()
        await page.route("**/*", block_trackers)

        print(f"'{hashtag}' için ana sayfaya gidiliyor...")
        await page.goto("https://x.com/home", wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(2000)

        # Arama çubuğuna tıkla
        print("Arama çubuğu aranıyor...")
        search_box = page.locator('a[aria-label="Ara ve keşfet"]')
        await search_box.wait_for(timeout=10000)
        await search_box.click()
        await page.wait_for_timeout(1000)

        # Arama kutusuna yaz
        search_input = page.locator('input[data-testid="SearchBox_Search_Input"]')
        await search_input.wait_for(timeout=10000)
        await search_input.fill(f"#{hashtag}")
        await page.wait_for_timeout(500)
        await search_input.press("Enter")

        # Sayfanın yüklenmesini bekle
        await page.wait_for_timeout(3000)
        print("Arama sayfası yüklendi. Tweetler aranıyor...")

        while len(tweets_data) < limit:
            article_locator = page.locator('article[data-testid="tweet"]')

            try:
                await article_locator.first.wait_for(timeout=5000)
            except Exception:
                print("Tweet'ler yüklenemedi veya bulunamadı.")
                break

            count = await article_locator.count()
            if count == 0:
                print("Tweet bulunamadı veya sayfa yüklenemedi.")
                break

            for i in range(count):
                article = article_locator.nth(i)
                try:
                    text_element = article.locator('div[data-testid="tweetText"]')
                    await text_element.wait_for(timeout=1000)
                    text = await text_element.inner_text()

                    # Aynı içerikli tweet'i iki kere ekleme
                    if text in seen_texts:
                        continue
                    seen_texts.add(text)

                    user_element = article.locator('span:has-text("@")').first
                    await user_element.wait_for(timeout=1000)
                    user = await user_element.inner_text()

                    time_element = article.locator('time').first
                    await time_element.wait_for(timeout=1000)
                    timestamp = await time_element.get_attribute('datetime')

                    # ── Toxicity Analizi ──────────────────────
                    toxicity = analyze_toxicity(toxicity_analyzer, text)
                    label = "🔴 TOKSİK" if toxicity["is_toxic"] else "🟢 TEMİZ"
                    print(f"   [{label}] @{user}: {text[:60]}...")
                    # ─────────────────────────────────────────

                    tweets_data.append({
                        "id":             timestamp,
                        "user":           user,
                        "tweet":          text,
                        "timestamp":      timestamp,
                        "is_toxic":       toxicity["is_toxic"],
                        "toxicity_score": toxicity["toxicity_score"]
                    })

                    if len(tweets_data) >= limit:
                        break

                except Exception:
                    pass

            if len(tweets_data) >= limit:
                break

            await page.mouse.wheel(0, 5000)
            await page.wait_for_timeout(2000)

    except Exception as e:
        print(f"Playwright (get_tweets_from_x) hatası: {e}")
        if page:
            await page.screenshot(path='search_error.png')
    finally:
        if page:
            await page.close()

    return tweets_data[:limit]


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────

@app.get("/scrape", response_model=List[Dict[str, Any]])
async def scrape_tweets_endpoint(request: Request, ticker: str):
    """
    Bir hisse senedi (ticker) sembolü alır,
    X'ten gerçek tweet verisi çeker ve toxicity analizi ekler.
    """
    print(f"'{ticker}' için GERÇEK veri isteği alındı...")

    clean_ticker = ticker.lstrip('#')
    context = request.app.state.context
    toxicity_analyzer = request.app.state.toxicity_analyzer

    tweets = await get_tweets_from_x(
        context,
        clean_ticker,
        toxicity_analyzer,
        limit=15
    )

    if not tweets:
        print(f"'{clean_ticker}' için veri çekilemedi.")
        return []

    toxic_count = sum(1 for t in tweets if t["is_toxic"])
    print(f"'{clean_ticker}' için {len(tweets)} tweet çekildi. "
          f"Toksik: {toxic_count} / {len(tweets)}")

    return tweets