"""
tweet_collector.py
------------------
Watchlist hisseleri icin X'ten kaliteli tweet toplar.

Strateji:
  - f=top: En cok etkilesim alan tweetler (retweet/like)
  - Aylik tarih dilimleri: Son 6 ay → her aya ~150 tweet
  - Toplam: ~900 tweet/hisse ama 6 aya yayilmis
  - Gunluk sentiment hesaplanabilir

Kullanim:
  python tweet_collector.py              # Tüm hisseler
  python tweet_collector.py --clean      # Önce MongoDB'yi temizle
  python tweet_collector.py --symbol AAPL
  python tweet_collector.py --months 3  # Son 3 ay
"""

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import argparse
from datetime import datetime, timezone, timedelta
import calendar

import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from pymongo import MongoClient, errors as mongo_errors

load_dotenv()

WATCHLIST  = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN',
              'JPM',  'JNJ',  'WMT',  'XOM',  'GOOGL']

MONGO_URI  = os.getenv("MONGO_URI", "mongodb://localhost:27017/finans")
AUTH_TOKEN = os.getenv("X_AUTH_TOKEN", "")

TWEETS_PER_MONTH  = 150   # Ay basina hedef tweet
MONTHS_BACK       = 12    # Kac ay geriye gidecegiz (1 yila cikarildi)
SCROLL_PAUSE      = 2500
MAX_EMPTY_SCROLLS = 6

# ─────────────────────────────────────────────
# MONGODB
# ─────────────────────────────────────────────

def get_collection():
    client = MongoClient(MONGO_URI)
    db     = client.get_default_database()
    col    = db["tweets"]
    col.create_index([("symbol", 1), ("createdAt", -1)])
    col.create_index([("user", 1), ("tweet", 1)], unique=True)
    col.create_index("is_analyzed")
    return col


def clean_collection(collection, symbol: str = None):
    if symbol:
        r = collection.delete_many({"symbol": symbol})
        print(f"   {symbol} icin {r.deleted_count} tweet silindi.")
    else:
        r = collection.delete_many({})
        print(f"   Tum tweetler silindi: {r.deleted_count} kayit.")


def save_tweets(collection, symbol: str, tweets: list) -> int:
    saved = 0
    for t in tweets:
        try:
            collection.insert_one({
                "symbol":          symbol,
                "user":            t["user"],
                "tweet":           t["tweet"],
                "timestamp":       t.get("timestamp", ""),
                "month_key":       t.get("month_key", ""),
                "is_toxic":        None,
                "toxicity_score":  None,
                "sentiment":       None,
                "sentiment_score": None,
                "is_analyzed":     False,
                "createdAt":       datetime.now(timezone.utc),
            })
            saved += 1
        except mongo_errors.DuplicateKeyError:
            pass
        except Exception as e:
            print(f"   Kayit hatasi: {e}")
    return saved


# ─────────────────────────────────────────────
# BOT ENGELLEYİCİ
# ─────────────────────────────────────────────

async def block_trackers(route):
    url = route.request.url
    if (
        route.request.resource_type == "image" or
        "googleads"        in url or
        "analytics"        in url or
        "googletagmanager" in url or
        "doubleclick.net"  in url
    ):
        await route.abort()
    else:
        await route.continue_()


# ─────────────────────────────────────────────
# AY ARALIKLARINI HESAPLA
# ─────────────────────────────────────────────

def get_month_ranges(months_back: int) -> list:
    """
    Son N ay icin (since, until) aralik listesi dondurur.
    Ornek: months_back=3 → [(2025-03-01, 2025-04-01), (2025-04-01, 2025-05-01), ...]
    """
    ranges = []
    today  = datetime.now()
    for i in range(months_back, 0, -1):
        # i ay önce
        year  = today.year
        month = today.month - i
        while month <= 0:
            month += 12
            year  -= 1
        # Ayin son gunu
        last_day = calendar.monthrange(year, month)[1]
        since    = f"{year}-{month:02d}-01"
        until_y  = year
        until_m  = month + 1
        if until_m > 12:
            until_m  = 1
            until_y += 1
        until = f"{until_y}-{until_m:02d}-01"
        ranges.append((since, until, f"{year}-{month:02d}"))
    return ranges


# ─────────────────────────────────────────────
# TWEET ÇEKME — TEK AY
# ─────────────────────────────────────────────

async def collect_month(context, symbol: str, since: str, until: str,
                         month_key: str, limit: int) -> list:
    """
    Bir ay icin X aramasinda en populer tweetleri toplar.
    f=top → en cok etkilesim alan tweetler (piyasa sinyali daha guclu).
    """
    tweets        = []
    seen_texts    = set()
    page          = None
    empty_scrolls = 0

    # $AAPL since:2025-01-01 until:2025-02-01 → en populer tweetler
    query      = f"%24{symbol}%20since%3A{since}%20until%3A{until}"
    search_url = f"https://x.com/search?q={query}&f=top&src=typed_query"

    try:
        page = await context.new_page()
        await page.route("**/*", block_trackers)

        await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(4)

        if "login" in page.url or "i/flow" in page.url:
            print(f"   [{symbol}] HATA: Oturum kapanmis!")
            return []

        # Yeniden Dene butonu varsa bas
        retry_btn = page.locator('span:has-text("Yeniden dene")')
        try:
            if await retry_btn.count() > 0:
                print(f"   [{symbol}][{month_key}] 'Yeniden dene' bulundu, tiklaniyor...")
                await retry_btn.first.click(timeout=3000)
                await asyncio.sleep(4)
        except:
            pass

        while len(tweets) < limit:
            article_locator = page.locator('article[data-testid="tweet"]')

            try:
                await article_locator.first.wait_for(timeout=12000)
            except Exception:
                # Son bir kez sayfayi yenileyip deneyelim
                await page.reload(wait_until="domcontentloaded")
                await asyncio.sleep(5)
                try:
                    await article_locator.first.wait_for(timeout=10000)
                except:
                    break  # Gercekten tweet yok

            count = await article_locator.count()
            new_this_scroll = 0

            for i in range(count):
                article = article_locator.nth(i)
                try:
                    text_el = article.locator('div[data-testid="tweetText"]')
                    await text_el.wait_for(timeout=1000)
                    text = await text_el.inner_text()

                    if text in seen_texts or len(text) < 10:
                        continue
                    seen_texts.add(text)

                    user_el = article.locator('span:has-text("@")').first
                    await user_el.wait_for(timeout=1000)
                    user = await user_el.inner_text()

                    time_el = article.locator('time').first
                    await time_el.wait_for(timeout=1000)
                    timestamp = await time_el.get_attribute('datetime')

                    tweets.append({
                        "user":      user,
                        "tweet":     text,
                        "timestamp": timestamp,
                        "month_key": month_key,
                    })
                    new_this_scroll += 1

                    if len(tweets) >= limit:
                        break

                except Exception:
                    pass

            if len(tweets) >= limit:
                break

            if new_this_scroll == 0:
                empty_scrolls += 1
                if empty_scrolls >= MAX_EMPTY_SCROLLS:
                    break
            else:
                empty_scrolls = 0

            # Daha guvenilir scroll (PageDown tusu)
            await page.keyboard.press("PageDown")
            await page.keyboard.press("PageDown")
            await asyncio.sleep(SCROLL_PAUSE / 1000)

    except Exception as e:
        print(f"   [{symbol}][{month_key}] Hata: {e}")
    finally:
        if page:
            await page.close()

    return tweets


# ─────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────

async def main(symbols: list, months_back: int, do_clean: bool):
    if not AUTH_TOKEN:
        print("HATA: X_AUTH_TOKEN .env dosyasinda bulunamadi!")
        return

    print("MongoDB baglantisi kuruluyor...")
    collection = get_collection()

    if do_clean:
        print("Temizlik yapiliyor...")
        clean_collection(collection)

    month_ranges = get_month_ranges(months_back)
    print(f"Donem: {month_ranges[0][2]} - {month_ranges[-1][2]} ({months_back} ay)")
    print(f"Hedef: ~{TWEETS_PER_MONTH} tweet/ay x {months_back} ay x {len(symbols)} hisse")
    print(f"       = ~{TWEETS_PER_MONTH * months_back * len(symbols)} tweet toplam\n")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
        )

        # Cookie ile giris
        await context.add_cookies([{
            "name":     "auth_token",
            "value":    AUTH_TOKEN,
            "domain":   ".x.com",
            "path":     "/",
            "secure":   True,
            "httpOnly": True,
            "sameSite": "None"
        }])

        # Giris dogrula
        print("Giris dogrulaniyor...")
        test_page = await context.new_page()
        try:
            await test_page.goto("https://x.com/home", wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)
            if "login" in test_page.url or "i/flow" in test_page.url:
                print("HATA: Token gecersiz veya suresi dolmus!")
                await browser.close()
                return
            try:
                await test_page.wait_for_selector('[data-testid="tweet"]', timeout=8000)
                print("Giris basarili! (tweet feed goruldu)\n")
            except Exception:
                if "home" in test_page.url:
                    print("Giris basarili! (URL dogrulandi)\n")
        finally:
            await test_page.close()

        total_saved = 0

        for symbol in symbols:
            print(f"\n{'='*55}")
            print(f"[{symbol}] islenIyor...")
            print(f"{'='*55}")

            symbol_total = 0

            for (since, until, month_key) in month_ranges:
                print(f"   [{symbol}][{month_key}] Aylik en populer tweetler aliniyor...")
                tweets = await collect_month(context, symbol, since, until,
                                             month_key, TWEETS_PER_MONTH)
                saved  = save_tweets(collection, symbol, tweets)
                symbol_total += saved
                print(f"   [{symbol}][{month_key}] {len(tweets)} tweet cekildI, {saved} MongoDB'ye kaydedildi.")
                await asyncio.sleep(2)

            total_saved += symbol_total
            print(f"   [{symbol}] TAMAMLANDI: {symbol_total} tweet ({months_back} ay)")

        await browser.close()

    print(f"\n{'='*55}")
    print(f"ISLEM TAMAM!")
    print(f"  Toplam kaydedilen: {total_saved} tweet")
    print(f"  Hisseler        : {', '.join(symbols)}")
    print(f"  Donem           : Son {months_back} ay (en populer tweetler)")
    print(f"{'='*55}")
    print(f"\nSonraki adim: python tweet_analyzer.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X Tweet Toplayici")
    parser.add_argument("--symbol",  type=str, help="Tek hisse (ornek: AAPL)")
    parser.add_argument("--months",  type=int, default=MONTHS_BACK, help="Kac ay geriye (varsayilan: 6)")
    parser.add_argument("--clean",   action="store_true", help="Onceden MongoDB'yi temizle")
    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else WATCHLIST
    months  = args.months

    print("Tweet Toplayici Basliyor")
    print(f"   Hisseler : {', '.join(symbols)}")
    print(f"   Donem    : Son {months} ay")
    print(f"   Mod      : EN POPULER (f=top)")
    print(f"   Hedef    : ~{TWEETS_PER_MONTH} tweet/ay/hisse\n")

    asyncio.run(main(symbols, months, args.clean))