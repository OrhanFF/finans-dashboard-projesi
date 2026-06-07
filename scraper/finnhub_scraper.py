import requests
import pandas as pd
from datetime import datetime, timedelta

# Frontend/Backend kodlarında kullandığın API anahtarını buraya entegre ettik
FINNHUB_API_KEY = "cu37qopr01qure9c388gcu37qopr01qure9c3890"

def fetch_historical_news(symbol, start_date, end_date):
    print(f"📰 [{symbol}] için {start_date} ile {end_date} arası gerçek haberler çekiliyor...")
    
    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": symbol,
        "from": start_date,
        "to": end_date,
        "token": FINNHUB_API_KEY
    }
    
    # API'ye isteği atıyoruz
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"❌ API Hatası: {response.status_code} - {response.text}")
        return pd.DataFrame()
        
    news_data = response.json()
    print(f"✅ Toplam {len(news_data)} adet gerçek haber başarıyla çekildi!")
    
    # Gelen veriyi yapay zekanın yiyeceği formata (DataFrame) dönüştürelim
    processed_news = []
    for item in news_data:
        # Sadece başlığı ve özeti olan anlamlı haberleri alıyoruz
        if item.get('headline') and item.get('summary'):
            # Finnhub tarihi UNIX timestamp (saniye) olarak veriyor, bunu YYYY-MM-DD formatına çeviriyoruz
            dt_object = datetime.fromtimestamp(item['datetime'])
            date_str = dt_object.strftime('%Y-%m-%d')
            
            # NLP modelinin bağlamı daha iyi anlaması için başlık ve özeti birleştiriyoruz
            full_text = f"{item['headline']}. {item['summary']}"
            
            processed_news.append({
                'Date': date_str,
                'News_Text': full_text
            })
            
    df = pd.DataFrame(processed_news)
    return df

if __name__ == "__main__":
    # Son 1 yılın tarihlerini dinamik olarak hesaplıyoruz
    bugun = datetime.now()
    gecen_yil = bugun - timedelta(days=365)
    
    end_date_str = bugun.strftime('%Y-%m-%d')
    start_date_str = gecen_yil.strftime('%Y-%m-%d')
    
    # Apple'ın son 1 yıllık gerçek haberlerini çekelim
    gercek_haberler_df = fetch_historical_news("AAPL", start_date_str, end_date_str)
    
    if not gercek_haberler_df.empty:
        print("\n📊 İlk 5 Gerçek Haber Satırı:")
        print(gercek_haberler_df.head())
        
        print("\n📅 En Çok Haber Çıkan 5 Gün:")
        print(gercek_haberler_df['Date'].value_counts().head())