from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import re
import os
import json
import httpx
from bs4 import BeautifulSoup

app = FastAPI(title="SentiScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://shopee.co.id/",
}


class AnalyzeRequest(BaseModel):
    url: str


def extract_shopee_ids(url: str):
    match = re.search(r"i\.(\d+)\.(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    match = re.search(r"-i\.(\d+)\.(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    return None, None


async def fetch_shopee_api(shop_id: str, item_id: str) -> dict:
    """Fetch product data from Shopee's internal API."""
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=20) as client:

        # Product detail
        product_url = f"https://shopee.co.id/api/v4/item/get?itemid={item_id}&shopid={shop_id}"
        product_resp = await client.get(product_url)
        product_data = product_resp.json()
        item = product_data.get("data", {}) or {}

        product_name = item.get("name", "Produk Shopee")
        price_raw = item.get("price", 0)
        price = f"Rp {int(price_raw / 100000):,}".replace(",", ".") if price_raw else ""
        rating = str(item.get("item_rating", {}).get("rating_star", ""))
        sold = str(item.get("historical_sold", ""))
        description = (item.get("description", "") or "")[:1000]

        # Reviews
        reviews = []
        review_url = (
            f"https://shopee.co.id/api/v2/item/get_ratings"
            f"?itemid={item_id}&shopid={shop_id}&limit=20&offset=0&type=0"
        )
        review_resp = await client.get(review_url)
        review_data = review_resp.json()
        ratings_list = review_data.get("data", {}).get("ratings", []) or []
        for r in ratings_list:
            comment = r.get("comment", "").strip()
            if comment and len(comment) > 5:
                reviews.append(comment)

        return {
            "product_name": product_name,
            "price": price,
            "rating": rating,
            "sold": sold,
            "description": description,
            "reviews": reviews,
        }


async def scrape_shopee(url: str) -> dict:
    # Resolve short URLs
    async with httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=15) as client:
        resp = await client.get(url)
        final_url = str(resp.url)

    shop_id, item_id = extract_shopee_ids(final_url)

    if not shop_id or not item_id:
        raise Exception("Tidak dapat membaca ID produk dari URL. Pastikan link langsung ke halaman produk Shopee.")

    return await fetch_shopee_api(shop_id, item_id)


def analyze_with_gemini(data: dict) -> dict:
    reviews_text = "\n".join([f"- {r}" for r in data["reviews"]]) if data["reviews"] else "Tidak ada ulasan."

    prompt = f"""Kamu adalah analis sentimen produk e-commerce profesional. Analisis data produk Shopee berikut dan berikan laporan lengkap dalam Bahasa Indonesia.

=== DATA PRODUK ===
Nama Produk: {data['product_name']}
Harga: {data['price']}
Rating: {data['rating']}
Terjual: {data['sold']}
Deskripsi: {data['description']}

=== ULASAN PEMBELI ===
{reviews_text}

=== INSTRUKSI ===
Berikan analisis dalam format JSON berikut (HANYA JSON, tanpa teks lain, tanpa markdown):

{{
  "product_name": "nama produk",
  "overall_sentiment": "Positif" | "Negatif" | "Campuran",
  "sentiment_score": angka 0-100,
  "summary": "ringkasan 2-3 kalimat tentang produk dan persepsi pembeli",
  "positives": ["poin positif 1", "poin positif 2"],
  "negatives": ["poin negatif 1", "poin negatif 2"],
  "key_themes": ["tema utama 1", "tema utama 2"],
  "recommendation": "rekomendasi 2-3 kalimat: apakah produk ini worth it, untuk siapa, dan saran pembelian",
  "buyer_profile": "profil pembeli yang cocok untuk produk ini",
  "red_flags": ["red flag jika ada"],
  "verdict": "BELI" | "PERTIMBANGKAN" | "HINDARI"
}}"""

    response = model.generate_content(prompt)
    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    url = req.url.strip()
    if "shopee" not in url.lower():
        raise HTTPException(status_code=400, detail="Hanya mendukung link Shopee.")

    try:
        scraped = await scrape_shopee(url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil data dari Shopee: {str(e)}")

    try:
        result = analyze_with_gemini(scraped)
        result["raw_data"] = {
            "rating": scraped.get("rating"),
            "sold": scraped.get("sold"),
            "price": scraped.get("price"),
            "review_count": len(scraped.get("reviews", [])),
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menganalisis: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}