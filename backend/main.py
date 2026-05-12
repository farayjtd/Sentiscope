from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import re
import os
import json
from playwright.async_api import async_playwright

app = FastAPI(title="SentiScope API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


class AnalyzeRequest(BaseModel):
    url: str


async def scrape_shopee(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ]
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            locale="id-ID",
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(4000)

            product_name = ""
            try:
                product_name = await page.locator("h1").first.inner_text(timeout=5000)
            except Exception:
                try:
                    product_name = await page.title()
                except Exception:
                    product_name = "Produk Shopee"

            rating = ""
            try:
                rating = await page.locator(
                    "[class*='rating'] span, [class*='score']"
                ).first.inner_text(timeout=3000)
            except Exception:
                pass

            sold = ""
            try:
                sold_el = page.locator("text=/terjual|sold/i").first
                sold = await sold_el.inner_text(timeout=3000)
            except Exception:
                pass

            price = ""
            try:
                price = await page.locator(
                    "[class*='price']:not([class*='original'])"
                ).first.inner_text(timeout=3000)
            except Exception:
                pass

            description = ""
            try:
                await page.locator("text=/deskripsi|description/i").first.click(timeout=3000)
                await page.wait_for_timeout(1000)
            except Exception:
                pass
            try:
                description = await page.locator(
                    "[class*='description'], [class*='desc']"
                ).first.inner_text(timeout=3000)
                description = description[:1000]
            except Exception:
                pass

            try:
                review_section = page.locator("text=/ulasan|rating & ulasan|review/i").first
                await review_section.scroll_into_view_if_needed(timeout=3000)
                await page.wait_for_timeout(2000)
            except Exception:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
                await page.wait_for_timeout(2000)

            reviews = []
            selectors = [
                "[class*='review'] [class*='content']",
                "[class*='comment'] [class*='text']",
                "[class*='shopee-product-comment'] span",
            ]
            for sel in selectors:
                els = await page.locator(sel).all()
                for el in els[:15]:
                    try:
                        text = await el.inner_text(timeout=2000)
                        text = text.strip()
                        if len(text) > 10 and text not in reviews:
                            reviews.append(text)
                    except Exception:
                        continue
                if reviews:
                    break

            return {
                "product_name": product_name.strip(),
                "price": price.strip(),
                "rating": rating.strip(),
                "sold": sold.strip(),
                "description": description.strip(),
                "reviews": reviews,
                "url": url,
            }

        finally:
            await browser.close()


def analyze_with_gemini(data: dict) -> dict:
    reviews_text = "\n".join([f"- {r}" for r in data["reviews"]]) if data["reviews"] else "Tidak ada ulasan yang berhasil diambil."

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
        raise HTTPException(status_code=500, detail=f"Gagal menganalisis dengan Gemini: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok"}