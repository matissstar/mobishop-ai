import os, re, json, time
from typing import List, Dict
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Annoy (cosine/Angular)
from annoy import AnnoyIndex

# OpenAI embeddings
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()
WC_BASE = os.getenv("WC_BASE", "").rstrip("/")
CK = os.getenv("WC_CK")
CS = os.getenv("WC_CS")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

assert WC_BASE and CK and CS, "Trūkst WooCommerce konfigurācijas .env"
assert OPENAI_API_KEY, "Trūkst OPENAI_API_KEY .env"

# OpenAI klients
client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI else None

def strip_html(s: str) -> str:
    if not s: return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"&[a-zA-Z#0-9]+;", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def wc_get(path: str, params: Dict=None):
    params = params or {}
    params.update({"consumer_key": CK, "consumer_secret": CS})
    r = requests.get(f"{WC_BASE}{path}", params=params, timeout=60)
    r.raise_for_status()
    return r

def fetch_all_products() -> List[Dict]:
    per_page = 100
    page = 1
    items = []
    while True:
        r = wc_get("/products", {"per_page": per_page, "page": page, "status": "publish"})
        arr = r.json()
        if not arr:
            break
        items.extend(arr)
        # mēģinām sekot lapošanai, bet ja header nav — ejam līdz tukšai lapai
        total_pages = int(r.headers.get("X-WP-TotalPages", page))
        if page >= total_pages: break
        page += 1
    return items

def build_doc(p: Dict) -> str:
    cats = " / ".join([c.get("name","") for c in p.get("categories",[]) if c.get("name")])
    attrs = []
    for a in p.get("attributes",[]):
        name = a.get("name") or ""
        vals = ", ".join(a.get("options",[]) or [])
        if name or vals:
            attrs.append(f"{name}: {vals}")
    attr_text = " | ".join(attrs)
    parts = [
        p.get("name",""),
        cats,
        attr_text,
        strip_html(p.get("short_description","")),
        strip_html(p.get("description","")),
        f"SKU: {p.get('sku') or ''}"
    ]
    return " \n".join([x for x in parts if x]).strip()

def embed_batch(texts: List[str]) -> List[List[float]]:
    # OpenAI embeddings ar nelielu throttling
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def main():
    print("⇒ Lejupielāde no WooCommerce…", flush=True)
    prods = fetch_all_products()
    if not prods:
        print("⚠️ Nav atrastu produktu (publicēti).")
        return
    print(f"✓ Produkti: {len(prods)}")

    corpus, meta = [], []
    for p in prods:
        corpus.append(build_doc(p))
        meta.append({
            "id": p["id"],
            "title": p.get("name",""),
            "price": p.get("price"),
            "price_html": p.get("price_html"),
            "stock_status": p.get("stock_status"),
            "image": (p.get("images") or [{}])[0].get("src"),
            "permalink": p.get("permalink"),
            "sku": p.get("sku")
        })

    # Embeddings partijās
    batch = 96
    vectors = []
    print("⇒ Ģenerēju embeddings…", flush=True)
    for i in tqdm(range(0, len(corpus), batch), desc="Embeddings"):
        chunk = corpus[i:i+batch]
        for t in range(3):  # līdz 3 mēģinājumiem
            try:
                embs = embed_batch(chunk)
                vectors.extend(embs)
                time.sleep(0.05)
                break
            except Exception as e:
                wait = 1.5 * (t+1)
                print(f"Embedding kļūda ({e}), gaidu {wait}s un mēģinu vēlreiz…", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError("Embedding izsaukumi neizdevās atkārtoti.")

    if not vectors:
        raise RuntimeError("Nesanāca iegūt embeddings.")

    dim = len(vectors[0])
    print(f"✓ Dimensija: {dim}")

    # Annoy indekss ar 'angular' (cosine līdzība)
    print("⇒ Būvēju Annoy indeksu…", flush=True)
    index = AnnoyIndex(dim, 'angular')
    for idx, vec in enumerate(vectors):
        index.add_item(idx, vec)
    index.build(50)  # 50 koki – labs kompromiss
    index.save("annoy.index")

    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print("✓ GATAVS: faili 'annoy.index' un 'meta.json' izveidoti.")

if __name__ == "__main__":
    main()
