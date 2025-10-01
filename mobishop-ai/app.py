import os, json, time, re, html, random, difflib
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS, cross_origin
from annoy import AnnoyIndex
import requests
from openai import OpenAI

# ===== Intenti =====
INTENT_GIFT = re.compile(r'\b(dāvināt|dāvana|ko uzdāvināt)\b', re.I)
QUESTION_WORDS = {"kāpēc","kapec","kas","kā","kur","kad","cik","vai","kāda","kāds","kamdēļ","kādēļ"}

BOOT_TS = int(time.time())
BOOT_ID = f"{BOOT_TS}-{os.getpid()}"

BASE_DIR = os.path.dirname(__file__)
CONF_DIR = os.path.join(BASE_DIR, "config")

# ===== ENV =====
WC_BASE = os.getenv("WC_BASE", "").rstrip("/")
CK = os.getenv("WC_CK")
CS = os.getenv("WC_CS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")  # izmantošanai admin UI
if not (WC_BASE and CK and CS and OPENAI_API_KEY):
    raise RuntimeError("ENV trūkst: WC_BASE / WC_CK / WC_CS / OPENAI_API_KEY")

# ===== OpenAI =====
def _make_openai():
    return OpenAI(api_key=OPENAI_API_KEY)
client = _make_openai()

# ===== Konfigurācijas =====
STATE: Dict[str, Any] = {"settings": None, "keywords": None, "ranking": None, "prompt": None, "mtimes": {}}

def _load_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".txt"):
            return f.read()
        return json.load(f)

def load_config(force: bool=False):
    files = {
        "settings": os.path.join(CONF_DIR, "settings.json"),
        "keywords": os.path.join(CONF_DIR, "keywords.json"),
        "ranking": os.path.join(CONF_DIR, "ranking.json"),
        "prompt":   os.path.join(CONF_DIR, "prompt.txt"),
    }
    for key, path in files.items():
        mtime = os.path.getmtime(path)
        if force or STATE["mtimes"].get(key) != mtime:
            STATE[key] = _load_file(path)
            STATE["mtimes"][key] = mtime

# Ielādējam sākumā
load_config(force=True)

EMB_MODEL = STATE["settings"]["embed_model"]
CHAT_MODEL = STATE["settings"]["chat_model"]

# ===== Metadati + Annoy =====
META_PATH  = os.path.join(BASE_DIR, "meta.json")
INDEX_PATH = os.path.join(BASE_DIR, "annoy.index")
with open(META_PATH, "r", encoding="utf-8") as f:
    META: List[Dict[str, Any]] = json.load(f)

def embed(text: str):
    r = client.embeddings.create(model=EMB_MODEL, input=[text])
    return r.data[0].embedding

_dim = len(embed("Sveiki!"))
ann = AnnoyIndex(_dim, "angular")
ann.load(INDEX_PATH)

# ===== Teksta normalizācija / vārdu vārdnīca (nosaukumiem) =====
_LAT_CHARS = "a-z0-9āčēģīķļņōŗšūž"
_RE_NORM = re.compile(f"[^{_LAT_CHARS}]+", re.IGNORECASE)

def _norm(s: str) -> str:
    return _RE_NORM.sub(" ", (s or "").lower()).strip()

def _tokens(s: str, min_len: int = 3) -> List[str]:
    return [w for w in _norm(s).split() if len(w) >= min_len]

def _title_vocab() -> set:
    vocab = set()
    for m in META:
        t = m.get("title") or m.get("name") or ""
        for w in _tokens(t):
            vocab.add(w)
    return vocab

TITLE_VOCAB = _title_vocab()

def title_has_query_anywhere(query: str) -> bool:
    """Vai kāds produkts katalogā satur KĀDU no vaicājuma tokeniem kā pilnu vārdu savā nosaukumā?"""
    q_tokens = _tokens(query)
    if not q_tokens:
        return False
    for m in META:
        t = m.get("title") or m.get("name") or ""
        tset = set(_tokens(t))
        if any(tok in tset for tok in q_tokens):
            return True
    return False

def suggest_similar_words(query: str, n: int = 5, cutoff: float = 0.72) -> List[str]:
    """Meklē līdzīgus vārdus no nosaukumu vārdnīcas (difflib)."""
    found = set()
    for tok in _tokens(query):
        for match in difflib.get_close_matches(tok, TITLE_VOCAB, n=n, cutoff=cutoff):
            found.add(match)
    return sorted(found)[:n]

# ===== Flask + CORS =====
app = Flask(__name__)
@app.after_request
def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp

CORS(app, resources={r"/chat": {"origins": STATE["settings"]["cors"]}}, supports_credentials=False)

# ===== Woo =====
def wc_get(path, params=None):
    params = params or {}
    params.update({"consumer_key": CK, "consumer_secret": CS})
    r = requests.get(f"{WC_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def wc_search(query: str, limit: int = 6):
    try:
        items = wc_get("/products", {"search": query, "per_page": limit})
    except Exception:
        return []
    out = []
    for p in items:
        out.append({
            "id": p["id"],
            "title": p.get("name",""),
            "price": p.get("price_html") or p.get("price"),
            "in_stock": (p.get("stock_status") == "instock"),
            "image": (p.get("images") or [{}])[0].get("src"),
            "link": p.get("permalink"),
            "add_url": f"{WC_BASE.split('/wp-json')[0]}/?add-to-cart={p['id']}&quantity=1",
        })
    return out

# ===== RAG palīgfunkcijas =====
def item_text_for_match(m: Dict[str, Any]) -> str:
    return (
        (m.get("title") or "") + " " +
        (m.get("permalink") or "") + " " +
        (m.get("categories") or "") + " " +
        (m.get("description") or "")
    ).lower()

def annoy_top_k(query: str, k: int):
    vec = embed(query)
    ids, dists = ann.get_nns_by_vector(vec, k, include_distances=True)
    out = []
    for i in ids:
        if 0 <= i < len(META):
            out.append(META[i])
    return out

def filter_by_keywords(items, words):
    wl = [w.lower() for w in words]
    res = []
    for m in items:
        t = item_text_for_match(m)
        if any(w in t for w in wl):
            res.append(m)
    return res

def search_by_keywords_exact_partial(query: str, limit: int = 9) -> List[Dict[str, Any]]:
    """
    Vienkārša precīza/daļēja meklēšana pa nosaukumiem + aprakstiem.
    Dod priekšroku pilnām sakritībām nosaukumā, pēc tam aprakstā, tad daļējām.
    """
    q_tokens = _tokens(query)
    if not q_tokens:
        return []

    scored = []
    for m in META:
        title = (m.get("title") or m.get("name") or "").lower()
        desc  = (m.get("description") or "").lower()
        if not title and not desc:
            continue

        # pilnie vārdi (tokenos)
        tset = set(_tokens(title))
        dset = set(_tokens(desc))

        score = 0
        for tok in q_tokens:
            if tok in tset: score += 4        # pilns vārds nosaukumā
            if tok in dset: score += 2        # pilns vārds aprakstā
            # daļējās sakritības (prefikss / apakšvirkne)
            if tok not in tset and tok in title: score += 1
            if tok not in dset  and tok in desc:  score += 0.5

        if score > 0:
            scored.append((score, m))

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [normalize_meta(m) for score, m in scored[:limit]]
    return hits

# === Telefoni: baterija + kamera (helperi) ===
_RE_MAH = re.compile(r'(\d{3,5})\s*mAh', re.I)
_RE_MP  = re.compile(r'(\d{2,3})\s*(MP|megapikseļ|megapikse|megapixel)', re.I)

def _price_num(m: Dict[str, Any]) -> float:
    v = m.get("price") or m.get("price_html") or ""
    if isinstance(v, (int, float)): return float(v)
    s = re.sub(r"[^\d.,]", "", str(v))
    s = s.replace(",", ".")
    try:
        return float(re.search(r"\d+(\.\d+)?", s).group(0))
    except Exception:
        return float("inf")

def _extract_battery_mah(text: str) -> Optional[int]:
    if not text: return None
    m = _RE_MAH.search(text)
    if not m: return None
    try:
        val = int(m.group(1))
        if 500 <= val <= 10000: return val
    except: pass
    return None

def _extract_camera_mp(text: str) -> Optional[int]:
    if not text: return None
    m = _RE_MP.search(text)
    if not m: return None
    try:
        val = int(m.group(1))
        if 5 <= val <= 250: return val
    except: pass
    return None

def _is_phone_meta(m: Dict[str, Any]) -> bool:
    ttl = (m.get("title") or m.get("name") or "").lower()
    cats = (m.get("categories") or "").lower()
    return any(k in ttl for k in ["telefon", "iphone", "samsung", "galaxy"]) or \
           any(k in cats for k in ["telefon", "phone"])

def find_phones_ranked_battery_camera(limit: int = 12) -> List[Dict[str, Any]]:
    scored = []
    for m in META:
        if not _is_phone_meta(m):
            continue
        text = " ".join([
            (m.get("title") or ""),
            (m.get("description") or ""),
            json.dumps(m.get("attributes", "")) if isinstance(m.get("attributes"), (list,dict)) else str(m.get("attributes") or "")
        ])
        b = _extract_battery_mah(text)
        c = _extract_camera_mp(text)
        if b is None and c is None:
            continue
        # Normalizējam aptuveni: baterija 3000–6000 mAh; kamera 12–60 MP
        b_norm = 0.0 if b is None else max(0.0, min(1.0, (b - 3000) / 3000))
        c_norm = 0.0 if c is None else max(0.0, min(1.0, (c - 12) / 48))
        score = b_norm + c_norm
        scored.append((score, m, b or 0, c or 0))

    if not scored:
        return []

    # paņemam TOP pēc score un tad cenu ↑
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max(limit, 6)]
    top.sort(key=lambda x: _price_num(x[1]))

    return [normalize_meta(m) for _, m, _, _ in top[:limit]]

def normalize_meta(m: Dict[str, Any]):
    return {
        "id": m.get("id"),
        "title": m.get("title") or m.get("name") or "",
        "price": m.get("price_html") or m.get("price"),
        "in_stock": (m.get("stock_status") == "instock") if m.get("stock_status") is not None else True,
        "image": m.get("image") or ((m.get("images") or [{}])[0].get("src") if isinstance(m.get("images"), list) else None),
        "link": m.get("permalink"),
        "add_url": f"{WC_BASE.split('/wp-json')[0]}/?add-to-cart={m.get('id')}&quantity=1" if m.get("id") else None,
    }

def detect_profile(q: str):
    ql = q.lower()
    for prof in STATE["keywords"]:
        if any(k in ql for k in prof["match"]):
            return prof
    return {}

def search_products(query: str):
    k = int(STATE["ranking"].get("annoy_top_k", 20))
    use_woo_fallback = bool(STATE["ranking"].get("woo_fallback", True))
    profile = detect_profile(query)
    annoy_items = annoy_top_k(query, k)

    if profile:
        filtered = filter_by_keywords(annoy_items, profile["match"])
        if filtered:
            items = filtered
        else:
            return wc_search(query) if use_woo_fallback else []
    else:
        items = annoy_items

    if not items and use_woo_fallback:
        return wc_search(query)

    return [normalize_meta(m) for m in items]

def pick_random_products(k: int = 6) -> List[Dict[str, Any]]:
    pool = [normalize_meta(m) for m in META if m.get("id")]
    if len(pool) <= k:
        return pool
    return random.sample(pool, k)

# ===== Atbildes ģenerēšana =====
def make_answer(user_q: str, n_products: int):
    system_prompt = STATE["prompt"] or "Tu esi e-veikala palīgs. Atbildi latviski, īsi; nerādi URL vai sarakstus tekstā."
    msgs = [
        {"role":"system","content": system_prompt},
        {"role":"user","content": f"Jautājums: {user_q}\nAtrasti produkti: {n_products}. Tekstā neliec linkus."}
    ]
    try:
        r = client.chat.completions.create(model=STATE["settings"]["chat_model"], messages=msgs, temperature=0.3)
        raw = (r.choices[0].message.content or "").strip()
        raw = re.sub(r"https?://\S+","", raw)
        return re.sub(r"\n{1,}", "\n\n", raw).strip() or "Lūdzu, apskati ieteikumus zemāk."
    except Exception:
        return "Lūdzu, apskati ieteikumus zemāk."

def make_no_title_match_answer(user_q: str, suggestions: List[str]) -> str:
    q_esc = user_q.strip()
    if suggestions:
        sug = ", ".join(suggestions)
        return f'Tu meklēji pēc vārda "{q_esc}" un mēs neatradām tiešu sakarību produktu nosaukumos. Varbūt domāji: {sug}. Zemāk parādām līdzīgus variantus:'
    else:
        return f'Tu meklēji pēc vārda "{q_esc}" un mēs neatradām tiešu sakarību produktu nosaukumos. Zemāk parādām līdzīgus vai populārus variantus:'

# ===== Publiskie API =====
@app.get("/health")
def health():
    load_config(force=False)
    return {
        "ok": True,
        "time": int(time.time()),
        "boot_ts": BOOT_TS,
        "pid": os.getpid(),
    }

def _detect_mobile_battery_camera(q: str) -> bool:
    ql = q.lower()
    has_phone = any(k in ql for k in ["mobilais","telefons","iphone","samsung","galaxy"])
    has_batt  = any(k in ql for k in ["baterija","akumulators","akums"])
    has_cam   = "kamera" in ql
    return has_phone and has_batt and has_cam

@app.route("/chat", methods=["POST","OPTIONS"])
def chat():
    # CORS preflight
    if request.method == "OPTIONS":
        return ("", 204)

    load_config(force=False)

    data = request.get_json(silent=True) or {}
    q = (
        data.get("message")
        or data.get("query")
        or data.get("q")
        or data.get("text")
        or data.get("prompt")
        or data.get("question")
        or ""
    )
    q = q.strip()

    if not q:
        return jsonify({"error":"Tukšs jautājums"}), 400

    # Dāvanu nolūka jautājums – vispirms precizē, nerādām preces
    if INTENT_GIFT.search(q):
        return jsonify({
            "mode": "need_gift_clarify",
            "text": "Kam meklē dāvanu? Izvēlies:",
            "chips": [
                {"t":"Sievietei"},{"t":"Vīrietim"},{"t":"Mammai"},{"t":"Tēvam"},
                {"t":"Brālim"},{"t":"Māsai"},{"t":"Bērnam"},
                {"t":"0–6"},{"t":"7–12"},{"t":"13–17"},{"t":"18–25"},{"t":"26–40"},{"t":"40+"},
                {"t":"IT/Programmētājam"},{"t":"Sportistam"},{"t":"Māksliniekam"},{"t":"Mājsaimniekam"}
            ],
            "stop_products": True
        })

    # CHAT režīms: ja vaicājums izklausās pēc jautājuma, nerādām produktus
    q_tokens = set(_tokens(q)) if '_tokens' in globals() else set(q.lower().split())
    if any((w in q_tokens) or (w in q.lower()) for w in QUESTION_WORDS):
        text = (f'Tu uzrakstīji “{q}”. Izskatās pēc jautājuma, nevis preču meklējuma. '
                f'Par ko tieši — par kādu preci, piegādi, cenu vai garantiju?')
        return jsonify({"mode":"chat","text":text,"products":[]})

    # Telefoni ar labu bateriju un kameru
    if _detect_mobile_battery_camera(q):
        phones = find_phones_ranked_battery_camera(limit=12)
        return jsonify({
            "mode": "phone_bc",
            "text": "Lūk telefoni ar labāko bateriju un kameru (cenā no mazākas uz lielāku).",
            "items": phones
        })

    # Precīza/daļēja atslēgvārdu meklēšana pa title/description (≤9)
    keyword_hits = search_by_keywords_exact_partial(q, limit=9)
    if keyword_hits:
        return jsonify({
            "mode": "keyword_hits",
            "items": keyword_hits
        })

    # Primārā meklēšana (embeddings/Annoy + profili)
    try:
        items = search_products(q)
    except Exception:
        items = []

    # Ja nevienā nosaukumā nav pilna vaicājuma vārda — “no_title_match” plūsma
    try:
        no_title_match = not title_has_query_anywhere(q)
    except Exception:
        no_title_match = True

    if no_title_match:
        try:
            suggestions = suggest_similar_words(q, n=5, cutoff=0.72)
        except Exception:
            suggestions = []
        candidates = items[:6] if items else pick_random_products(6)

        # svaiginām pirmos 3 ar Woo
        fresh = []
        for m in candidates[:3]:
            pid = m.get("id")
            if not pid:
                fresh.append(m); continue
            try:
                p = wc_get(f"/products/{pid}")
                fresh.append({
                    "id": p["id"],
                    "title": p.get("name",""),
                    "price": p.get("price_html") or p.get("price"),
                    "in_stock": p.get("stock_status")=="instock",
                    "image": (p.get("images") or [{}])[0].get("src"),
                    "link": p.get("permalink"),
                    "add_url": f"{WC_BASE.split('/wp-json')[0]}/?add-to-cart={p['id']}&quantity=1",
                })
            except Exception:
                fresh.append(m)

        answer = make_no_title_match_answer(q, suggestions)
        return jsonify({
            "mode":"no_title_match",
            "text": answer,
            "suggestions": suggestions,
            "products": fresh + candidates[3:6]
        })

    # Parastā plūsma (ir atbilstības nosaukumos)
    fresh = []
    for m in items[:3]:
        pid = m.get("id")
        if not pid:
            fresh.append(m); continue
        try:
            p = wc_get(f"/products/{pid}")
            fresh.append({
                "id": p["id"],
                "title": p.get("name",""),
                "price": p.get("price_html") or p.get("price"),
                "in_stock": p.get("stock_status")=="instock",
                "image": (p.get("images") or [{}])[0].get("src"),
                "link": p.get("permalink"),
                "add_url": f"{WC_BASE.split('/wp-json')[0]}/?add-to-cart={p['id']}&quantity=1",
            })
        except Exception:
            fresh.append(m)

    answer = make_answer(q, len(fresh))
    return jsonify({"text": answer, "products": fresh})

# ===== Admin: reload (API) =====
@app.route("/admin/reload", methods=["GET","POST"])
def admin_reload():
    if ADMIN_TOKEN and request.args.get("token") != ADMIN_TOKEN and request.headers.get("X-Admin-Token") != ADMIN_TOKEN:
        return ("", 403)
    load_config(force=True)
    global EMB_MODEL, CHAT_MODEL, client, TITLE_VOCAB
    EMB_MODEL = STATE["settings"]["embed_model"]
    CHAT_MODEL = STATE["settings"]["chat_model"]
    client = _make_openai()
    TITLE_VOCAB = _title_vocab()
    return {"reloaded": True, "time": int(time.time())}

# ===== Admin: vienkāršs WEB UI =====
ADMIN_FILES = [
    ("prompt",   "prompt.txt",   "Teksta uzvedne (system prompt)"),
    ("keywords", "keywords.json","Atslēgvārdu profili (precēm/tematiem)"),
    ("ranking",  "ranking.json", "Meklēšanas svari (annoy_top_k, woo_fallback)"),
    ("settings", "settings.json","Modeļi un CORS iestatījumi")
]

def _admin_guard():
    if not ADMIN_TOKEN:
        return None
    token = request.args.get("token") or request.headers.get("X-Admin-Token")
    if token != ADMIN_TOKEN:
        return Response("Forbidden", status=403)
    return None

@app.get("/admin/ui")
def admin_ui():
    guard = _admin_guard()
    if guard: return guard
    rows = []
    for key, fname, title in ADMIN_FILES:
        rows.append(f"""
          <tr>
            <td style='padding:6px 10px'>{html.escape(title)}</td>
            <td style='padding:6px 10px;opacity:.7'>{html.escape(fname)}</td>
            <td style='padding:6px 10px'><a href='/admin/edit?name={html.escape(fname)}&token={html.escape(request.args.get("token",""))}'>Rediģēt</a></td>
          </tr>
        """)
    token_q = f"?token={html.escape(request.args.get('token',''))}" if request.args.get("token") else ""
    body = f"""
      <html><head><meta charset='utf-8'><title>Admin konfigurācija</title>
      <style>
        body{{font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial; padding:20px}}
        table{{border-collapse:collapse}} td,th{{border:1px solid #e5e7eb}}
        .btn{{padding:6px 10px; background:#111; color:#fff; text-decoration:none; border-radius:6px}}
      </style></head><body>
      <h2>Konfigurācijas faili</h2>
      <table>
        <tr><th style='padding:8px 10px'>Nosaukums</th><th style='padding:8px 10px'>Fails</th><th style='padding:8px 10px'>Darbība</th></tr>
        {''.join(rows)}
      </table>
      <p style='margin-top:16px'><a class='btn' href='/admin/reload{token_q}'>Pārlādēt konfigurāciju</a></p>
      </body></html>
    """
    return Response(body, mimetype="text/html")

@app.get("/admin/edit")
def admin_edit():
    guard = _admin_guard()
    if guard: return guard
    name = (request.args.get("name") or "").strip()
    path = os.path.join(CONF_DIR, name)
    if not os.path.isfile(path):
        return Response("Fails nav atrasts", status=404)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    is_json = name.endswith(".json")
    area = html.escape(content)
    token = html.escape(request.args.get("token",""))
    body = f"""
      <html><head><meta charset='utf-8'><title>Edit {html.escape(name)}</title>
      <style>
        body{{font:14px/1.4 system-ui, -apple-system, Segoe UI, Roboto, Arial; padding:20px}}
        textarea{{width:100%; height:60vh; font:13px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace}}
        .row{{margin:10px 0}}
        .btn{{padding:8px 12px; background:#111; color:#fff; border:0; border-radius:6px; cursor:pointer}}
        .bar{{display:flex; gap:10px; align-items:center; margin-bottom:10px}}
      </style></head><body>
      <div class='bar'>
        <a href='/admin/ui?token={token}'>← Atpakaļ</a>
        <h3 style='margin:0 0 0 10px'>Rediģē: {html.escape(name)}</h3>
      </div>
      <form method='post' action='/admin/save'>
        <input type='hidden' name='name' value='{html.escape(name)}'>
        <input type='hidden' name='token' value='{token}'>
        <div class='row'>
          <textarea name='content' spellcheck='false'>{area}</textarea>
        </div>
        <div class='row'>
          <button class='btn' type='submit'>Saglabāt</button>
          <a class='btn' href='/admin/reload?token={token}' style='background:#0b5cab'>Pārlādēt</a>
        </div>
        <p style='opacity:.7'>{"JSON fails — validē saturu, pirms saglabā." if is_json else "Teksta fails."}</p>
      </form>
      </body></html>
    """
    return Response(body, mimetype="text/html")

@app.post("/admin/save")
def admin_save():
    guard = _admin_guard()
    if guard: return guard
    name = (request.form.get("name") or "").strip()
    token = (request.form.get("token") or "").strip()
    content = request.form.get("content") or ""
    path = os.path.join(CONF_DIR, name)
    if not os.path.isfile(path):
        return Response("Fails nav atrasts", status=404)

    try:
        if name.endswith(".json"):
            json.loads(content)
    except Exception as e:
        return Response(f"JSON kļūda: {html.escape(str(e))}", status=400)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    load_config(force=True)
    global EMB_MODEL, CHAT_MODEL, client, TITLE_VOCAB
    EMB_MODEL = STATE["settings"]["embed_model"]
    CHAT_MODEL = STATE["settings"]["chat_model"]
    client = _make_openai()
    TITLE_VOCAB = _title_vocab()

    return Response(f"<html><body><script>location.href='/admin/ui?token={html.escape(token)}'</script>Saglabāts.</body></html>", mimetype="text/html")

if __name__ == "__main__":
    app.run("0.0.0.0", 8000, debug=True)

@app.get("/")
def root():
    return {"ok": True, "health": "/health", "chat": "/chat"}

# ===== Alias maršruti zem /ai saderībai =====
@app.get("/ai/health")
def health_ai():
    return health()

@app.route("/ai/chat", methods=["POST","OPTIONS"])
@cross_origin(origins=STATE["settings"]["cors"])
def chat_ai():
    return chat()
