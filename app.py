import streamlit as st
import streamlit.components.v1 as components
from similarity_engine import SimilarityEngine
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Similarity Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def clean_val(val, fallback="—"):
    if val is None: return fallback
    s = str(val).strip()
    return fallback if s.lower() in ("nan","none","","n/a") else s

def clean_category(val):
    s = clean_val(val)
    if s == "—": return "—"
    s = re.sub(r"^\[[\"\']?", "", s)
    s = re.sub(r"[\"\'].*$", "", s)
    s = s.split(">>")[0].strip().strip('"').strip("'")
    return s if s else "—"

def clean_price(val):
    try:
        f = float(val)
        if f > 0: return f"₹{f:,.0f}", f
    except: pass
    return None, None

def clean_image(val):
    s = clean_val(val)
    return s if (s != "—" and s.startswith("http")) else None

def format_price(price_val, retail_val):
    p_str, p_float = clean_price(price_val)
    r_str, r_float = clean_price(retail_val)
    if p_str is None and r_str is None:
        return '<span style="color:#a09890;font-style:italic;font-size:0.85rem;">Price not available</span>'
    if p_str is None:
        return f'<span style="color:#ff8c00;font-weight:600;">{r_str}</span>'
    if r_str is None or r_float <= p_float:
        return f'<span style="color:#ff8c00;font-weight:600;">{p_str}</span>'
    disc = round((1 - p_float / r_float) * 100)
    return (f'<span style="color:#ff8c00;font-weight:600;font-size:1.05rem;">{p_str}</span> '
            f'<small style="color:#605c6e;text-decoration:line-through;">{r_str}</small> '
            f'<small style="background:#e8f5ee;color:#2d7a4f;padding:1px 6px;border-radius:99px;font-weight:700;">−{disc}%</small>')

def img_box(url, height=300, radius=14):
    bg = "#f8f5f0"
    if url:
        return (f'<div style="width:100%;height:{height}px;background:{bg};border-radius:{radius}px;'
                f'overflow:hidden;display:flex;align-items:center;justify-content:center;">'
                f'<img src="{url}" style="width:100%;height:100%;object-fit:contain;padding:12px;"'
                f' onerror="this.parentElement.innerHTML=\'<div style=background:#f0ece4;width:100%;'
                f'height:100%;display:flex;align-items:center;justify-content:center;color:#a09890;'
                f'font-size:0.7rem;letter-spacing:0.1em>NO IMAGE</div>\'"/></div>')
    return (f'<div style="width:100%;height:{height}px;background:#f0ece4;border-radius:{radius}px;'
            f'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;">'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 100 100">'
            f'<g transform="translate(50,50)">'
            f'<rect x="-22" y="-10" width="44" height="34" rx="5" fill="none" stroke="#b8b0a4" stroke-width="4"/>'
            f'<path d="M-12,-10 Q-12,-24 0,-24 Q12,-24 12,-10" fill="none" stroke="#b8b0a4" stroke-width="4"/>'
            f'</g></svg>'
            f'<span style="font-family:sans-serif;font-size:0.7rem;color:#a09890;letter-spacing:0.1em">NO IMAGE</span>'
            f'</div>')

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"]{
    background:#080810 !important;color:#e8e4dc !important;
    font-family:'DM Sans',sans-serif !important;}
[data-testid="stAppViewContainer"]{
    background:
        radial-gradient(ellipse 70% 45% at 20% 10%,rgba(255,140,0,0.09),transparent),
        radial-gradient(ellipse 60% 40% at 80% 85%,rgba(200,40,80,0.07),transparent),
        #080810 !important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stDecoration"]{display:none;}
[data-testid="stMainBlockContainer"]{padding-top:0 !important;}
[data-testid="stHorizontalBlock"]{gap:1rem;align-items:flex-end;}

[data-testid="stTextInput"]>label{font-family:'Syne',sans-serif !important;font-size:0.62rem !important;
    font-weight:700 !important;letter-spacing:0.2em !important;text-transform:uppercase !important;color:#ff8c00 !important;}
[data-testid="stTextInput"]>div>div{background:rgba(255,255,255,0.04) !important;
    border:1.5px solid rgba(255,255,255,0.09) !important;border-radius:14px !important;}
[data-testid="stTextInput"] input{background:transparent !important;color:#f0ece4 !important;
    font-family:'DM Sans',sans-serif !important;font-size:0.95rem !important;padding:0.85rem 1.1rem !important;}
[data-testid="stTextInput"] input::placeholder{color:#45424f !important;}

[data-testid="stSelectbox"]>label{font-family:'Syne',sans-serif !important;font-size:0.62rem !important;
    font-weight:700 !important;letter-spacing:0.2em !important;text-transform:uppercase !important;color:#9994a8 !important;}
[data-testid="stSelectbox"]>div>div{background:rgba(255,255,255,0.04) !important;
    border:1.5px solid rgba(255,255,255,0.09) !important;border-radius:14px !important;
    color:#f0ece4 !important;font-size:0.9rem !important;}

.stButton>button{background:linear-gradient(135deg,#ff8c00,#ff3c64) !important;
    color:#fff !important;font-family:'Syne',sans-serif !important;font-size:0.8rem !important;
    font-weight:700 !important;letter-spacing:0.1em !important;text-transform:uppercase !important;
    border:none !important;border-radius:14px !important;padding:0.8rem 2.5rem !important;
    width:100% !important;box-shadow:0 4px 20px rgba(255,60,100,0.3) !important;
    transition:opacity 0.2s,transform 0.15s !important;}
.stButton>button:hover{opacity:0.88 !important;transform:translateY(-2px) !important;}

.sh{font-family:'Syne',sans-serif;font-size:0.6rem;font-weight:700;letter-spacing:0.24em;
    text-transform:uppercase;color:#9994a8;margin:2rem 0 1.2rem;display:flex;align-items:center;gap:1rem;}
.sh::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.06);}
.sl{font-family:'Syne',sans-serif;font-size:0.6rem;font-weight:700;letter-spacing:0.24em;
    text-transform:uppercase;color:#ff8c00;margin:1.5rem 0 1rem;display:flex;align-items:center;gap:0.6rem;}
.sl::after{content:'';flex:1;height:1px;background:rgba(255,140,0,0.15);}

.pnt{font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:#f0ece4;
    line-height:1.25;margin-bottom:1.2rem;letter-spacing:-0.02em;}
.mg{display:grid;grid-template-columns:1fr 1fr;gap:1rem 2rem;}
.mi{display:flex;flex-direction:column;gap:0.2rem;}
.mk{font-size:0.58rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#45424f;}
.mv{font-size:0.92rem;color:#c8c4bc;}
.mv.price{color:#ff8c00;font-weight:600;font-size:1.05rem;}
.mv.rating{color:#f5c842;}
.mv.cat{font-size:0.72rem;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
    border-radius:6px;padding:0.2rem 0.65rem;display:inline-block;color:#9994a8;}

.sp{background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.055);
    border-radius:20px;padding:1.6rem 1.8rem 1.4rem;margin:0.5rem 0 1rem;position:relative;overflow:hidden;}
.sp::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,140,0,0.3),transparent);}
.nr{text-align:center;padding:3rem;color:#45424f;font-size:0.95rem;}
</style>""", unsafe_allow_html=True)

# ─── Load Engine ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_engine():
    return SimilarityEngine("data/products_clean.csv")

with st.spinner("⚙️ Loading similarity engine…"):
    engine = load_engine()
df = engine.data

# ─── Session State ─────────────────────────────────────────────────────────────
if "cat_history" not in st.session_state:
    st.session_state.cat_history = []       # list of (product_name, category) tuples
if "last_tracked" not in st.session_state:
    st.session_state.last_tracked = None    # last product_name that was tracked

# ─── Sidebar Analytics ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Session Analytics")
    if st.session_state.cat_history:
        cat_names = [c for _, c in st.session_state.cat_history]
        counts    = Counter(cat_names)
        top3      = counts.most_common(3)
        medals    = ["🥇","🥈","🥉"]
        st.markdown("**🔥 Top Categories Explored**")
        for i,(cat,n) in enumerate(top3):
            st.markdown(f"{medals[i]} **{cat}** &nbsp;·&nbsp; {n} search{'es' if n>1 else ''}")
        st.markdown("---")
        st.metric("Total Searches", len(st.session_state.cat_history))
        st.metric("Unique Categories", len(counts))
        if st.button("🗑 Clear History", key="clear_hist"):
            st.session_state.cat_history = []
            st.session_state.last_tracked = None
            st.rerun()
    else:
        st.caption("Search for products to see your analytics here.")

# ─── Hero ──────────────────────────────────────────────────────────────────────
total  = len(df)
cats_n = df['category'].nunique() if 'category' in df.columns else "—"
brnd_n = df['brand'].nunique()    if 'brand'    in df.columns else "—"

try:
    sample_imgs = df[df['image'].notna()].sample(min(8,len(df)))['image'].tolist()
except:
    sample_imgs = []

tile_pos = [("4%","10%","108px","−3deg","0.9"),("62%","5%","95px","6deg","0.8"),
            ("78%","20%","118px","−5deg","0.85"),("2%","54%","100px","8deg","0.75"),
            ("72%","62%","105px","−8deg","0.8"),("20%","70%","90px","4deg","0.7"),
            ("50%","68%","115px","−3deg","0.85"),("40%","4%","88px","10deg","0.75")]
tiles = ""
for i,url in enumerate(sample_imgs[:8]):
    if i>=len(tile_pos): break
    l,t,sz,r,op = tile_pos[i]
    tiles += (f'<div style="position:absolute;left:{l};top:{t};width:{sz};height:{sz};'
              f'transform:rotate({r});opacity:{op};background:rgba(255,255,255,0.07);'
              f'backdrop-filter:blur(6px);border:1px solid rgba(255,255,255,0.12);'
              f'border-radius:16px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,0.3);'
              f'animation:fl 6s ease-in-out infinite alternate;">'
              f'<img src="{url}" style="width:100%;height:100%;object-fit:contain;padding:6px;"'
              f' onerror="this.parentElement.style.display=\'none\'"/></div>')

hero = f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:'DM Sans',sans-serif;overflow:hidden;}}
@keyframes fl{{0%{{transform:translateY(0)}}100%{{transform:translateY(-10px)}}}}
@keyframes sh{{to{{background-position:200% center;}}}}
.wrap{{position:relative;width:100%;height:420px;
    background:linear-gradient(135deg,#0d0b12 0%,#130f1a 40%,#1a0e10 70%,#0f0c18 100%);
    border-radius:24px;overflow:hidden;display:flex;align-items:center;justify-content:center;}}
.wrap::before{{content:'';position:absolute;inset:0;
    background:radial-gradient(ellipse 55% 45% at 15% 20%,rgba(255,140,0,0.18),transparent),
               radial-gradient(ellipse 50% 40% at 85% 75%,rgba(220,50,90,0.15),transparent);}}
.glass{{position:relative;z-index:10;background:rgba(10,8,16,0.55);backdrop-filter:blur(18px);
    border:1px solid rgba(255,255,255,0.1);border-radius:20px;padding:2.8rem 3.5rem 2.4rem;
    max-width:580px;text-align:center;box-shadow:0 24px 60px rgba(0,0,0,0.4);}}
.eye{{display:inline-block;font-size:0.58rem;font-weight:600;letter-spacing:0.22em;text-transform:uppercase;
    color:#ff8c00;background:rgba(255,140,0,0.1);border:1px solid rgba(255,140,0,0.25);
    border-radius:99px;padding:0.28rem 1rem;margin-bottom:1.1rem;}}
h1{{font-family:'Syne',sans-serif;font-size:2.5rem;font-weight:800;line-height:1.08;
    letter-spacing:-0.03em;color:#f2ede4;margin-bottom:0.8rem;}}
.gr{{background:linear-gradient(120deg,#ff8c00,#ff3c64,#ff8c00);background-size:200% auto;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    animation:sh 3s linear infinite;}}
.sub{{font-size:0.85rem;color:#8a8494;font-weight:300;line-height:1.6;margin-bottom:1.8rem;}}
.stats{{display:flex;justify-content:center;gap:2rem;padding-top:1.4rem;
    border-top:1px solid rgba(255,255,255,0.07);}}
.sn{{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
    background:linear-gradient(135deg,#ff8c00,#ffbf00);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;}}
.sl2{{font-size:0.58rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;color:#55516a;}}
.div{{width:1px;background:rgba(255,255,255,0.08);align-self:stretch;}}
</style></head><body>
<div class="wrap">
    {tiles}
    <div class="glass">
        <div class="eye">✦ &nbsp;Powered by TF-IDF · Flipkart Dataset</div>
        <h1>Find <span class="gr">Similar</span><br>Products Instantly</h1>
        <p class="sub">Search across {total:,} products · matches by description,<br>category & semantic similarity</p>
        <div class="stats">
            <div><div class="sn">{total:,}</div><div class="sl2">Products</div></div>
            <div class="div"></div>
            <div><div class="sn">{cats_n}</div><div class="sl2">Categories</div></div>
            <div class="div"></div>
            <div><div class="sn">{brnd_n}</div><div class="sl2">Brands</div></div>
        </div>
    </div>
</div>
</body></html>"""
components.html(hero, height=432, scrolling=False)

# ─── Search Panel ──────────────────────────────────────────────────────────────


raw_cats = df['category'].dropna().unique().tolist()
cleaned_cats = sorted(set(clean_category(c) for c in raw_cats if clean_category(c) != "—"))
cat_to_raw = {}
for raw in raw_cats:
    cat_to_raw.setdefault(clean_category(raw), []).append(raw)

col_s, col_f = st.columns([3, 1])
with col_s:
    q = st.text_input("🔍  Search for a product",
        placeholder="e.g. wireless earbuds, cotton kurti, laptop bag…")
with col_f:
    cf = st.selectbox("Filter by Category", ["All Categories"] + cleaned_cats)



fdf = df[df["category"].isin(cat_to_raw.get(cf, []))] if cf != "All Categories" else df
matches = fdf[fdf["product_name"].str.contains(q, case=False, na=False)] if q else fdf

if len(matches) == 0:
    st.markdown('<div class="nr">😕 No products found. Try a different search.</div>', unsafe_allow_html=True)
    st.stop()

choice = st.selectbox(f"Select a product ({len(matches):,} results)", matches["product_name"].tolist())
_, bcol, _ = st.columns([1, 2, 1])
with bcol:
    clicked = st.button("✦ Find Similar Products")

# ═══════════════════════════════════════════════════════════════════════════════
if clicked:
    idx     = df[df["product_name"] == choice].index[0]
    sel     = df.iloc[idx]
    sel_cat = clean_category(sel.get("category",""))

    # ── Track ONLY if this is a NEW product (not a re-click on same one) ──────
    if choice != st.session_state.last_tracked:
        if sel_cat and sel_cat != "—":
            st.session_state.cat_history.append((choice, sel_cat))
        st.session_state.last_tracked = choice

    # ── Selected Product ──────────────────────────────────────────────────────
    st.markdown('<div class="sl">✦ &nbsp; Selected Product</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    with c1:
        ib = img_box(clean_image(sel.get("image")), height=300)
        components.html(f"<!DOCTYPE html><html><body style='margin:0;background:transparent'>{ib}</body></html>", height=308)
    with c2:
        st.markdown(f'<div class="pnt">{sel["product_name"]}</div>', unsafe_allow_html=True)
        brand    = clean_val(sel.get("brand"))
        category = clean_category(sel.get("category",""))
        rating   = clean_val(sel.get("overall_rating"))
        rat_disp = f"★ {rating}" if rating!="—" else '<span style="color:#605c6e;font-style:italic;">No rating</span>'
        pr_disp  = format_price(sel.get("discounted_price"), sel.get("retail_price"))
        st.markdown(f"""<div class="mg">
            <div class="mi"><span class="mk">Brand</span><span class="mv">{brand}</span></div>
            <div class="mi"><span class="mk">Rating</span><span class="mv rating">{rat_disp}</span></div>
            <div class="mi"><span class="mk">Price</span><span class="mv price">{pr_disp}</span></div>
            <div class="mi"><span class="mk">Category</span><span class="mv cat">{category[:40]}</span></div>
        </div>""", unsafe_allow_html=True)

    # ── Similar Products ──────────────────────────────────────────────────────
    st.markdown('<div class="sh">✦ &nbsp; Top Similar Products — Same Category</div>', unsafe_allow_html=True)
    results = engine.get_similar_products(idx)

    if not results:
        st.markdown('<div class="nr">No similar products found.</div>', unsafe_allow_html=True)
    else:
        ci = ""
        for r in results:
            sc = r['score']
            col = "#2d7a4f" if sc>=0.7 else ("#b85c00" if sc>=0.4 else "#c0392b")
            bg  = "#e8f5ee"  if sc>=0.7 else ("#fff3e0"  if sc>=0.4 else "#fdecea")

            ps, pf = clean_price(r.get('price'))
            rs, rf = clean_price(r.get('retail_price'))
            if ps is None and rs is None:
                pr = '<span style="color:#a09890;font-style:italic;font-size:0.72rem;">Price unavailable</span>'
            elif ps is None:
                pr = f'<span class="pp">{rs}</span>'
            elif rs is None or (rf and rf<=pf):
                pr = f'<span class="pp">{ps}</span>'
            else:
                d = round((1-pf/rf)*100) if rf else 0
                pr = f'<span class="pp">{ps}</span>'
                if d>0: pr += f'<span class="pr">{rs}</span><span class="pd">−{d}%</span>'

            iu = clean_image(r.get("image"))
            if iu:
                ih = (f'<div class="iw"><img src="{iu}" class="im"'
                      f' onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'"/>'
                      f'<div class="ni" style="display:none">No Image</div>'
                      f'<span class="badge" style="color:{col};background:{bg};">{sc:.0%}</span></div>')
            else:
                ih = (f'<div class="iw" style="background:#f0ece4;">'
                      f'<div class="ni" style="display:flex">No Image</div>'
                      f'<span class="badge" style="color:{col};background:{bg};">{sc:.0%}</span></div>')

            br = clean_val(r.get("brand"))
            rt = clean_val(r.get("rating"))
            ct = clean_category(r.get("category",""))
            de = clean_val(r.get("description",""), fallback="")
            de = de[:100]+"…" if len(de)>100 else de
            try:
                sv=float(rt); fl=int(sv); em=5-fl
                stars=(f'<span style="color:#e8a000">{"★"*fl}</span>'
                       f'<span style="color:#ddd">{"☆"*em}</span>'
                       f' <span style="color:#999;font-size:0.6rem">({rt})</span>')
            except:
                stars='<span style="color:#bbb;font-size:0.68rem;font-style:italic;">No rating</span>'

            ci += (f'<div class="card">{ih}<div class="body">'
                   f'<div class="cat">{ct}</div>'
                   f'<div class="nm">{r["name"]}</div>'
                   f'{"<div class=br>"+br+"</div>" if br!="—" else ""}'
                   f'{"<div class=de>"+de+"</div>" if de else ""}'
                   f'<div class="dv"></div>'
                   f'<div class="st">{stars}</div>'
                   f'<div class="pr-row">{pr}</div>'
                   f'</div></div>')

        cards_html = f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:'DM Sans',sans-serif;}}
.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;padding:4px 2px 18px;}}
.card{{background:#fff;border-radius:12px;overflow:hidden;display:flex;flex-direction:column;
    box-shadow:0 1px 4px rgba(0,0,0,0.06);transition:transform 0.22s,box-shadow 0.22s;cursor:pointer;}}
.card:hover{{transform:translateY(-5px);box-shadow:0 16px 40px rgba(0,0,0,0.13);}}
.iw{{position:relative;width:100%;height:180px;background:#f5f2ed;overflow:hidden;flex-shrink:0;}}
.im{{width:100%;height:100%;object-fit:contain;padding:12px;display:block;}}
.ni{{width:100%;height:100%;align-items:center;justify-content:center;
    background:#f0ece4;color:#a09890;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;}}
.badge{{position:absolute;top:8px;left:8px;font-family:'DM Sans',sans-serif;
    font-size:0.6rem;font-weight:700;padding:3px 9px;border-radius:99px;}}
.body{{padding:13px;display:flex;flex-direction:column;gap:4px;flex:1;}}
.cat{{font-size:0.55rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:#b0a898;}}
.nm{{font-family:'Playfair Display',serif;font-size:0.85rem;font-weight:600;color:#1a1814;
    line-height:1.35;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}}
.br{{font-size:0.6rem;color:#999;text-transform:uppercase;letter-spacing:0.08em;}}
.de{{font-size:0.67rem;color:#7a7570;line-height:1.5;display:-webkit-box;
    -webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;margin-top:2px;}}
.dv{{width:28px;height:1.5px;background:#e0d8ce;margin:6px 0;}}
.st{{font-size:0.72rem;margin-bottom:2px;}}
.pr-row{{display:flex;align-items:baseline;gap:5px;flex-wrap:wrap;}}
.pp{{font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:#c0392b;}}
.pr{{font-size:0.67rem;color:#bbb;text-decoration:line-through;}}
.pd{{font-size:0.6rem;font-weight:700;color:#2d7a4f;background:#e8f5ee;padding:1px 6px;border-radius:99px;}}
</style></head><body>
<div class="grid">{ci}</div>
</body></html>"""
        components.html(cards_html, height=520, scrolling=False)

    # ════════════════════════════════════════════════════════════════════════════
    # ▶  FEATURE 1 — Cosine Similarity Bar Chart
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sh">📊 &nbsp; Cosine Similarity Scores</div>', unsafe_allow_html=True)

    try:
        names_s = [r['name'][:40]+"…" if len(r['name'])>40 else r['name'] for r in results]
        scores  = [r['score'] for r in results]
        bcolors = ["#4caf7d" if s>=0.7 else ("#ff8c00" if s>=0.4 else "#ff3c64") for s in scores]

        fig, ax = plt.subplots(figsize=(10, 3.2))
        fig.patch.set_facecolor('#0f0d18')
        ax.set_facecolor('#0f0d18')

        ax.barh(names_s, [1]*len(names_s), color='#1a1728', height=0.55, zorder=1)
        bars = ax.barh(names_s, scores, color=bcolors, height=0.55, zorder=2, alpha=0.92)

        for bar, score in zip(bars, scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', ha='left',
                    color='#e8e4dc', fontsize=9.5, fontfamily='monospace', fontweight='bold')

        ax.set_xlim(0, 1.18)
        ax.set_xlabel('Similarity Score', color='#45424f', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a1728')
        ax.xaxis.grid(True, color='#1a1728', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors='#9994a8', labelsize=9)
        plt.xticks(color='#45424f')
        plt.tight_layout(pad=1.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.caption(f"Chart error: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # ▶  FEATURE 3 — Similarity Heatmap
    # ════════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sh">🔥 &nbsp; Similarity Heatmap — Selected vs Top Matches</div>', unsafe_allow_html=True)

    try:
        r_indices = []
        for r in results:
            m = df[df["product_name"] == r['name']]
            if len(m): r_indices.append(m.index[0])

        all_idx  = [idx] + r_indices
        all_lbl  = ["★ Selected"] + [
            (r['name'][:20]+"…" if len(r['name'])>20 else r['name'])
            for r in results[:len(r_indices)]
        ]
        sub = engine.similarity_matrix[np.ix_(all_idx, all_idx)]

        cmap = mcolors.LinearSegmentedColormap.from_list("flame", [
            (0.00,"#0d0b12"),(0.30,"#1a0e18"),(0.55,"#5c1a00"),
            (0.75,"#c04800"),(0.88,"#ff8c00"),(1.00,"#ffe066")
        ])

        fig2, ax2 = plt.subplots(figsize=(8, 5.5))
        fig2.patch.set_facecolor('#0f0d18')
        ax2.set_facecolor('#0f0d18')

        im = ax2.imshow(sub, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        for i in range(len(all_lbl)):
            for j in range(len(all_lbl)):
                v = sub[i,j]
                ax2.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                         color='#000' if v>0.7 else '#e8e4dc', fontweight='bold')

        ax2.set_xticks(range(len(all_lbl)))
        ax2.set_yticks(range(len(all_lbl)))
        ax2.set_xticklabels(all_lbl, rotation=35, ha='right', color='#9994a8', fontsize=8)
        ax2.set_yticklabels(all_lbl, color='#9994a8', fontsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#1a1728')

        cb = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.03)
        cb.ax.tick_params(colors='#9994a8', labelsize=8)
        cb.outline.set_edgecolor('#1a1728')

        plt.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
    except Exception as e:
        st.caption(f"Heatmap error: {e}")

    # ════════════════════════════════════════════════════════════════════════════
    # ▶  FEATURE 3 — Session Analytics: Top Categories
    # ════════════════════════════════════════════════════════════════════════════
    if len(st.session_state.cat_history) >= 1:
        st.markdown('<div class="sh">📈 &nbsp; Your Session — Top Explored Categories</div>', unsafe_allow_html=True)

        cat_names_only = [c for _, c in st.session_state.cat_history]
        counts  = Counter(cat_names_only)
        top5    = counts.most_common(5)
        t_names = [c for c,_ in top5]
        t_vals  = [n for _,n in top5]
        pal     = ["#ff8c00","#ff3c64","#a259ff","#00c9a7","#ffd600"]

        fig3, ax3 = plt.subplots(figsize=(9, 2.8))
        fig3.patch.set_facecolor('#0f0d18')
        ax3.set_facecolor('#0f0d18')

        max_v = max(t_vals) if t_vals else 1
        ax3.barh(t_names, [max_v]*len(t_names), color='#1a1728', height=0.5, zorder=1)
        ax3.barh(t_names, t_vals, color=pal[:len(t_names)], height=0.5, zorder=2, alpha=0.9)

        for name, val in zip(t_names, t_vals):
            ax3.text(val + max_v*0.02, t_names.index(name), f' {val} search{"es" if val>1 else ""}',
                     va='center', color='#e8e4dc', fontsize=9.5, fontweight='bold')

        ax3.set_xlim(0, max_v * 1.5)
        ax3.set_xlabel('Searches', color='#45424f', fontsize=9)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#1a1728')
        ax3.xaxis.grid(True, color='#1a1728', linewidth=0.8)
        ax3.set_axisbelow(True)
        ax3.tick_params(colors='#9994a8', labelsize=9)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        m1, m2, m3 = st.columns(3)
        with m1: st.metric("🔎 Total Searches", len(st.session_state.cat_history))
        with m2: st.metric("📁 Unique Categories", len(counts))
        with m3: st.metric("🏆 Favourite", top5[0][0] if top5 else "—")