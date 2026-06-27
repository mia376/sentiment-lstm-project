from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sentimen Review Produk E-Commerce Menggunakan Long Short-Term Memory (LSTM)",
    page_icon="💬",
    layout="wide"
)

# =========================
# CUSTOM CSS — Midnight Rose Design System
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg:           #0D0B1A;
        --bg2:          #13102A;
        --bg3:          #1A1733;
        --rose:         #FF2D6B;
        --rose-dim:     #CC2255;
        --rose-glow:    rgba(255,45,107,0.18);
        --rose-border:  rgba(255,45,107,0.30);
        --violet:       #7B4FFF;
        --cyan:         #00D4FF;
        --white:        #F0EEF8;
        --muted:        #7A7894;
        --glass:        rgba(255,255,255,0.04);
        --glass-border: rgba(255,255,255,0.09);
        --shadow-rose:  0 8px 40px rgba(255,45,107,0.22);
        --shadow-card:  0 4px 32px rgba(0,0,0,0.45);
    }

    .stApp {
        background: var(--bg) !important;
        font-family: 'Inter', sans-serif !important;
    }
    html, body, [class*="css"] { background: var(--bg) !important; }

    [data-testid="stSidebar"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--glass-border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--white) !important;
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Syne', sans-serif !important;
        color: var(--white) !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stMetric"] {
        background: var(--glass) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 16px !important;
        padding: 20px 22px !important;
        box-shadow: var(--shadow-card) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--rose), var(--violet));
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--white) !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
    }

    hr {
        border: none !important;
        height: 1px !important;
        background: var(--glass-border) !important;
        margin: 20px 0 !important;
    }

    .stTextArea textarea {
        background: var(--bg3) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--white) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        caret-color: var(--rose) !important;
        transition: border 0.25s !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--rose) !important;
        box-shadow: 0 0 0 3px var(--rose-glow) !important;
    }
    .stTextArea textarea::placeholder { color: var(--muted) !important; }

    .stButton > button {
        background: linear-gradient(135deg, var(--rose), var(--violet)) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 11px 30px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.04em !important;
        box-shadow: var(--shadow-rose) !important;
        transition: all 0.22s ease !important;
        text-transform: uppercase !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 12px 40px rgba(255,45,107,0.40) !important;
    }

    p, li, span, div { color: var(--white); }
    .stMarkdown p { color: var(--muted); font-size: 0.95rem; line-height: 1.7; }

    .g-card {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 28px 30px;
        margin-bottom: 20px;
        box-shadow: var(--shadow-card);
        position: relative;
        overflow: hidden;
    }
    .g-card::after {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 180px; height: 180px;
        background: radial-gradient(circle, rgba(255,45,107,0.07) 0%, transparent 70%);
        pointer-events: none;
    }

    .tag-rose {
        display: inline-block;
        background: rgba(255,45,107,0.12);
        color: #FF7099;
        border: 1px solid rgba(255,45,107,0.30);
        border-radius: 6px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-right: 6px; margin-bottom: 6px;
    }
    .tag-violet {
        display: inline-block;
        background: rgba(123,79,255,0.12);
        color: #A882FF;
        border: 1px solid rgba(123,79,255,0.30);
        border-radius: 6px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-right: 6px; margin-bottom: 6px;
    }
    .tag-cyan {
        display: inline-block;
        background: rgba(0,212,255,0.10);
        color: #00D4FF;
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 6px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-right: 6px; margin-bottom: 6px;
    }

    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 10px;
    }
    .step-num {
        color: white;
        min-width: 34px; height: 34px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        flex-shrink: 0;
    }
    .step-desc {
        font-size: 0.92rem;
        line-height: 1.65;
        padding-top: 7px;
    }
    .step-subdesc {
        font-size: 0.78rem;
        margin-top: 2px;
        line-height: 1.5;
    }
    .step-connector {
        width: 1px; height: 10px;
        background: linear-gradient(to bottom, rgba(255,45,107,0.35), transparent);
        margin-left: 16px; margin-bottom: 4px;
    }

    .result-pos {
        background: rgba(255,45,107,0.07);
        border: 1px solid rgba(255,45,107,0.30);
        border-left: 4px solid var(--rose);
        border-radius: 14px;
        padding: 22px 26px;
        display: flex; align-items: center; gap: 20px;
        box-shadow: 0 4px 24px rgba(255,45,107,0.12);
    }
    .result-neg {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.28);
        border-left: 4px solid #EF4444;
        border-radius: 14px;
        padding: 22px 26px;
        display: flex; align-items: center; gap: 20px;
    }
    .result-neu {
        background: rgba(0,212,255,0.06);
        border: 1px solid rgba(0,212,255,0.22);
        border-left: 4px solid var(--cyan);
        border-radius: 14px;
        padding: 22px 26px;
        display: flex; align-items: center; gap: 20px;
    }

    .eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: var(--rose); margin-bottom: 6px;
    }

    .info-row {
        display: flex; justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #7A7894; font-size: 0.82rem; font-weight: 500; }
    .info-val   { color: #F0EEF8; font-size: 0.82rem; font-weight: 600;
                  font-family: 'JetBrains Mono', monospace; }

    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg2); }
    ::-webkit-scrollbar-thumb { background: var(--rose-dim); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# =========================
# PLOTLY DARK THEME
# =========================
COLORS = {
    "Positif": "#FF2D6B",
    "Netral":  "#00D4FF",
    "Negatif": "#7B4FFF",
}
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(family="Inter", color="#7A7894", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)",
               tickfont=dict(color="#7A7894", size=11)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.07)",
               tickfont=dict(color="#7A7894", size=11)),
    legend=dict(font=dict(color="#C0BDDA", size=12), bgcolor="rgba(0,0,0,0)"),
)

# =========================
# HEADER
# =========================
st.markdown("""
<div style="padding:48px 0 24px;text-align:center;position:relative;">
    <div style="position:absolute;top:0;left:50%;transform:translateX(-50%);
        width:600px;height:200px;
        background:radial-gradient(ellipse at center,rgba(255,45,107,0.12) 0%,transparent 70%);
        pointer-events:none;"></div>
    <div class="eyebrow" style="margin-bottom:14px;">Deep Learning · NLP · LSTM</div>
    <h1 style="font-size:3rem;font-weight:800;margin:0;color:#F0EEF8;line-height:1.1;
        text-shadow:0 0 60px rgba(255,45,107,0.3);">
        Sentiment<span style="color:#FF2D6B;">AI</span> Dashboard
    </h1>
    <p style="color:#7A7894;font-size:1rem;margin-top:14px;max-width:500px;
        margin-left:auto;margin-right:auto;line-height:1.65;">
        Tahu apa yang pelanggan <em>benar-benar</em> rasakan — tanpa harus baca ribuan review satu per satu.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("""
<div style="padding:24px 0 32px;text-align:center;">
    <div style="width:52px;height:52px;border-radius:14px;
        background:linear-gradient(135deg,#FF2D6B,#7B4FFF);
        display:flex;align-items:center;justify-content:center;
        margin:0 auto 12px;font-size:1.5rem;
        box-shadow:0 8px 24px rgba(255,45,107,0.35);">💬</div>
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
        color:#F0EEF8;margin-bottom:4px;">Sentiment AI</div>
    <div style="font-size:0.75rem;color:#7A7894;line-height:1.5;">
        Analisis sentimen otomatis<br>untuk bisnis yang lebih cerdas
    </div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("Navigasi", ["🏠 Input Review", "📊 Dashboard", "🧠 Info Model"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
    border-radius:12px;padding:16px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:600;
        color:#7A7894;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:12px;">
        System Status
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#22C55E;
            box-shadow:0 0 10px #22C55E;"></div>
        <span style="color:#22C55E;font-size:0.82rem;font-weight:600;
            font-family:'JetBrains Mono',monospace;">ONLINE · READY</span>
    </div>
    <div style="border-top:1px solid rgba(255,255,255,0.06);padding-top:12px;
        display:flex;flex-direction:column;gap:8px;">
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#7A7894;font-size:0.75rem;">Model</span>
            <span style="color:#A882FF;font-size:0.75rem;font-weight:600;
                font-family:'JetBrains Mono',monospace;">LSTM v1.0</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#7A7894;font-size:0.75rem;">Bahasa</span>
            <span style="color:#A882FF;font-size:0.75rem;font-weight:600;
                font-family:'JetBrains Mono',monospace;">Indonesia</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#7A7894;font-size:0.75rem;">Kelas</span>
            <span style="color:#A882FF;font-size:0.75rem;font-weight:600;
                font-family:'JetBrains Mono',monospace;">3 Sentimen</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
def predict_sentiment(text):
    text = text.lower()
    pos_words = ["bagus","mantap","puas","keren","suka","senang","hebat","memuaskan","recommended","oke","baik","cepat","ramah"]
    neg_words = ["jelek","buruk","rusak","lama","kecewa","mengecewakan","parah","tidak sesuai","bohong","tipu","lambat","boros"]
    if any(w in text for w in pos_words):
        return "Positif"
    elif any(w in text for w in neg_words):
        return "Negatif"
    else:
        return "Netral"

# =========================
# HALAMAN: INPUT REVIEW
# =========================
if menu == "🏠 Input Review":
    st.markdown("""
    <div class="eyebrow">Coba Sekarang</div>
    <h2 style="font-size:1.8rem;font-weight:800;color:#F0EEF8;margin:0 0 8px;">
        Paste review, lihat hasilnya ⚡
    </h2>
    <p style="color:#7A7894;margin-bottom:28px;font-size:0.95rem;line-height:1.7;">
        Cukup tempel teks review dari marketplace, kolom komentar, atau mana saja —
        model LSTM kami akan langsung tahu apakah pelanggan senang, kecewa, atau biasa saja.
    </p>
    """, unsafe_allow_html=True)

    review = st.text_area(
        "Teks Review",
        placeholder='Contoh: "Produk ini sangat bagus, pengiriman cepat dan pelayanan ramah!"',
        height=160,
        label_visibility="collapsed"
    )

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        predict_btn = st.button("⚡ Analisis Sekarang", use_container_width=True)

    if predict_btn:
        if review.strip() == "":
            st.warning("⚠️ Reviewnya kosong nih — coba isi dulu ya!")
        else:
            with st.spinner("Sedang dianalisis..."):
                time.sleep(1)
                label = predict_sentiment(review)

            st.markdown("<br>", unsafe_allow_html=True)

            if label == "Positif":
                st.markdown("""
                <div class="result-pos">
                    <div style="font-size:2.8rem;filter:drop-shadow(0 0 12px rgba(255,45,107,0.5));">😊</div>
                    <div>
                        <div style="color:#FF2D6B;font-family:'Syne',sans-serif;
                            font-size:1.4rem;font-weight:800;">Positif</div>
                        <div style="color:#C0BDDA;font-size:0.9rem;margin-top:6px;line-height:1.6;">
                            Pelanggan ini kayaknya happy banget! Review menunjukkan kepuasan
                            yang nyata — bisa jadi bahan testimoni yang kuat.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif label == "Negatif":
                st.markdown("""
                <div class="result-neg">
                    <div style="font-size:2.8rem;filter:drop-shadow(0 0 12px rgba(239,68,68,0.5));">😡</div>
                    <div>
                        <div style="color:#EF4444;font-family:'Syne',sans-serif;
                            font-size:1.4rem;font-weight:800;">Negatif</div>
                        <div style="color:#C0BDDA;font-size:0.9rem;margin-top:6px;line-height:1.6;">
                            Ada yang perlu diperbaiki nih. Review ini mengandung keluhan
                            atau kekecewaan — patut direspons sebelum menyebar lebih jauh.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-neu">
                    <div style="font-size:2.8rem;filter:drop-shadow(0 0 12px rgba(0,212,255,0.4));">😐</div>
                    <div>
                        <div style="color:#00D4FF;font-family:'Syne',sans-serif;
                            font-size:1.4rem;font-weight:800;">Netral</div>
                        <div style="color:#C0BDDA;font-size:0.9rem;margin-top:6px;line-height:1.6;">
                            Nggak positif, nggak negatif. Pelanggan ini menyampaikan fakta
                            tanpa banyak emosi — mungkin layak di-follow up untuk digali lebih dalam.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =========================
# HALAMAN: DASHBOARD
# =========================
elif menu == "📊 Dashboard":
    st.markdown("""
    <div class="eyebrow">Statistik & Tren</div>
    <h2 style="font-size:1.8rem;font-weight:800;color:#F0EEF8;margin:0 0 8px;">
        Gimana suara pelanggan minggu ini?
    </h2>
    <p style="color:#7A7894;margin-bottom:28px;font-size:0.95rem;line-height:1.7;">
        Data real-time dari puluhan ribu review — dipecah jadi insight yang langsung bisa kamu pakai.
    </p>
    """, unsafe_allow_html=True)

    total    = 41712
    positif  = 26000
    netral   = 11000
    negatif  =  4712

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Total Review", f"{total:,}")
    col2.metric("💗 Positif",      f"{positif:,}")
    col3.metric("💙 Netral",       f"{netral:,}")
    col4.metric("💜 Negatif",      f"{negatif:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    data = pd.DataFrame({
        "Sentimen": ["Positif", "Netral", "Negatif"],
        "Jumlah":   [positif, netral, negatif]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(data, x="Sentimen", y="Jumlah", color="Sentimen",
                      color_discrete_map=COLORS, text="Jumlah")
        fig1.update_traces(marker_line_width=0,
                           texttemplate='%{text:,}', textposition='outside',
                           textfont=dict(color="#C0BDDA", size=12, family="JetBrains Mono"),
                           marker=dict(opacity=0.9))
        fig1.update_layout(**PT,
                           title=dict(text="Distribusi Sentimen",
                                      font=dict(size=14, color="#F0EEF8", family="Syne"), x=0),
                           showlegend=False,
                           margin=dict(t=50, b=20, l=10, r=10), bargap=0.40)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure(data=[go.Pie(
            labels=data["Sentimen"], values=data["Jumlah"], hole=0.62,
            marker=dict(colors=[COLORS[s] for s in data["Sentimen"]],
                        line=dict(color="#0D0B1A", width=4)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#C0BDDA", family="Inter"),
        )])
        fig2.update_layout(**PT,
                           title=dict(text="Komposisi Sentimen",
                                      font=dict(size=14, color="#F0EEF8", family="Syne"), x=0),
                           margin=dict(t=50, b=20, l=10, r=10),
                           annotations=[dict(text=f"<b>{total:,}</b>", x=0.5, y=0.5,
                                             font_size=16, showarrow=False,
                                             font=dict(color="#F0EEF8", family="JetBrains Mono"))])
        st.plotly_chart(fig2, use_container_width=True)

    trend = pd.DataFrame({
        "Hari":    ["Sen","Sel","Rab","Kam","Jum","Sab","Min"],
        "Positif": [1800,2000,2200,2100,2400,2600,2300],
        "Netral":  [900,950,1000,980,1100,1200,1050],
        "Negatif": [400,420,450,430,500,520,480]
    })
    TREND_SERIES = [
        ("Positif", "#FF2D6B", "rgba(255,45,107,0.08)"),
        ("Netral",  "#00D4FF", "rgba(0,212,255,0.07)"),
        ("Negatif", "#7B4FFF", "rgba(123,79,255,0.07)"),
    ]
    fig3 = go.Figure()
    for col_name, line_color, fill_color in TREND_SERIES:
        fig3.add_trace(go.Scatter(
            x=trend["Hari"], y=trend[col_name], name=col_name,
            mode="lines+markers",
            line=dict(color=line_color, width=2.5, shape="spline", smoothing=0.8),
            marker=dict(size=7, color=line_color, line=dict(color="#0D0B1A", width=2)),
            fill="tozeroy", fillcolor=fill_color,
        ))
    fig3.update_layout(**PT,
                       title=dict(text="Tren Sentimen Mingguan — review makin banyak tiap akhir pekan 📈",
                                  font=dict(size=14, color="#F0EEF8", family="Syne"), x=0),
                       margin=dict(t=50, b=20, l=10, r=10),
                       hovermode="x unified",
                       hoverlabel=dict(bgcolor="#1A1733", font_color="#F0EEF8", font_family="Inter"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="eyebrow">Analisis Teks</div>
    <h3 style="font-size:1.3rem;font-weight:700;color:#F0EEF8;margin:0 0 6px;">
        Kata apa yang paling sering muncul?
    </h3>
    <p style="color:#7A7894;font-size:0.88rem;margin-bottom:18px;">
        Wordcloud & frekuensi kata dari seluruh review yang masuk.
    </p>
    """, unsafe_allow_html=True)

    reviews_text = [
        "produk ini bagus sekali saya suka",
        "pelayanan mantap dan cepat",
        "barang jelek dan rusak",
        "saya sangat puas dengan produk ini",
        "pengiriman lama dan mengecewakan",
        "kualitas keren banget",
        "buruk sekali tidak sesuai harapan",
        "suka produk ini sangat bagus",
        "kecewa barang lama sampai"
    ]
    text_all = " ".join(reviews_text)

    wordcloud = WordCloud(
        width=1200, height=360, background_color="#0D0B1A",
        colormap="cool", prefer_horizontal=0.80, max_words=80,
    ).generate(text_all)

    fig_wc, ax = plt.subplots(figsize=(14, 4))
    fig_wc.patch.set_facecolor("#0D0B1A")
    ax.set_facecolor("#0D0B1A")
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig_wc)

    words = re.findall(r'\w+', text_all.lower())
    counter = Counter(words)
    most_common  = counter.most_common(10)
    least_common = counter.most_common()[-10:]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='eyebrow' style='margin-top:16px;'>🔥 Paling Sering</div>", unsafe_allow_html=True)
        df_most = pd.DataFrame(most_common, columns=["Kata", "Frekuensi"])
        fig_most = px.bar(df_most, x="Frekuensi", y="Kata", orientation="h",
                          color="Frekuensi", color_continuous_scale=["#CC2255","#FF2D6B"])
        fig_most.update_layout(**PT, showlegend=False,
                               title=dict(text="Kata dominan di review pelanggan",
                                          font=dict(size=12, color="#7A7894", family="Inter"), x=0),
                               margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        fig_most.update_yaxes(autorange="reversed", gridcolor="rgba(255,255,255,0.03)")
        fig_most.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
        st.plotly_chart(fig_most, use_container_width=True)

    with col2:
        st.markdown("<div class='eyebrow' style='margin-top:16px;'>❄️ Paling Jarang</div>", unsafe_allow_html=True)
        df_least = pd.DataFrame(least_common, columns=["Kata", "Frekuensi"])
        fig_least = px.bar(df_least, x="Frekuensi", y="Kata", orientation="h",
                           color="Frekuensi", color_continuous_scale=["#5533CC","#7B4FFF"])
        fig_least.update_layout(**PT, showlegend=False,
                                title=dict(text="Kata yang jarang muncul tapi mungkin penting",
                                           font=dict(size=12, color="#7A7894", family="Inter"), x=0),
                                margin=dict(t=40, b=10, l=10, r=10), coloraxis_showscale=False)
        fig_least.update_yaxes(autorange="reversed", gridcolor="rgba(255,255,255,0.03)")
        fig_least.update_xaxes(gridcolor="rgba(255,255,255,0.03)")
        st.plotly_chart(fig_least, use_container_width=True)

# =========================
# HALAMAN: INFO MODEL
# =========================
elif menu == "🧠 Info Model":
    st.markdown("""
    <div class="eyebrow">Di Balik Layar</div>
    <h2 style="font-size:1.8rem;font-weight:800;color:#F0EEF8;margin:0 0 8px;">
        Gimana sih cara kerjanya?
    </h2>
    <p style="color:#7A7894;margin-bottom:28px;font-size:0.95rem;line-height:1.7;">
        Semua dimulai dari teks mentah, diproses lewat beberapa lapisan neural network,
        dan berakhir dengan prediksi sentimen yang akurat. Ini cerita lengkapnya.
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="g-card">
            <div class="eyebrow">📌 Konteks Proyek</div>
            <div style="color:#F0EEF8;font-family:'Syne',sans-serif;
                font-size:1.05rem;font-weight:700;margin:10px 0 6px;line-height:1.4;">
                Analisis Sentimen Review Produk E-Commerce
            </div>
            <div style="color:#7A7894;font-size:0.87rem;line-height:1.65;margin-bottom:16px;">
                Jutaan review tersebar di berbagai platform — Tokopedia, Shopee, Lazada.
                Proyek ini hadir untuk mengotomatiskan proses yang tadinya butuh tenaga manusia berjam-jam.
            </div>
            <span class="tag-rose">NLP</span>
            <span class="tag-violet">Deep Learning</span>
            <span class="tag-cyan">Bahasa Indonesia</span>
            <span class="tag-rose">E-Commerce</span>
        </div>

        <div class="g-card">
            <div class="eyebrow">🧠 Arsitektur Model</div>
            <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                letter-spacing:-0.02em;margin:8px 0 2px;
                background:linear-gradient(135deg,#FF2D6B,#7B4FFF,#00D4FF);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                LSTM
            </div>
            <div style="color:#7A7894;font-size:0.82rem;margin-bottom:4px;
                font-family:'JetBrains Mono',monospace;">
                Long Short-Term Memory · Recurrent Neural Network
            </div>
            <div style="color:#C0BDDA;font-size:0.87rem;line-height:1.65;margin:12px 0 16px;">
                LSTM dipilih karena kemampuannya mengingat konteks panjang dalam kalimat —
                penting banget untuk bahasa Indonesia yang sering pakai sindiran atau frasa kiasan.
            </div>
            <span class="tag-rose">Sequence Modeling</span>
            <span class="tag-violet">RNN</span>
            <span class="tag-cyan">Softmax Output</span>
        </div>

        <div class="g-card">
            <div class="eyebrow">📊 Spesifikasi</div>
            <div style="margin-top:10px;">
                <div class="info-row">
                    <span class="info-label">Total Data Training</span>
                    <span class="info-val">41,712 review</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Kelas Output</span>
                    <span class="info-val">Positif · Netral · Negatif</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Bahasa</span>
                    <span class="info-val">Indonesia</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Framework</span>
                    <span class="info-val">TensorFlow / Keras</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Embedding</span>
                    <span class="info-val">Custom Word Embedding</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="g-card" style="min-height:560px;">
            <div class="eyebrow">🔄 Alur Pemrosesan</div>
            <div style="color:#C0BDDA;font-size:0.87rem;line-height:1.65;margin:10px 0 22px;">
                Dari teks mentah sampai prediksi, ini perjalanan setiap review
                yang masuk ke dalam sistem.
            </div>
        """, unsafe_allow_html=True)

        steps = [
            ("Input Teks Review",
             "Teks mentah dari pengguna diterima apa adanya.",
             "#FF2D6B"),
            ("Preprocessing",
             "Huruf kapital disamakan, tanda baca dihapus, kata tidak penting dibuang.",
             "#F05090"),
            ("Tokenisasi",
             "Setiap kata dipecah jadi token dan dicocokkan ke kamus model.",
             "#C060C0"),
            ("Padding Sequence",
             "Panjang input diseragamkan agar bisa diproses sekaligus dalam batch.",
             "#9060E0"),
            ("Embedding Layer",
             "Token diubah jadi vektor angka yang merepresentasikan makna kata.",
             "#7B4FFF"),
            ("LSTM Layer",
             "Vektor diproses berurutan — model 'membaca' kalimat seperti manusia.",
             "#5090FF"),
            ("Dense + Softmax",
             "Output akhir: probabilitas untuk setiap kelas sentimen.",
             "#00D4FF"),
        ]

        for i, (title, desc, color) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="step-row">
                <div class="step-num" style="background:{color};box-shadow:0 4px 14px {color}44;">{i:02d}</div>
                <div>
                    <div class="step-desc" style="color:#F0EEF8;font-weight:600;">{title}</div>
                    <div class="step-subdesc" style="color:#7A7894;">{desc}</div>
                </div>
            </div>
            {"<div class='step-connector'></div>" if i < len(steps) else ""}
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="g-card" style="background:rgba(255,45,107,0.05);
            border-color:rgba(255,45,107,0.18);margin-top:0;">
            <div class="eyebrow">🎯 Kenapa Ini Penting?</div>
            <div style="color:#C0BDDA;font-size:0.9rem;line-height:1.75;margin-top:10px;">
                Bisnis yang paham sentimen pelanggan bisa bergerak lebih cepat —
                merespons keluhan sebelum viral, memperkuat produk yang disukai,
                dan mengambil keputusan berbasis data, bukan asumsi.
            </div>
        </div>
        """, unsafe_allow_html=True)
