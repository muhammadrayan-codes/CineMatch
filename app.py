import streamlit as st
import pickle
import pandas as pd
import torch
import torch.nn as nn
import requests
import re
import json
import numpy as np

# ── Model definition ──────────────────────────────────────────────────────────
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, items):
        return (self.user_emb(users) * self.item_emb(items)).sum(dim=1)

# ── Load ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/svd_model_cpu.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    return model.cpu(), enc

@st.cache_data
def load_movies():
    df = pd.read_csv("data/movies.csv")
    all_genres = set()
    for g in df["genres"].str.split("|"):
        all_genres.update(g)
    return df, sorted(list(all_genres))

@st.cache_data
def load_ratings():
    # ML-1M format: UserID::MovieID::Rating::Timestamp
    return pd.read_csv("data/ml-1m/ratings.dat", sep="::", engine="python", 
                       names=["user_id", "item_id", "rating", "timestamp"])

@st.cache_data
def get_user_history(user_id, ratings_df, movies_df):
    user_ratings = ratings_df[ratings_df["user_id"] == user_id].sort_values("rating", ascending=False)
    history = []
    for _, row in user_ratings.head(10).iterrows():
        movie = movies_df[movies_df["item_id"] == row["item_id"]]
        if not movie.empty:
            history.append({
                "title": movie.iloc[0]["title"],
                "genres": movie.iloc[0]["genres"],
                "user_rating": row["rating"]
            })
    return history

@st.cache_data(show_spinner=False)
def get_movie_info(title):
    try:
        # 1. Parse title and year: "Toy Story (1995)" -> "Toy Story", "1995"
        match = re.search(r"^(.*) \((\d{4})\)$", title)
        if match:
            raw_title, year = match.groups()
        else:
            raw_title, year = title, None

        # 2. Normalize title: "American President, The" -> "The American President"
        if ", The" in raw_title:
            raw_title = "The " + raw_title.replace(", The", "").strip()
        elif ", A" in raw_title:
            raw_title = "A " + raw_title.replace(", A", "").strip()
        elif ", An" in raw_title:
            raw_title = "An " + raw_title.replace(", An", "").strip()

        # Remove parentheticals like "Seven (Se7en)" -> "Seven"
        raw_title = re.sub(r" \(.*\)", "", raw_title).strip()
        headers = {'User-Agent': 'CineMatch/1.0 (contact@example.com)'}

        def fetch_summary(slug):
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug.replace(' ', '_')}"
            try:
                res = requests.get(url, headers=headers, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    # Only return if there's actually a thumbnail
                    if data.get("thumbnail"):
                        return data.get("thumbnail", {}).get("source"), data.get("extract", "")[:300]
            except: pass
            return None, None

        # 3. Try multiple variants for direct slugs
        variants = [raw_title]
        if year:
            variants.append(f"{raw_title} ({year} film)")
        variants.append(f"{raw_title} (film)")

        for v in variants:
            poster, plot = fetch_summary(v)
            if poster:
                return poster, plot, "N/A", ""

        # 4. Fallback: Search API to find the best page title
        search_query = f"{raw_title} {year} film" if year else f"{raw_title} film"
        search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&format=json"
        try:
            search_res = requests.get(search_url, headers=headers, timeout=5)
            if search_res.status_code == 200:
                search_data = search_res.json()
                results = search_data.get("query", {}).get("search", [])
                if results:
                    best_slug = results[0]["title"]
                    poster, plot = fetch_summary(best_slug)
                    if poster:
                        return poster, plot, "N/A", ""
        except: pass

        return None, None, "N/A", ""
    except Exception as e:
        return None, None, "N/A", ""

model, enc = load_model()
movies_df, genres_list = load_movies()
ratings_df = load_ratings()
user2idx   = enc["user2idx"]
item2idx   = enc["item2idx"]
idx2item   = enc["idx2item"]

# ── Recommend ─────────────────────────────────────────────────────────────────
def recommend(user_id, top_k=10, filter_genres=None):
    if user_id not in user2idx:
        return []
    u_idx    = user2idx[user_id]
    u_tensor = torch.tensor([u_idx], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        scores = (model.user_emb(u_tensor) @ model.item_emb.weight.T).squeeze()
    
    # Get more than top_k if filtering
    search_k = top_k * 5 if filter_genres else top_k
    top_indices = torch.topk(scores, min(search_k, len(scores))).indices.numpy()
    rec_items   = [idx2item[i] for i in top_indices]
    
    results = []
    for item_id in rec_items:
        row = movies_df[movies_df["item_id"] == item_id]
        if row.empty: continue
        
        movie_genres = row.iloc[0]["genres"].split("|")
        if filter_genres:
            if not any(g in movie_genres for g in filter_genres):
                continue
        
        title                      = row.iloc[0]["title"]
        genres                     = row.iloc[0]["genres"]
        poster, plot, rating, year = get_movie_info(title)
        results.append({"title": title, "genres": genres,
                        "poster": poster, "plot": plot,
                        "rating": rating, "year": year})
        
        if len(results) >= top_k:
            break
    return results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch Premium", page_icon="🎬", layout="wide")

# ── Glassmorphism CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #7c6af7;
        --secondary: #5a4fcf;
        --bg-dark: #0f0c29;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
        background-color: var(--bg-dark);
        color: #e0e0e0;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 60px 20px;
        border-radius: 24px;
        margin-bottom: 40px;
        text-align: center;
        border: 1px solid var(--glass-border);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(to right, #fff, #7c6af7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    /* Glass Cards */
    .movie-card {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        overflow: hidden;
        position: relative;
    }

    .movie-card:hover {
        transform: translateY(-10px);
        border-color: var(--primary);
        box-shadow: 0 10px 20px rgba(124, 106, 247, 0.2);
    }

    .movie-poster-container {
        position: relative;
        width: 100%;
        aspect-ratio: 2/3;
    }

    .genre-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(124, 106, 247, 0.2);
        color: #7c6af7;
        font-size: 0.7rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 5px;
        border: 1px solid rgba(124, 106, 247, 0.3);
    }

    /* Stats & Metrics */
    .stat-card {
        background: var(--glass-bg);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid var(--glass-border);
    }

    .stat-val { font-size: 2.2rem; font-weight: 700; color: var(--primary); }
    .stat-label { color: #888; text-transform: uppercase; letter-spacing: 1px; font-size: 0.7rem; }

    /* Custom Buttons */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #7c6af7 0%, #5a4fcf 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 106, 247, 0.3);
    }

    div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(124, 106, 247, 0.5);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 0;
        padding-top: 10px;
        font-weight: 600;
    }

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulseGlow {
        0% { text-shadow: 0 0 10px rgba(124, 106, 247, 0.3); }
        50% { text-shadow: 0 0 30px rgba(124, 106, 247, 0.7), 0 0 10px #fff; }
        100% { text-shadow: 0 0 10px rgba(124, 106, 247, 0.3); }
    }

    @keyframes rotateReel {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .hero-title {
        animation: pulseGlow 3s infinite ease-in-out;
    }

    .animate-card {
        animation: fadeInUp 0.8s cubic-bezier(0.23, 1, 0.32, 1) forwards;
        opacity: 0;
    }

    .loading-reel {
        width: 100px;
        height: 100px;
        animation: rotateReel 4s linear infinite;
        margin: 0 auto 20px auto;
        filter: drop-shadow(0 0 15px rgba(124, 106, 247, 0.6));
    }

    .scanner-bar {
        width: 200px;
        height: 4px;
        background: rgba(124, 106, 247, 0.1);
        border-radius: 10px;
        margin: 20px auto;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(124, 106, 247, 0.2);
    }

    .scanner-pulse {
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, #7c6af7, transparent);
        animation: scanning 1.5s infinite linear;
    }

    @keyframes scanning {
        0% { left: -100%; }
        100% { left: 100%; }
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🎬 CineMatch</div>
    <div style="font-size: 1.2rem; color: #a0a0b0; max-width: 700px; margin: 0 auto;">
        Discover your next favorite movie using state-of-the-art <b>Matrix Factorization</b>. 
        Personalized recommendations trained on 1M+ MovieLens ratings.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie-projector.png", width=80)
    st.title("Settings")
    user_id = st.number_input("👤 User ID", min_value=1, max_value=6040, value=1)
    top_k = st.slider("🎯 Recommendation Count", 5, 20, 10)
    
    st.divider()
    
    st.subheader("Filters")
    sel_genres = st.multiselect("🏷️ Filter by Genres", genres_list)
    
    st.divider()
    st.caption("Developed by Muhammad Rayan & Shaheer Beig")

# ── Main Content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🍿 Recommendations", "🕰️ User History", "🔍 Find Similar", "📊 Model Insights"])

with tab1:
    col_btn_1, col_btn_2 = st.columns([1, 4])
    with col_btn_1:
        clicked = st.button("Generate Picks", key="btn_rec")
    
    if clicked:
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
                <div style="text-align: center; padding: 60px 20px; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); backdrop-filter: blur(10px);">
                    <img src="https://img.icons8.com/fluency/96/film-reel.png" class="loading-reel">
                    <h2 style="color: #fff; font-weight: 700; margin-bottom: 5px;">Curating Your Private Cinema...</h2>
                    <p style="color: #888; font-size: 1.1rem;">Matrix Factorization in progress</p>
                    <div class="scanner-bar">
                        <div class="scanner-pulse"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        recs = recommend(user_id, top_k, sel_genres)
        loading_placeholder.empty()
        
        if not recs:
            st.error("No recommendations found matching your criteria.")
        else:
            st.markdown(f"### 🎬 Top {len(recs)} Recommendations for User {user_id}")
            cols = st.columns(5)
            for i, rec in enumerate(recs):
                with cols[i % 5]:
                    poster_url = rec["poster"] if rec["poster"] else "https://placehold.co/300x450?text=No+Poster"
                    genre_html = "".join([f'<span class="genre-badge">{g.strip()}</span>' for g in rec["genres"].split("|")[:2]])
                    
                    st.markdown(f"""
                    <div class="movie-card animate-card" style="animation-delay: {i*0.1}s">
                        <div class="movie-poster-container">
                            <img src="{poster_url}" style="width:100%; height:100%; object-fit:cover;">
                        </div>
                        <div style="padding: 12px;">
                            <div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; color: #fff; height: 40px; overflow: hidden;">
                                {rec['title']}
                            </div>
                            <div style="margin-bottom: 8px;">{genre_html}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if rec.get("plot"):
                        with st.expander("Plot Summary"):
                            st.write(rec["plot"])

with tab2:
    st.markdown(f"### 🎞️ What User {user_id} Liked")
    history = get_user_history(user_id, ratings_df, movies_df)
    
    if not history:
        st.info("No rating history found for this user.")
    else:
        h_cols = st.columns(5)
        for i, h in enumerate(history):
            with h_cols[i % 5]:
                # For history, we'll just show text for speed, but we could fetch posters too
                st.markdown(f"""
                <div class="stat-card" style="height: 180px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-weight:600; color:#fff;">{h['title']}</div>
                    <div style="color:var(--primary); font-size:1.5rem; font-weight:700; margin:10px 0;">{h['user_rating']} ⭐</div>
                    <div class="stat-label">Your Rating</div>
                </div>
                """, unsafe_allow_html=True)

with tab3:
    st.markdown("### 🔍 Find Similar Movies")
    target_movie = st.selectbox("Select a movie you love:", movies_df["title"].values)
    
    if st.button("Find Matches"):
        with st.spinner("Finding cinematic twins..."):
            # Get item ID
            m_row = movies_df[movies_df["title"] == target_movie]
            if not m_row.empty:
                m_id = m_row.iloc[0]["item_id"]
                if m_id in item2idx:
                    m_idx = item2idx[m_id]
                    m_emb = model.item_emb.weight[m_idx].detach().numpy()
                    
                    # Compute cosine similarity manually against all item embeddings
                    all_embs = model.item_emb.weight.detach().numpy()
                    norm_m = np.linalg.norm(m_emb)
                    norm_all = np.linalg.norm(all_embs, axis=1)
                    sims = (all_embs @ m_emb) / (norm_all * norm_m)
                    
                    # Get top 6 (excluding itself)
                    sim_indices = np.argsort(sims)[::-1][1:6]
                    sim_results = []
                    for s_idx in sim_indices:
                        s_item_id = idx2item[s_idx]
                        s_row = movies_df[movies_df["item_id"] == s_item_id]
                        if not s_row.empty:
                            s_title = s_row.iloc[0]["title"]
                            s_poster, _, _, _ = get_movie_info(s_title)
                            sim_results.append({"title": s_title, "poster": s_poster})
                    
                    s_cols = st.columns(5)
                    for i, s in enumerate(sim_results):
                        with s_cols[i]:
                            if s["poster"]: st.image(s["poster"])
                            else: st.image("https://placehold.co/300x450?text=No+Poster")
                            st.caption(s["title"])

with tab4:
    st.markdown("### 📊 Model Performance Evaluation")
    try:
        with open("data/results.json", "r") as f:
            metrics_data = json.load(f)
        
        m_df = pd.DataFrame(metrics_data)
        
        # Display as a pretty bar chart
        st.subheader("Hit Ratio @ 10 Comparison")
        st.bar_chart(m_df.set_index("model")["HitRatio@10"])
        
        st.subheader("Detailed Metrics")
        st.table(m_df)
        
        # Key Stat Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="stat-card"><div class="stat-val">61.7%</div><div class="stat-label">SVD HitRatio</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stat-card"><div class="stat-val">0.1646</div><div class="stat-label">SVD NDCG@10</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stat-card"><div class="stat-val">1.0M+</div><div class="stat-label">Training Samples</div></div>', unsafe_allow_html=True)
            
    except:
        st.error("Could not load evaluation metrics.")