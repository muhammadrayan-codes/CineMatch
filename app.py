import streamlit as st
import pickle
import pandas as pd
import torch
import torch.nn as nn
import requests
import os

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

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
    return pd.read_csv("data/movies.csv")

@st.cache_data(show_spinner=False)
def get_movie_info(title):
    try:
        clean   = title.split(" (")[0]
        search  = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + clean.replace(" ", "_"),
            timeout=5
        )
        data    = search.json()
        poster  = data.get("thumbnail", {}).get("source", None)
        plot    = data.get("extract", "")[:300]
        return poster, plot, "N/A", ""
    except:
        return None, None, None, None

model, enc = load_model()
movies_df  = load_movies()
user2idx   = enc["user2idx"]
item2idx   = enc["item2idx"]
idx2item   = enc["idx2item"]

# ── Recommend ─────────────────────────────────────────────────────────────────
def recommend(user_id, top_k=10):
    if user_id not in user2idx:
        return []
    u_idx    = user2idx[user_id]
    u_tensor = torch.tensor([u_idx], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        scores = (model.user_emb(u_tensor) @ model.item_emb.weight.T).squeeze()
    top_indices = torch.topk(scores, top_k).indices.numpy()
    rec_items   = [idx2item[i] for i in top_indices]
    results = []
    for item_id in rec_items:
        row = movies_df[movies_df["item_id"] == item_id]
        if row.empty:
            continue
        title                      = row.iloc[0]["title"]
        genres                     = row.iloc[0]["genres"]
        poster, plot, rating, year = get_movie_info(title)
        results.append({"title": title, "genres": genres,
                        "poster": poster, "plot": plot,
                        "rating": rating, "year": year})
    return results

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
    }
    .hero h1 { font-size: 3rem; font-weight: 700; color: #ffffff; margin: 0 0 8px 0; }
    .hero p  { font-size: 1.1rem; color: #a0a0b0; margin: 0; }

    .stats-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .stats-number { font-size: 2rem; font-weight: 700; color: #7c6af7; }
    .stats-label  { font-size: 0.8rem; color: #888; margin-top: 4px; }

    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #7c6af7, #5a4fcf);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>Personalized movie recommendations powered by Matrix Factorization · MovieLens 1M</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stats-box"><div class="stats-number">6,040</div><div class="stats-label">Users</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stats-box"><div class="stats-number">3,706</div><div class="stats-label">Movies</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stats-box"><div class="stats-number">1M+</div><div class="stats-label">Ratings</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stats-box"><div class="stats-number">61.7%</div><div class="stats-label">Hit Ratio@10</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Controls ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    user_id = st.number_input("👤 Enter User ID", min_value=1, max_value=6040, value=1)
with col2:
    top_k = st.slider("🎯 Number of recommendations", 5, 20, 10)
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    clicked = st.button("Get Recommendations")

# ── Results ───────────────────────────────────────────────────────────────────
if clicked:
    with st.spinner("Finding your perfect movies..."):
        recs = recommend(user_id, top_k)

    if not recs:
        st.error("User not found in dataset.")
    else:
        st.markdown(f"### 🍿 Top {top_k} picks for User {user_id}")
        cols = st.columns(5)
        for i, rec in enumerate(recs):
            with cols[i % 5]:
                if rec["poster"]:
                    st.image(rec["poster"], use_container_width=True)
                else:
                    st.image("https://placehold.co/300x450?text=🎬", use_container_width=True)

                st.markdown(f"**{rec['title']}**")

                if rec.get("year"):
                    st.caption(rec["year"])

                st.caption(" · ".join(g.strip() for g in rec["genres"].split("|")))

                if rec.get("rating") and rec["rating"] != "N/A":
                    st.caption(f"⭐ IMDb: {rec['rating']}")

                if rec.get("plot"):
                    with st.expander("Plot"):
                        st.write(rec["plot"])