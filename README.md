# 🎬 CineMatch: A Comparative Study of Recommendation Models

A personalized movie recommendation system built as a Final Course Project for Recommender Systems. Three collaborative filtering models of increasing complexity were implemented, trained, and evaluated on the MovieLens 1M dataset.

---

## Models Compared

| Model | Architecture | Recall@10 | Precision@10 | NDCG@10 | HitRatio@10 |
|-------|-------------|-----------|--------------|---------|-------------|
| **SVD** | Matrix Factorization | **0.1457** | **0.1158** | **0.1646** | **0.6168** |
| NCF | Neural Collaborative Filtering (NeuMF) | 0.1298 | 0.1068 | 0.1465 | 0.5810 |
| LightGCN | Light Graph Convolutional Network | 0.0825 | 0.0678 | 0.0928 | 0.4322 |

SVD achieved the best performance across all metrics.

---

## Dataset

- **MovieLens 1M** — 1,000,209 ratings from 6,040 users on 3,706 movies
- Ratings ≥ 4 treated as positive implicit feedback (575,281 interactions)
- 80/10/10 train/validation/test split

---

## Methodology

- All models trained with **Bayesian Personalized Ranking (BPR)** loss
- Negative sampling used for implicit feedback training
- Evaluation on standard ranking metrics at K=10
- Training done on Google Colab with T4 GPU

---

## Project Structure
cinematch/
├── notebook/
│   └── cinematch.ipynb       # Full training notebook (Colab)
├── models/
│   ├── svd_model_cpu.pkl     # Trained SVD model
│   └── encoders.pkl          # User/item index mappings
├── data/
│   ├── movies.csv            # Movie titles and genres
│   └── results.json          # Evaluation metrics
├── app.py                    # Streamlit web application
├── requirements.txt
└── README.md
---

## Web Application

The CineMatch app uses the best performing SVD model to generate real-time personalized movie recommendations. Movie posters and plot summaries are fetched via the Wikipedia API.

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## References

1. Koren et al. (2009). Matrix Factorization Techniques for Recommender Systems. IEEE Computer.
2. He et al. (2017). Neural Collaborative Filtering. WWW 2017.
3. He et al. (2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR 2020.
4. Rendle et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009.
5. Harper & Konstan (2015). The MovieLens Datasets. ACM TIIS.

---

## Team

| Name | Roll No |
|------|---------|
| Muhammad Rayan | 22K-4332 |
| Shaheer Beig | 22K-4321 |

**Supervisor:** Dr. Farrukh Shahid  
**Institution:** FAST NUCES Karachi  
**Course:** Recommender Systems (Final Project)