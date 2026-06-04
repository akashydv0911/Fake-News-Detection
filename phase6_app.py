# ============================================================
#  FAKE NEWS DETECTION — PHASE 6: Streamlit Web App
# ============================================================
#  Requirements:
#    pip install streamlit scikit-learn joblib pandas numpy
#    matplotlib seaborn wordcloud nltk
#
#  Run:
#    streamlit run phase6_app.py
#
#  Needs in same folder:
#    ├── models/best_model.pkl
#    ├── models/tfidf_vectorizer.pkl
#    └── cleaned_news.csv
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import os
from collections import Counter
from wordcloud import WordCloud

# ── sklearn (for re-training if pkl missing) ─────────────────
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #185FA5, #E24B4A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .subtitle { color: #666; font-size: 1rem; margin-top: 0; }
  .fake-box {
    background: #FCEBEB; border-left: 5px solid #E24B4A;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 1rem;
  }
  .real-box {
    background: #EAF3DE; border-left: 5px solid #639922;
    border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 1rem;
  }
  .metric-card {
    background: #F7F9FC; border-radius: 10px;
    padding: 1rem; text-align: center; border: 1px solid #E0E8F0;
  }
  .metric-num { font-size: 1.8rem; font-weight: 700; color: #185FA5; }
  .metric-lbl { font-size: 0.8rem; color: #888; margin-top: 2px; }
  .section-header {
    font-size: 1.2rem; font-weight: 600; color: #185FA5;
    border-bottom: 2px solid #185FA5; padding-bottom: 4px;
    margin: 1.5rem 0 1rem;
  }
  .badge-fake {
    background:#E24B4A; color:white; border-radius:20px;
    padding:3px 12px; font-size:.85rem; font-weight:600;
  }
  .badge-real {
    background:#639922; color:white; border-radius:20px;
    padding:3px 12px; font-size:.85rem; font-weight:600;
  }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner="Loading model...")
def load_model_and_vectorizer():
    """Load saved model & TF-IDF, or retrain if files missing."""
    pkl_model = "models/best_model.pkl"
    pkl_tfidf = "models/tfidf_vectorizer.pkl"

    if os.path.exists(pkl_model) and os.path.exists(pkl_tfidf):
        model = joblib.load(pkl_model)
        tfidf = joblib.load(pkl_tfidf)
        return model, tfidf, None   # no metrics on load

    # ── Retrain from cleaned_news.csv ────────────────────────
    if not os.path.exists("cleaned_news.csv"):
        return None, None, None

    df = pd.read_csv("cleaned_news.csv").dropna(subset=["text", "label"])
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(df["text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, pkl_model)
    joblib.dump(tfidf, pkl_tfidf)
    return model, tfidf, None


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    if os.path.exists("cleaned_news.csv"):
        return pd.read_csv("cleaned_news.csv").dropna(subset=["text","label"])
    return None


@st.cache_data(show_spinner="Training all models for comparison...")
def get_all_model_metrics():
    df = load_data()
    if df is None:
        return None, None, None

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(df["text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         MultinomialNB(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM":                 LinearSVC(random_state=42, max_iter=2000),
    }

    results, trained = {}, {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred)*100, 2),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0)*100, 2),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0)*100, 2),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0)*100, 2),
            "CM":        confusion_matrix(y_test, y_pred),
        }
        trained[name] = m
    return results, trained, tfidf


def predict_article(text, model, tfidf):
    cleaned = clean_text(text)
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]

    # Confidence via decision_function or predict_proba
    try:
        proba = model.predict_proba(vec)[0]
        conf  = max(proba) * 100
    except AttributeError:
        try:
            df_val = model.decision_function(vec)[0]
            conf   = min(99.9, 50 + abs(float(df_val)) * 8)
        except Exception:
            conf = 85.0
    return int(pred), round(conf, 1)


# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/news.png", width=64)
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home & Predict", "📊 EDA & Insights", "🤖 Model Comparison",
         "📈 Evaluation Metrics", "ℹ️ Project Summary"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Dataset**")
    df_sidebar = load_data()
    if df_sidebar is not None:
        st.metric("Total Articles", f"{len(df_sidebar):,}")
        fake_n = int((df_sidebar.label == 1).sum())
        real_n = int((df_sidebar.label == 0).sum())
        st.metric("Fake News",  f"{fake_n:,}")
        st.metric("Real News",  f"{real_n:,}")
    st.markdown("---")
    st.caption("Phase 6 · Fake News Detection · ML Project")


# ════════════════════════════════════════════════════════════
#  LOAD RESOURCES
# ════════════════════════════════════════════════════════════
model, tfidf_main, _ = load_model_and_vectorizer()
df_main = load_data()

# ════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & PREDICT
# ════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":
    st.markdown('<p class="main-title">🔍 Fake News Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">End-to-end NLP · TF-IDF · 5 ML Models · Real-time Prediction</p>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Quick stats ─────────────────────────────────────────
    if df_main is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-num">{len(df_main):,}</div><div class="metric-lbl">Total Articles</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-num">{int((df_main.label==1).sum()):,}</div><div class="metric-lbl">Fake Articles</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-num">{int((df_main.label==0).sum()):,}</div><div class="metric-lbl">Real Articles</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-num">5</div><div class="metric-lbl">ML Models Trained</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📝 Predict an Article</div>', unsafe_allow_html=True)

    if model is None or tfidf_main is None:
        st.error("⚠️  Model not found. Make sure `models/best_model.pkl` exists or `cleaned_news.csv` is present for auto-retraining.")
    else:
        # ── Example buttons ─────────────────────────────────
        st.markdown("**Try an example:**")
        ex1, ex2, ex3 = st.columns(3)
        examples = {
            "🟢 Real example": "Scientists at NASA have confirmed the discovery of water ice on the Moon's south pole, according to data from the LCROSS mission published in Science journal.",
            "🔴 Fake example": "BREAKING: Government secretly putting mind-control chemicals in tap water! Leaked documents expose the deep state conspiracy nobody wants you to see!",
            "✏️ Custom": ""
        }
        chosen_example = ""
        if ex1.button("🟢 Real example", use_container_width=True):
            chosen_example = examples["🟢 Real example"]
        if ex2.button("🔴 Fake example", use_container_width=True):
            chosen_example = examples["🔴 Fake example"]

        user_input = st.text_area(
            "Paste news article or headline:",
            value=chosen_example,
            height=160,
            placeholder="Type or paste a news article here...",
        )

        col_btn, col_clear = st.columns([1, 5])
        analyze = col_btn.button("🔍 Analyze", type="primary", use_container_width=True)

        if analyze and user_input.strip():
            with st.spinner("Analyzing..."):
                pred, conf = predict_article(user_input, model, tfidf_main)

            if pred == 1:
                st.markdown(f"""
                <div class="fake-box">
                  <h3>🔴 &nbsp;<span class="badge-fake">FAKE NEWS</span></h3>
                  <p style="margin-top:.6rem;font-size:1rem">
                    This article shows patterns consistent with <strong>fake/unreliable news</strong>.
                    <br>Confidence: <strong>{conf}%</strong>
                  </p>
                  <p style="color:#888;font-size:.85rem;margin-top:.5rem">
                    ⚠️ Always verify with trusted sources before sharing.
                  </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="real-box">
                  <h3>🟢 &nbsp;<span class="badge-real">REAL NEWS</span></h3>
                  <p style="margin-top:.6rem;font-size:1rem">
                    This article shows patterns consistent with <strong>real/reliable news</strong>.
                    <br>Confidence: <strong>{conf}%</strong>
                  </p>
                  <p style="color:#888;font-size:.85rem;margin-top:.5rem">
                    ✅ Cross-check sources for full verification.
                  </p>
                </div>""", unsafe_allow_html=True)

            # ── Confidence bar ───────────────────────────────
            st.markdown("**Confidence Level**")
            bar_color = "#E24B4A" if pred == 1 else "#639922"
            st.markdown(f"""
            <div style="background:#eee;border-radius:8px;height:18px;overflow:hidden;">
              <div style="width:{conf}%;background:{bar_color};height:100%;border-radius:8px;
                          display:flex;align-items:center;justify-content:center;
                          color:white;font-size:.75rem;font-weight:600">{conf}%</div>
            </div>""", unsafe_allow_html=True)

            # ── Cleaned text preview ─────────────────────────
            with st.expander("🔎 View cleaned text fed to model"):
                st.code(clean_text(user_input), language=None)

        elif analyze:
            st.warning("Please enter some text to analyze.")


# ════════════════════════════════════════════════════════════
#  PAGE 2 — EDA & INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if df_main is None:
        st.error("cleaned_news.csv not found.")
        st.stop()

    fake_df = df_main[df_main.label == 1]
    real_df = df_main[df_main.label == 0]

    # ── Distribution pie ─────────────────────────────────────
    st.markdown('<div class="section-header">1. Label Distribution</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        sizes  = [len(fake_df), len(real_df)]
        labels = [f"Fake\n{len(fake_df):,}", f"Real\n{len(real_df):,}"]
        colors = ["#E24B4A", "#3B6D11"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.1f%%",
            colors=colors, startangle=90,
            wedgeprops={"edgecolor":"white","linewidth":2}
        )
        for at in autotexts:
            at.set_fontsize(12); at.set_color("white"); at.set_fontweight("bold")
        ax.set_title("Fake vs Real Distribution", fontweight="bold")
        st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Fake", "Real"], [len(fake_df), len(real_df)],
               color=["#E24B4A", "#3B6D11"], edgecolor="white", linewidth=0.8)
        for i, v in enumerate([len(fake_df), len(real_df)]):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontweight="bold")
        ax.set_title("Article Count by Label", fontweight="bold")
        ax.set_ylabel("Count")
        st.pyplot(fig); plt.close()

    # ── Word count distribution ──────────────────────────────
    st.markdown('<div class="section-header">2. Article Length Analysis</div>', unsafe_allow_html=True)

    if "word_count" not in df_main.columns:
        df_main["word_count"] = df_main["text"].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(fake_df["word_count"].clip(upper=1500), bins=50,
                 color="#E24B4A", alpha=0.75, edgecolor="white")
    axes[0].set_title("Fake News — Word Count", fontweight="bold")
    axes[0].set_xlabel("Word Count"); axes[0].set_ylabel("Frequency")

    axes[1].hist(real_df["word_count"].clip(upper=1500), bins=50,
                 color="#3B6D11", alpha=0.75, edgecolor="white")
    axes[1].set_title("Real News — Word Count", fontweight="bold")
    axes[1].set_xlabel("Word Count"); axes[1].set_ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg word count (Fake)", f"{fake_df['word_count'].mean():.0f}")
    c2.metric("Avg word count (Real)", f"{real_df['word_count'].mean():.0f}")
    c3.metric("Max word count", f"{df_main['word_count'].max():,}")

    # ── Top words ────────────────────────────────────────────
    st.markdown('<div class="section-header">3. Most Common Words</div>', unsafe_allow_html=True)

    stopwords_simple = set(["the","a","an","and","or","but","in","on","at","to",
                             "for","of","with","is","was","are","be","been","by",
                             "that","this","it","as","from","have","had","has","his",
                             "her","their","he","she","they","we","you","i","not",
                             "will","said","also","its","which","were","would","after",
                             "before","there","can","all","about","when","who","if"])

    def top_words(texts, n=15):
        words = " ".join(texts).split()
        words = [w for w in words if w not in stopwords_simple and len(w) > 2]
        return Counter(words).most_common(n)

    c1, c2 = st.columns(2)
    with c1:
        tw_fake = top_words(fake_df["text"].fillna("").tolist())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([w for w,_ in tw_fake[::-1]], [c for _,c in tw_fake[::-1]],
                color="#E24B4A", alpha=0.85)
        ax.set_title("Top 15 Words — Fake News", fontweight="bold", color="#E24B4A")
        ax.set_xlabel("Frequency")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        tw_real = top_words(real_df["text"].fillna("").tolist())
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([w for w,_ in tw_real[::-1]], [c for _,c in tw_real[::-1]],
                color="#3B6D11", alpha=0.85)
        ax.set_title("Top 15 Words — Real News", fontweight="bold", color="#3B6D11")
        ax.set_xlabel("Frequency")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Word Clouds ──────────────────────────────────────────
    st.markdown('<div class="section-header">4. Word Clouds</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        wc = WordCloud(width=700, height=350, background_color="white",
                       colormap="Reds", max_words=100,
                       stopwords=stopwords_simple).generate(
            " ".join(fake_df["text"].fillna("").tolist())
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title("Fake News Word Cloud", fontweight="bold", color="#E24B4A")
        st.pyplot(fig); plt.close()

    with c2:
        wc = WordCloud(width=700, height=350, background_color="white",
                       colormap="Greens", max_words=100,
                       stopwords=stopwords_simple).generate(
            " ".join(real_df["text"].fillna("").tolist())
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title("Real News Word Cloud", fontweight="bold", color="#3B6D11")
        st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<p class="main-title">🤖 Model Comparison</p>', unsafe_allow_html=True)
    st.info("Training all 5 models — this may take 30–60 seconds on first load.", icon="⏳")

    results, trained, tfidf_all = get_all_model_metrics()

    if results is None:
        st.error("cleaned_news.csv not found.")
        st.stop()

    MODEL_NAMES  = list(results.keys())
    SHORT_NAMES  = ["LR", "NB", "DT", "RF", "SVM"]

    # ── Summary table ────────────────────────────────────────
    st.markdown('<div class="section-header">Metrics Summary</div>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in vals.items() if k not in ("CM",)}
        for name, vals in results.items()
    }).T
    best_name = metrics_df["F1 Score"].idxmax()
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#D4EDDA"), use_container_width=True)
    st.success(f"🏆 Best Model: **{best_name}** with F1 = **{metrics_df.loc[best_name,'F1 Score']}%**")

    # ── Bar chart all metrics ─────────────────────────────────
    st.markdown('<div class="section-header">Metrics Bar Chart</div>', unsafe_allow_html=True)
    metric_choice = st.selectbox("Select metric to visualize:", ["Accuracy","Precision","Recall","F1 Score"])
    fig, ax = plt.subplots(figsize=(10, 4))
    vals   = [results[m][metric_choice] for m in MODEL_NAMES]
    colors = ["#185FA5" if m != best_name else "#E24B4A" for m in MODEL_NAMES]
    bars   = ax.bar(SHORT_NAMES, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.15,
                f"{val:.2f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(min(vals)-5, 102)
    ax.set_ylabel(f"{metric_choice} (%)")
    ax.set_title(f"{metric_choice} Comparison — All Models", fontweight="bold")
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1,color="#185FA5",label="Models"),
        plt.Rectangle((0,0),1,1,color="#E24B4A",label=f"Best ({best_name})")
    ])
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Feature importance ───────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance (Logistic Regression)</div>', unsafe_allow_html=True)
    lr_model     = trained["Logistic Regression"]
    feature_names = np.array(tfidf_all.get_feature_names_out())
    coefs         = lr_model.coef_[0]

    top_fake_idx = np.argsort(coefs)[-20:][::-1]
    top_real_idx = np.argsort(coefs)[:20]

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(feature_names[top_fake_idx], coefs[top_fake_idx],
                color="#E24B4A", alpha=0.85)
        ax.set_title("Top Words → FAKE", fontweight="bold", color="#E24B4A")
        ax.set_xlabel("Coefficient"); ax.invert_yaxis()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.barh(feature_names[top_real_idx], np.abs(coefs[top_real_idx]),
                color="#3B6D11", alpha=0.85)
        ax.set_title("Top Words → REAL", fontweight="bold", color="#3B6D11")
        ax.set_xlabel("Coefficient (abs)"); ax.invert_yaxis()
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATION METRICS
# ════════════════════════════════════════════════════════════
elif page == "📈 Evaluation Metrics":
    st.markdown('<p class="main-title">📈 Evaluation Metrics</p>', unsafe_allow_html=True)
    st.info("Training models for evaluation visuals...", icon="⏳")

    results, trained, tfidf_all = get_all_model_metrics()
    if results is None:
        st.error("cleaned_news.csv not found.")
        st.stop()

    MODEL_NAMES = list(results.keys())
    SHORT_NAMES = ["LR", "NB", "DT", "RF", "SVM"]

    # ── Confusion matrices ───────────────────────────────────
    st.markdown('<div class="section-header">Confusion Matrices — All Models</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    cmaps     = ["Blues","Reds","Greens","Oranges","Purples"]
    for ax, name, short, cmap in zip(axes, MODEL_NAMES, SHORT_NAMES, cmaps):
        cm = results[name]["CM"]
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"],
                    linewidths=0.5, cbar=False, annot_kws={"size":12})
        ax.set_title(short, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Accuracy vs F1 grouped bar ───────────────────────────
    st.markdown('<div class="section-header">Accuracy vs F1 Score</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    acc_v = [results[m]["Accuracy"]  for m in MODEL_NAMES]
    f1_v  = [results[m]["F1 Score"]  for m in MODEL_NAMES]
    x     = np.arange(len(MODEL_NAMES)); w = 0.35
    b1 = ax.bar(x-w/2, acc_v, w, label="Accuracy", color="#185FA5", alpha=0.85)
    b2 = ax.bar(x+w/2, f1_v,  w, label="F1 Score",  color="#E24B4A", alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, rotation=15, ha="right")
    ax.set_ylim(min(acc_v+f1_v)-5, 103); ax.set_ylabel("Score (%)")
    ax.set_title("Accuracy vs F1 Score — All Models", fontweight="bold"); ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Classification report (best model) ──────────────────
    metrics_df  = pd.DataFrame({n:{k:v for k,v in r.items() if k!="CM"} for n,r in results.items()}).T
    best_name   = metrics_df["F1 Score"].idxmax()
    st.markdown(f'<div class="section-header">Classification Report — {best_name}</div>', unsafe_allow_html=True)

    if df_main is not None:
        tfidf_r   = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
        X_all     = tfidf_r.fit_transform(df_main["text"])
        y_all     = df_main["label"]
        _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
        best_m    = trained[best_name]

        try:
            tfidf_b   = tfidf_all
            X_test_b  = tfidf_b.transform(df_main["text"].iloc[int(len(df_main)*0.8):])
            y_test_b  = y_all.iloc[int(len(df_main)*0.8):]
            y_pred_b  = best_m.predict(X_test_b)
        except Exception:
            y_pred_b  = best_m.predict(X_test)
            y_test_b  = y_test

        report = classification_report(y_test_b, y_pred_b,
                                       target_names=["REAL","FAKE"], output_dict=True)
        report_df = pd.DataFrame(report).T.round(3)
        st.dataframe(report_df, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 5 — PROJECT SUMMARY
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ Project Summary":
    st.markdown('<p class="main-title">ℹ️ Project Summary</p>', unsafe_allow_html=True)
    st.markdown("---")

    sections = {
        "🎯 Problem Statement": """
Detect whether a given news article is **Fake** or **Real** using Natural Language Processing
and supervised Machine Learning. Binary classification task: **1 = Fake**, **0 = Real**.
        """,
        "📦 Dataset Overview": """
- **Source**: Kaggle — *Fake and Real News Dataset*
- **Files**: `Fake.csv` + `True.csv`
- **Size**: ~44,000 articles
- **Features used**: `title` + `text` (combined into `content`)
        """,
        "🧹 Data Preprocessing": """
1. Merged fake and real datasets, added labels
2. Removed null values and duplicate rows
3. Combined `title` + `text` into a single `content` column
4. Lowercased all text, removed URLs, HTML tags, special characters
5. Saved as `cleaned_news.csv`
        """,
        "🔬 Text Preprocessing & Feature Extraction": """
- **Tokenization**: split text into tokens
- **Stopword removal**: removed common English stopwords
- **Punctuation removal**: regex-based cleaning
- **TF-IDF Vectorizer**: `max_features=5000`, `ngram_range=(1,2)`, English stopwords
        """,
        "📊 EDA Findings": """
- Dataset is roughly **balanced** (~50/50 fake vs real)
- Fake news articles tend to use more **emotional, sensational** language
- Real news articles use more **formal, factual** terminology
- Average article length: ~800 words
        """,
        "🤖 Models Trained": """
| Model | Typical Accuracy |
|---|---|
| Logistic Regression | ~98% |
| Naive Bayes | ~94% |
| Decision Tree | ~99% |
| Random Forest | ~99% |
| SVM (LinearSVC) | ~99% |
        """,
        "🏆 Conclusion": """
- TF-IDF + traditional ML achieves **>98% accuracy** on this dataset
- **Decision Tree / Random Forest / SVM** are top performers
- Feature importance shows clear linguistic differences between fake and real news
- Model saved as `models/best_model.pkl` for inference
        """,
        "🚀 Future Enhancements": """
- Use **BERT / RoBERTa** transformer models for deeper semantic understanding
- Add **source credibility** as a feature (domain, author)
- Deploy as a **REST API** with FastAPI
- Add **multilingual** fake news detection
- Integrate **real-time news feed** for live detection
        """,
    }

    for title, content in sections.items():
        with st.expander(title, expanded=True):
            st.markdown(content)
