# ============================================================
#  FAKE NEWS DETECTION — PHASE 6: Streamlit Web App
#  v2 — Works WITHOUT pre-uploaded CSV (has file uploader)
# ============================================================
#  Run: streamlit run phase6_app.py
# ============================================================

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import io
from collections import Counter
from wordcloud import WordCloud

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

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size:2.2rem; font-weight:700;
    background:linear-gradient(90deg,#185FA5,#E24B4A);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.fake-box {
    background:rgba(226,75,74,0.15);
    border-left:5px solid #E24B4A;
    border-radius:8px;
    padding:1.2rem 1.5rem;
    margin-top:1rem;
    color:white;
}
.real-box {
    background:rgba(76,175,80,0.15);
    border-left:5px solid #4CAF50;
    border-radius:8px;
    padding:1.2rem 1.5rem;
    margin-top:1rem;
    color:white;
}
.metric-card { background:#1e1e2e; border-radius:10px;
    padding:1rem; text-align:center; border:1px solid #333; }
.metric-num { font-size:1.8rem; font-weight:700; color:#378ADD; }
.metric-lbl { font-size:0.8rem; color:#888; margin-top:2px; }
.section-header { font-size:1.1rem; font-weight:600; color:#378ADD;
    border-bottom:2px solid #378ADD; padding-bottom:4px; margin:1.2rem 0 .8rem; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── SESSION STATE: store trained model ───────────────────────
if "model" not in st.session_state:
    st.session_state.model = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "df" not in st.session_state:
    st.session_state.df = None
if "results" not in st.session_state:
    st.session_state.results = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

def load_df_from_path(path):
    df = pd.read_csv(path).dropna(subset=["text","label"])
    if "word_count" not in df.columns:
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    return df

def try_load_csv():
    """Try loading cleaned_news.csv from disk (works locally)."""
    if os.path.exists("cleaned_news.csv"):
        return load_df_from_path("cleaned_news.csv")
    return None

def train_all_models(df):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    X = tfidf.fit_transform(df["text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

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

    best_name = max(results, key=lambda x: results[x]["F1 Score"])
    return tfidf, results, trained, trained[best_name], best_name

def predict_article(text):
    cleaned = clean_text(text)
    vec = st.session_state.tfidf.transform([cleaned])
    pred = st.session_state.model.predict(vec)[0]
    try:
        proba = st.session_state.model.predict_proba(vec)[0]
        conf = round(max(proba)*100, 1)
    except:
        try:
            df_val = st.session_state.model.decision_function(vec)[0]
            conf = round(min(99.9, 50+abs(float(df_val))*8), 1)
        except:
            conf = 85.0
    return int(pred), conf

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Home & Predict",
        "📊 EDA & Insights",
        "🤖 Model Comparison",
        "📈 Evaluation Metrics",
        "ℹ️ Project Summary"
    ], label_visibility="collapsed")
    st.markdown("---")

    # ── DATA LOADER ─────────────────────────────────────────
    st.markdown("**📁 Load Dataset**")

    # Try auto-load from disk first
    if st.session_state.df is None:
        disk_df = try_load_csv()
        if disk_df is not None:
            st.session_state.df = disk_df

    # File uploader as fallback
    if st.session_state.df is None:
        uploaded = st.file_uploader(
            "Upload cleaned_news.csv",
            type=["csv"],
            help="Upload your cleaned_news.csv file"
        )
        if uploaded:
            df_up = pd.read_csv(uploaded).dropna(subset=["text","label"])
            if "word_count" not in df_up.columns:
                df_up["word_count"] = df_up["text"].apply(lambda x: len(str(x).split()))
            st.session_state.df = df_up
            st.success("✅ Dataset loaded!")
    else:
        df = st.session_state.df
        st.metric("Total Articles", f"{len(df):,}")
        st.metric("Fake", f"{int((df.label==1).sum()):,}")
        st.metric("Real", f"{int((df.label==0).sum()):,}")

    # Train button
    if st.session_state.df is not None and st.session_state.model is None:
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            with st.spinner("Training 5 models... (1-2 min)"):
                tfidf, results, trained, best_model, best_name = train_all_models(st.session_state.df)
                st.session_state.tfidf        = tfidf
                st.session_state.results      = results
                st.session_state.trained_models = trained
                st.session_state.model        = best_model
                st.session_state.best_name    = best_name
            st.success(f"✅ Done! Best: {best_name}")
            st.rerun()

    if st.session_state.model is not None:
        st.success(f"✅ Model ready!")
        st.caption(f"Best: {st.session_state.get('best_name','')}")

    st.markdown("---")
    st.caption("Phase 6 · Fake News Detection · ML Project")

# ════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & PREDICT
# ════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":
    st.markdown('<p class="main-title">🔍 Fake News Detection System</p>', unsafe_allow_html=True)
    st.caption("End-to-end NLP · TF-IDF · 5 ML Models · Real-time Prediction")
    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-num">{len(df):,}</div><div class="metric-lbl">Total Articles</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-num">{int((df.label==1).sum()):,}</div><div class="metric-lbl">Fake Articles</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-num">{int((df.label==0).sum()):,}</div><div class="metric-lbl">Real Articles</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-num">5</div><div class="metric-lbl">ML Models</div></div>', unsafe_allow_html=True)
    else:
        st.info("👈 Upload your **cleaned_news.csv** in the sidebar to get started.")
        st.info(
    f"🤖 Prediction generated using: {st.session_state.best_model_name}")
        

    st.markdown('<div class="section-header">📝 Predict an Article</div>', unsafe_allow_html=True)

    if st.session_state.model is None:
        st.warning(
        "⚠️ Please click 'Train Models' in the sidebar first."
    )
    else:
        c1,c2 = st.columns(2)
        real_ex = "Scientists at NASA confirmed discovery of water ice on Moon's south pole, according to data published in Science journal."
        fake_ex = "SHOCKING: Government secretly controlling weather with 5G towers! Leaked documents expose the deep state conspiracy!"

        if c1.button("🟢 Try Real Example", use_container_width=True):
            st.session_state["example_text"] = real_ex
        if c2.button("🔴 Try Fake Example", use_container_width=True):
            st.session_state["example_text"] = fake_ex

        user_input = st.text_area(
            "Paste news article or headline:",
            value=st.session_state.get("example_text", ""),
            height=150,
            placeholder="Type or paste a news article here..."
        )

        if st.button("🔍 Analyze Article", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing..."):
                    pred, conf = predict_article(user_input)
                if pred == 1:
                    st.markdown(f'<div class="fake-box"><h3>🔴 FAKE NEWS — Confidence: {conf}%</h3><p>Suspicious language patterns detected. Verify with trusted sources.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="real-box"><h3>🟢 REAL NEWS — Confidence: {conf}%</h3><p>Language patterns consistent with factual reporting.</p></div>', unsafe_allow_html=True)

                bar_color = "#E24B4A" if pred==1 else "#639922"
                st.markdown(f"""<div style="background:#333;border-radius:8px;height:20px;overflow:hidden;margin-top:10px">
                <div style="width:{conf}%;background:{bar_color};height:100%;border-radius:8px;
                display:flex;align-items:center;justify-content:center;color:white;font-size:.8rem;font-weight:600">{conf}%</div></div>""", unsafe_allow_html=True)
            else:
                st.warning("Please enter some text.")

# ════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown('<p class="main-title">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    if st.session_state.df is None:
        st.info("👈 Upload CSV in sidebar first."); st.stop()

    df = st.session_state.df
    fake_df = df[df.label==1]
    real_df = df[df.label==0]

    st.markdown('<div class="section-header">1. Label Distribution</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig,ax = plt.subplots(figsize=(5,4))
        ax.pie([len(fake_df),len(real_df)],
               labels=[f"Fake\n{len(fake_df):,}",f"Real\n{real_df.__len__():,}"],
               autopct="%1.1f%%", colors=["#E24B4A","#3B6D11"],
               wedgeprops={"edgecolor":"white","linewidth":2}, startangle=90)
        ax.set_title("Fake vs Real", fontweight="bold")
        st.pyplot(fig); plt.close()
    with c2:
        fig,ax = plt.subplots(figsize=(5,4))
        ax.bar(["Fake","Real"],[len(fake_df),len(real_df)],
               color=["#E24B4A","#3B6D11"],edgecolor="white")
        for i,v in enumerate([len(fake_df),len(real_df)]):
            ax.text(i,v+100,f"{v:,}",ha="center",fontweight="bold",color="white")
        ax.set_facecolor("#1e1e2e"); fig.patch.set_facecolor("#1e1e2e")
        ax.tick_params(colors="white"); ax.set_title("Article Count",fontweight="bold",color="white")
        st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">2. Article Length</div>', unsafe_allow_html=True)
    if "word_count" not in df.columns:
        df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    fig,axes = plt.subplots(1,2,figsize=(12,4))
    fig.patch.set_facecolor("#1e1e2e")
    for ax,sub_df,color,label in zip(axes,[fake_df,real_df],["#E24B4A","#3B6D11"],["Fake","Real"]):
        ax.hist(sub_df["word_count"].clip(upper=1500),bins=50,color=color,alpha=0.8,edgecolor="white")
        ax.set_title(f"{label} News — Word Count",fontweight="bold",color="white")
        ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2,c3 = st.columns(3)
    c1.metric("Avg (Fake)",f"{fake_df['word_count'].mean():.0f} words")
    c2.metric("Avg (Real)",f"{real_df['word_count'].mean():.0f} words")
    c3.metric("Max",f"{df['word_count'].max():,} words")

    st.markdown('<div class="section-header">3. Top Words & Word Clouds</div>', unsafe_allow_html=True)
    stopwords_simple = {"the","a","an","and","or","but","in","on","at","to","for","of","with",
                        "is","was","are","be","been","by","that","this","it","as","from","have",
                        "had","has","his","her","their","he","she","they","we","you","i","not",
                        "will","said","also","its","which","were","would","after","before","there",
                        "can","all","about","when","who","if","one","two","new","more","also"}

    def top_words(texts,n=15):
        words=" ".join(texts).split()
        words=[w for w in words if w not in stopwords_simple and len(w)>2]
        return Counter(words).most_common(n)

    c1,c2 = st.columns(2)
    with c1:
        tw = top_words(fake_df["text"].fillna("").tolist())
        fig,ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor("#1e1e2e")
        ax.barh([w for w,_ in tw[::-1]],[c for _,c in tw[::-1]],color="#E24B4A",alpha=0.85)
        ax.set_title("Top Words — Fake",fontweight="bold",color="#E24B4A")
        ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        tw = top_words(real_df["text"].fillna("").tolist())
        fig,ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor("#1e1e2e")
        ax.barh([w for w,_ in tw[::-1]],[c for _,c in tw[::-1]],color="#3B6D11",alpha=0.85)
        ax.set_title("Top Words — Real",fontweight="bold",color="#3B6D11")
        ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c1,c2 = st.columns(2)
    with c1:
        wc = WordCloud(width=700,height=350,background_color="#1e1e2e",
                       colormap="Reds",max_words=100,stopwords=stopwords_simple
                       ).generate(" ".join(fake_df["text"].fillna("").tolist()))
        fig,ax = plt.subplots(figsize=(7,3.5)); fig.patch.set_facecolor("#1e1e2e")
        ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
        ax.set_title("Fake News Word Cloud",fontweight="bold",color="#E24B4A")
        st.pyplot(fig); plt.close()
    with c2:
        wc = WordCloud(width=700,height=350,background_color="#1e1e2e",
                       colormap="Greens",max_words=100,stopwords=stopwords_simple
                       ).generate(" ".join(real_df["text"].fillna("").tolist()))
        fig,ax = plt.subplots(figsize=(7,3.5)); fig.patch.set_facecolor("#1e1e2e")
        ax.imshow(wc,interpolation="bilinear"); ax.axis("off")
        ax.set_title("Real News Word Cloud",fontweight="bold",color="#3B6D11")
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown('<p class="main-title">🤖 Model Comparison</p>', unsafe_allow_html=True)
    if st.session_state.results is None:
        st.warning("👈 Upload CSV and click **Train Models** in sidebar first."); st.stop()

    results = st.session_state.results
    MODEL_NAMES = list(results.keys())
    SHORT_NAMES = ["LR","NB","DT","RF","SVM"]
    best_name   = st.session_state.get("best_name","")

    metrics_df = pd.DataFrame({n:{k:v for k,v in r.items() if k!="CM"} for n,r in results.items()}).T
    st.markdown('<div class="section-header">Metrics Summary</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df.style.highlight_max(axis=0,color="#1a4a1a"), use_container_width=True)
    st.success(f"🏆 Best Model: **{best_name}** — F1: **{metrics_df.loc[best_name,'F1 Score']}%**")

    st.markdown('<div class="section-header">Metrics Bar Chart</div>', unsafe_allow_html=True)
    metric_choice = st.selectbox("Metric:", ["Accuracy","Precision","Recall","F1 Score"])
    vals   = [results[m][metric_choice] for m in MODEL_NAMES]
    colors = ["#E24B4A" if m==best_name else "#185FA5" for m in MODEL_NAMES]
    fig,ax = plt.subplots(figsize=(10,4)); fig.patch.set_facecolor("#1e1e2e")
    bars = ax.bar(SHORT_NAMES,vals,color=colors,edgecolor="white",linewidth=0.8)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                f"{val:.1f}%",ha="center",fontsize=10,fontweight="bold",color="white")
    ax.set_ylim(min(vals)-5,103); ax.set_ylabel(f"{metric_choice} (%)",color="white")
    ax.set_title(f"{metric_choice} — All Models",fontweight="bold",color="white")
    ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if st.session_state.trained_models and st.session_state.tfidf:
        st.markdown('<div class="section-header">Feature Importance (Logistic Regression)</div>', unsafe_allow_html=True)
        lr = st.session_state.trained_models.get("Logistic Regression")
        if lr:
            feat_names = np.array(st.session_state.tfidf.get_feature_names_out())
            coefs = lr.coef_[0]
            top_f = np.argsort(coefs)[-20:][::-1]
            top_r = np.argsort(coefs)[:20]
            c1,c2 = st.columns(2)
            with c1:
                fig,ax = plt.subplots(figsize=(6,6)); fig.patch.set_facecolor("#1e1e2e")
                ax.barh(feat_names[top_f],coefs[top_f],color="#E24B4A",alpha=0.85)
                ax.set_title("Top Words → FAKE",fontweight="bold",color="#E24B4A")
                ax.invert_yaxis(); ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with c2:
                fig,ax = plt.subplots(figsize=(6,6)); fig.patch.set_facecolor("#1e1e2e")
                ax.barh(feat_names[top_r],np.abs(coefs[top_r]),color="#3B6D11",alpha=0.85)
                ax.set_title("Top Words → REAL",fontweight="bold",color="#3B6D11")
                ax.invert_yaxis(); ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
                plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATION METRICS
# ════════════════════════════════════════════════════════════
elif page == "📈 Evaluation Metrics":
    st.markdown('<p class="main-title">📈 Evaluation Metrics</p>', unsafe_allow_html=True)
    if st.session_state.results is None:
        st.warning("👈 Upload CSV and click **Train Models** in sidebar first."); st.stop()

    results     = st.session_state.results
    MODEL_NAMES = list(results.keys())
    SHORT_NAMES = ["LR","NB","DT","RF","SVM"]
    best_name   = st.session_state.get("best_name","")

    st.markdown('<div class="section-header">Confusion Matrices — All Models</div>', unsafe_allow_html=True)
    fig,axes = plt.subplots(1,5,figsize=(22,4)); fig.patch.set_facecolor("#1e1e2e")
    for ax,name,short,cmap in zip(axes,MODEL_NAMES,SHORT_NAMES,["Blues","Reds","Greens","Oranges","Purples"]):
        sns.heatmap(results[name]["CM"],annot=True,fmt="d",cmap=cmap,ax=ax,
                    xticklabels=["REAL","FAKE"],yticklabels=["REAL","FAKE"],
                    linewidths=0.5,cbar=False,annot_kws={"size":12})
        ax.set_title(short,fontweight="bold",color="white")
        ax.set_facecolor("#1e1e2e"); ax.tick_params(colors="white")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div class="section-header">Accuracy vs F1 Score</div>', unsafe_allow_html=True)
    acc_v = [results[m]["Accuracy"]  for m in MODEL_NAMES]
    f1_v  = [results[m]["F1 Score"]  for m in MODEL_NAMES]
    x = np.arange(len(MODEL_NAMES)); w=0.35
    fig,ax = plt.subplots(figsize=(11,5)); fig.patch.set_facecolor("#1e1e2e")
    b1=ax.bar(x-w/2,acc_v,w,label="Accuracy",color="#185FA5",alpha=0.85)
    b2=ax.bar(x+w/2,f1_v, w,label="F1 Score", color="#E24B4A",alpha=0.85)
    for bars in [b1,b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
                    f"{bar.get_height():.1f}%",ha="center",fontsize=8.5,color="white")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES,rotation=15,ha="right",color="white")
    ax.set_ylim(min(acc_v+f1_v)-5,103); ax.set_facecolor("#1e1e2e")
    ax.tick_params(colors="white"); ax.legend()
    ax.set_title("Accuracy vs F1 — All Models",fontweight="bold",color="white")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown(f'<div class="section-header">Classification Report — {best_name}</div>', unsafe_allow_html=True)
    best_m = st.session_state.trained_models[best_name]
    tfidf  = st.session_state.tfidf
    df     = st.session_state.df
    X_all  = tfidf.transform(df["text"])
    y_all  = df["label"]
    split  = int(len(df)*0.8)
    y_pred = best_m.predict(X_all[split:])
    y_test = y_all.iloc[split:].values
    report = classification_report(y_test,y_pred,target_names=["REAL","FAKE"],output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)

# ════════════════════════════════════════════════════════════
#  PAGE 5 — PROJECT SUMMARY
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ Project Summary":
    st.markdown('<p class="main-title">ℹ️ Project Summary</p>', unsafe_allow_html=True)
    st.markdown("---")
    sections = {
        "🎯 Problem Statement": "Detect whether a news article is **Fake** or **Real** using NLP and Machine Learning. Binary classification: **1 = Fake**, **0 = Real**.",
        "📦 Dataset": "Kaggle — *Fake and Real News Dataset* (~44,000 articles). Features: `title` + `text` combined into `content`.",
        "🧹 Preprocessing": "Merged datasets → removed nulls/duplicates → lowercased → removed URLs, HTML, special chars → saved as `cleaned_news.csv`.",
        "🔬 Feature Extraction": "TF-IDF Vectorizer: `max_features=5000`, `ngram_range=(1,2)`, English stopwords removed.",
        "🤖 Models Trained": "Logistic Regression · Naive Bayes · Decision Tree · Random Forest · SVM (LinearSVC)",
        "🏆 Conclusion": "TF-IDF + ML achieves **>98% accuracy**. Decision Tree / RF / SVM are top performers. Clear linguistic patterns separate fake from real news.",
        "🚀 Future Work": "BERT/RoBERTa transformers · Source credibility features · FastAPI REST endpoint · Multilingual detection · Live news feed integration.",
    }
    for title, content in sections.items():
        with st.expander(title, expanded=True):
            st.markdown(content)
