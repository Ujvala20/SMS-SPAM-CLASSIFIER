import random
from html import escape
 
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
 
import eda as eda_module
from data import load_data
from features import add_features, FEATURE_COLS
from eda import (class_balance, length_distribution, digit_boxplot,
                 feature_rates, top_words_chart)
from evaluate import (load_results, load_predictions, metrics_comparison,
                      confusion_matrix_fig, roc_fig, pr_fig)
from error_analysis import error_table
 
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")
 
# ======================================================================
# Design tokens
# ======================================================================
INK = "#14233B"
PAPER = "#F5F7FA"
HAM = "#17877A"
HAM_SOFT = "#E1F2EF"
SPAM = "#D43F5E"
SPAM_SOFT = "#FBE6EA"
SLATE = "#5B6B80"
LINE = "#DCE2EA"
 
# red always means spam, teal always means ham (also inside eda.py charts)
eda_module.COLORS.update({"ham": HAM, "spam": SPAM})
 
pio.templates["spamlab"] = go.layout.Template(layout=go.Layout(
    font=dict(family="DM Sans, sans-serif", color=INK, size=13),
    title=dict(font=dict(family="Bricolage Grotesque, sans-serif", size=17, color=INK),
               x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#274C77", "#6096BA", "#8B6FB8", "#C3B1E1", "#E9A23B", "#5B6B80"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
))
pio.templates.default = "plotly_white+spamlab"
 
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');
 
:root {
  --ink:#14233B; --paper:#F5F7FA; --slate:#5B6B80; --line:#DCE2EA;
  --ham:#17877A; --ham-soft:#E1F2EF; --ham-text:#0E6258;
  --spam:#D43F5E; --spam-soft:#FBE6EA; --spam-text:#A82C46;
}
 
.stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input, .stApp textarea {
  font-family: 'DM Sans', sans-serif;
}
.stApp { background: var(--paper); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
  font-family: 'Bricolage Grotesque', sans-serif;
  color: var(--ink);
  letter-spacing: -0.015em;
}
.block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
 
/* ---------- hero ---------- */
.stApp .hero {
  display: grid; grid-template-columns: 1.15fr 1fr; gap: 2.5rem;
  align-items: center; padding: 0.5rem 0 1.8rem 0;
}
.stApp .hero h1 {
  font-size: 3.2rem; line-height: 1.04; font-weight: 800; margin: 0 0 0.9rem 0; padding: 0;
}
.stApp .hero p.lead {
  font-size: 1.12rem; color: var(--slate); max-width: 34rem; line-height: 1.55; margin: 0 0 1.5rem 0;
}
.stApp .hero-stats {
  display: flex; gap: 2.2rem; flex-wrap: wrap; border-top: 1px solid var(--line); padding-top: 1rem;
}
.stApp .hero-stats div { display: flex; flex-direction: column; }
.stApp .hero-stats b {
  font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.8rem; font-weight: 700; line-height: 1.1;
}
.stApp .hero-stats span { color: var(--slate); font-size: 0.86rem; }
.stApp .phone { display: flex; flex-direction: column; gap: 1rem; }
.stApp .bubble {
  background: #fff; border: 1px solid var(--line); border-radius: 18px 18px 18px 4px;
  padding: 0.95rem 1.15rem; font-size: 0.98rem; line-height: 1.5; max-width: 31rem;
}
.stApp .bubble.ham { align-self: flex-end; border-radius: 18px 18px 4px 18px; }
.stApp .bubble .tag {
  display: inline-block; margin-top: 0.6rem; font-weight: 600; font-size: 0.82rem;
  padding: 0.18rem 0.75rem; border-radius: 999px;
}
.stApp .bubble.spam .tag { background: var(--spam-soft); color: var(--spam-text); }
.stApp .bubble.ham .tag { background: var(--ham-soft); color: var(--ham-text); }
 
/* ---------- section headings and callouts ---------- */
.stApp .sec { margin: 1.4rem 0 0.4rem 0; }
.stApp .sec h2 { font-size: 1.65rem; margin: 0; padding: 0; font-weight: 700; }
.stApp .sec p { color: var(--slate); margin: 0.3rem 0 0.8rem 0; max-width: 46rem; line-height: 1.5; }
.stApp .callout {
  border-left: 4px solid var(--ink); background: #EAEFF5; padding: 0.8rem 1rem;
  border-radius: 0 8px 8px 0; margin: 0.5rem 0 0.8rem 0; line-height: 1.5;
}
.stApp .callout.ham { border-color: var(--ham); background: var(--ham-soft); }
.stApp .callout.spam { border-color: var(--spam); background: var(--spam-soft); }
 
/* ---------- pipeline (a real sequence, so numbered) ---------- */
.stApp ul.pipeline {
  list-style: none; counter-reset: step; display: grid;
  grid-template-columns: repeat(6, 1fr); gap: 1.1rem; padding: 0; margin: 0.8rem 0 0 0;
}
.stApp ul.pipeline li { counter-increment: step; border-top: 3px solid var(--ink); padding-top: 0.7rem; margin: 0; }
.stApp ul.pipeline li::before {
  content: counter(step); font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 700; color: var(--slate); display: block; font-size: 0.9rem;
}
.stApp ul.pipeline b {
  display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.1rem; margin: 0.1rem 0 0.25rem 0;
}
.stApp ul.pipeline span { display: block; color: var(--slate); font-size: 0.88rem; line-height: 1.4; }
 
/* ---------- predictor ---------- */
.stApp .verdict {
  display: flex; align-items: baseline; gap: 0.9rem; padding: 1rem 1.2rem;
  border-radius: 0 12px 12px 0; border-left: 6px solid; margin-bottom: 1rem;
}
.stApp .verdict.spam { background: var(--spam-soft); border-color: var(--spam); }
.stApp .verdict.ham { background: var(--ham-soft); border-color: var(--ham); }
.stApp .verdict .label {
  font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.1rem; font-weight: 800; line-height: 1;
}
.stApp .verdict.spam .label { color: var(--spam-text); }
.stApp .verdict.ham .label { color: var(--ham-text); }
.stApp .verdict .prob { color: var(--slate); font-size: 1rem; }
.stApp .sms {
  background: #fff; border: 1px solid var(--line); border-radius: 18px 18px 18px 4px;
  padding: 1rem 1.2rem; font-size: 1.05rem; line-height: 2.05;
}
.stApp .legend-note { color: var(--slate); font-size: 0.86rem; margin-top: 0.5rem; }
 
/* ---------- streamlit widgets ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 600; font-size: 1.05rem;
  color: var(--slate); padding: 0.7rem 0.1rem; background: transparent;
}
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--ink); height: 3px; }
 
[data-testid="stMetric"] {
  background: #fff; border: 1px solid var(--line); border-top: 3px solid var(--ink);
  border-radius: 6px; padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] p { color: var(--slate); font-size: 0.86rem; }
[data-testid="stMetricValue"] { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; }
 
.stButton > button {
  border-radius: 999px; border: 1px solid var(--line); background: #fff; font-weight: 600; padding: 0.35rem 1rem;
}
.stButton > button:hover { border-color: var(--ink); color: var(--ink); }
.stTextArea textarea { border-radius: 12px; background: #fff; font-size: 1.02rem; }
 
@media (max-width: 820px) {
  .stApp .hero { grid-template-columns: 1fr; }
  .stApp .hero h1 { font-size: 2.3rem; }
  .stApp ul.pipeline { grid-template-columns: repeat(2, 1fr); }
}
"""
st.markdown("<style>" + CSS + "</style>", unsafe_allow_html=True)
 
 
# ======================================================================
# Small UI helpers
# ======================================================================
def render(markup):
    """Render HTML. Leading whitespace is stripped so Markdown never treats it as code."""
    st.markdown("".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)
 
 
def section(title, text):
    render(f'<div class="sec"><h2>{title}</h2><p>{text}</p></div>')
 
 
def callout(text, kind="info"):
    render(f'<div class="callout {kind}">{text}</div>')
 
 
def show(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=56))
    st.plotly_chart(fig, theme=None)
 
 
# ======================================================================
# Cached loaders
# ======================================================================
@st.cache_data
def get_data():
    return add_features(load_data())
 
 
@st.cache_data
def get_results():
    return load_results()
 
 
@st.cache_data
def get_predictions():
    return load_predictions()
 
 
@st.cache_resource
def get_model():
    return joblib.load("spam_model.joblib")
 
 
df = get_data()
data = get_results()
results, best = data["results"], data["best_model"]
 
 
@st.cache_data
def make_wordcloud(label):
    palette = (["#D43F5E", "#A82C46", "#E57A8E", "#7A1F33"] if label == "spam"
               else ["#17877A", "#0E6258", "#4FB3A6", "#1F4E79"])
 
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return random.choice(palette)
 
    text = " ".join(df[df["label"] == label]["text"])
    wc = WordCloud(width=900, height=450, background_color=PAPER, stopwords=STOPWORDS,
                   color_func=color_func, prefer_horizontal=0.95).generate(text)
    return wc.to_array()
 
 
# ======================================================================
# Predictor helpers
# ======================================================================
def make_input(texts):
    d = add_features(pd.DataFrame({"text": texts}))
    return d[["text"] + FEATURE_COLS]
 
 
def spam_probability(model, text):
    return float(model.predict_proba(make_input([text]))[0, 1])
 
 
def word_contributions(model, text):
    """Occlusion: remove each word and see how much the spam probability drops."""
    words = text.split()
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    p_full = model.predict_proba(make_input([text]))[0, 1]
    p_without = model.predict_proba(make_input(variants))[:, 1]
    return words, p_full, p_full - p_without
 
 
def highlight(words, contrib):
    biggest = max(np.abs(contrib).max(), 1e-9)
    parts = []
    for w, c in zip(words, contrib):
        safe = escape(w)
        if abs(c) < 0.005:
            parts.append(safe)
            continue
        alpha = min(abs(c) / biggest, 1) * 0.7
        color = "212,63,94" if c > 0 else "23,135,122"
        parts.append(
            f'<span style="background-color: rgba({color},{alpha:.2f}); '
            f'padding:2px 5px; border-radius:4px;">{safe}</span>'
        )
    return " ".join(parts)
 
 
def gauge(p, threshold):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number={"suffix": "%", "font": {"family": "Bricolage Grotesque, sans-serif", "size": 44}},
        title={"text": "Spam probability"},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": SLATE},
            "bar": {"color": SPAM if p >= threshold else HAM, "thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [{"range": [0, threshold * 100], "color": HAM_SOFT},
                      {"range": [threshold * 100, 100], "color": SPAM_SOFT}],
            "threshold": {"line": {"color": INK, "width": 4},
                          "thickness": 0.85, "value": threshold * 100},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=70, b=10, l=30, r=30))
    return fig
 
 
def contribution_chart(words, contrib, n=8):
    labels = [w.strip(".,!?;:\"'()").lower() or w for w in words]
    tmp = (pd.DataFrame({"word": labels, "impact": contrib})
           .groupby("word", as_index=False)["impact"].sum())
    tmp = tmp.reindex(tmp["impact"].abs().sort_values(ascending=False).index).head(n)
    tmp = tmp.sort_values("impact")
    tmp["direction"] = np.where(tmp["impact"] > 0, "towards spam", "towards ham")
    return px.bar(tmp, x="impact", y="word", orientation="h", color="direction",
                  color_discrete_map={"towards spam": SPAM, "towards ham": HAM},
                  title="Words with the biggest influence",
                  labels={"impact": "change in spam probability", "word": ""})
 
 
def error_counts(errors):
    counts = errors["type"].value_counts().reset_index()
    counts.columns = ["type", "count"]
    fig = px.bar(counts, x="type", y="count", color="type",
                 color_discrete_map={"False negative (spam missed)": SPAM,
                                     "False positive (ham flagged)": HAM},
                 title="Errors by type", labels={"type": "", "count": "messages"})
    fig.update_layout(showlegend=False)
    return fig
 
 
EXAMPLES = {
    "spam": ("WINNER! You've won a £900 prize. Call 09061701461 or visit "
             "www.claim-prize.com to claim."),
    "ham": "Hey, are we still meeting at the library at 5? I'll bring the notes.",
}
 
 
def set_example(key):
    st.session_state["msg"] = EXAMPLES[key]
 
 
if "msg" not in st.session_state:
    st.session_state["msg"] = EXAMPLES["spam"]
 
# ======================================================================
# Hero (uses real model output for the two sample messages)
# ======================================================================
model = get_model()
p_spam_example = spam_probability(model, EXAMPLES["spam"])
p_ham_example = spam_probability(model, EXAMPLES["ham"])
 
render(f"""
<div class="hero">
  <div>
    <h1>Is this text spam?</h1>
    <p class="lead">A classifier trained on {len(df):,} real SMS messages. It gives a verdict,
    a probability, and shows which words decided it.</p>
    <div class="hero-stats">
      <div><b>{len(df):,}</b><span>messages</span></div>
      <div><b>{df['y'].mean():.0%}</b><span>are spam</span></div>
      <div><b>{results[best]['f1']:.3f}</b><span>F1 of the best model</span></div>
      <div><b>{len(results)}</b><span>models compared</span></div>
    </div>
  </div>
  <div class="phone">
    <div class="bubble spam">{escape(EXAMPLES['spam'])}<br><span class="tag">Spam, {p_spam_example:.0%} likely</span></div>
    <div class="bubble ham">{escape(EXAMPLES['ham'])}<br><span class="tag">Ham, {1 - p_ham_example:.0%} likely</span></div>
  </div>
</div>
""")
 
tab_overview, tab_eda, tab_models, tab_predict, tab_errors = st.tabs(
    ["Overview", "Data Analysis", "Model Comparison", "Live Predictor", "Error Analysis"]
)
 
# ======================================================================
# Overview
# ======================================================================
with tab_overview:
    section("What this project does",
            "It reads an SMS and decides whether it is spam or a legitimate message (ham).")
    left, right = st.columns([1, 1], gap="large")
    with left:
        show(class_balance(df))
    with right:
        st.write(
            "Only about one message in eight is spam, so accuracy alone is misleading: "
            "a model that always answers \"ham\" would still be right most of the time. "
            "The project therefore judges models on precision, recall and F1, "
            "and tests whether handcrafted signals (URL, digits, length, currency) help."
        )
        b = results[best]
        callout(f"<b>Best model:</b> {best}<br>"
                f"F1 {b['f1']:.3f}, precision {b['precision']:.3f}, recall {b['recall']:.3f}",
                "ham")
 
    section("How it works", "Six steps from raw messages to an explained prediction.")
    render("""
    <ul class="pipeline">
      <li><b>Clean</b><span>Remove duplicate messages and label spam as 1, ham as 0.</span></li>
      <li><b>Features</b><span>Length, word count, digits, URL, currency symbol, capitals.</span></li>
      <li><b>TF-IDF</b><span>Turn words and word pairs into weighted numbers.</span></li>
      <li><b>Models</b><span>Train Naive Bayes and Logistic Regression, with and without features.</span></li>
      <li><b>Evaluate</b><span>Precision, recall, F1, ROC and precision-recall curves.</span></li>
      <li><b>Explain</b><span>Highlight the words that pushed each prediction.</span></li>
    </ul>
    """)
 
# ======================================================================
# Data Analysis
# ======================================================================
with tab_eda:
    section("What spam looks like",
            "Spam differs from normal messages in length, digits, links and money.")
 
    spam_df, ham_df = df[df["label"] == "spam"], df[df["label"] == "ham"]
    callout(
        f"Spam messages average <b>{spam_df['length'].mean():.0f}</b> characters "
        f"versus <b>{ham_df['length'].mean():.0f}</b> for ham, and contain "
        f"<b>{spam_df['digit_count'].mean():.1f}</b> digits versus "
        f"<b>{ham_df['digit_count'].mean():.1f}</b>.<br>"
        f"<b>{spam_df['has_url'].mean():.0%}</b> of spam contains a URL "
        f"(ham: {ham_df['has_url'].mean():.0%}) and "
        f"<b>{spam_df['has_currency'].mean():.0%}</b> contains a currency symbol "
        f"(ham: {ham_df['has_currency'].mean():.0%}).",
        "spam")
 
    c1, c2 = st.columns(2, gap="large")
    with c1:
        show(length_distribution(df))
    with c2:
        show(digit_boxplot(df))
 
    c3, c4 = st.columns(2, gap="large")
    with c3:
        show(feature_rates(df))
    with c4:
        st.markdown("**Average of each feature**")
        avg = df.groupby("label")[["length", "word_count", "digit_count",
                                   "has_url", "has_currency", "caps_ratio"]].mean().round(3)
        st.dataframe(avg.T)
 
    section("Most common words", "Frequent words in each class, after removing common stop words.")
    n_words = st.slider("Number of words", 5, 30, 15)
    c5, c6 = st.columns(2, gap="large")
    with c5:
        show(top_words_chart(df, "spam", n_words))
    with c6:
        show(top_words_chart(df, "ham", n_words))
 
    section("Word clouds", "Larger words appear more often.")
    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.caption("Spam")
        st.image(make_wordcloud("spam"))
    with w2:
        st.caption("Ham")
        st.image(make_wordcloud("ham"))
 
# ======================================================================
# Model Comparison
# ======================================================================
with tab_models:
    section("Which model is best?",
            "Four models were trained on the same split and scored on the same unseen test messages.")
 
    table = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, color="#CDEBE6"))
    callout(f"Best model by F1: <b>{best}</b>. The highlighted cell in each column is the top score.", "ham")
 
    show(metrics_comparison(results))
 
    section("Confusion matrix", "Pick a model to see exactly how many messages it got right and wrong.")
    chosen = st.selectbox("Model", list(results.keys()),
                          index=list(results.keys()).index(best))
    show(confusion_matrix_fig(results, chosen))
 
    section("Ranking quality", "How well each model separates spam from ham across all thresholds.")
    preds = get_predictions()
    r1, r2 = st.columns(2, gap="large")
    with r1:
        show(roc_fig(preds))
    with r2:
        show(pr_fig(preds))
 
    section("Did the extra features help?",
            "F1 with the handcrafted features compared with TF-IDF alone.")
    m1, m2 = st.columns(2)
    for col, base in zip((m1, m2), ["Naive Bayes", "Logistic Regression"]):
        f_without = results[f"{base} (TF-IDF)"]["f1"]
        f_with = results[f"{base} (TF-IDF + features)"]["f1"]
        col.metric(base, f"{f_with:.3f}", delta=f"{f_with - f_without:+.3f} vs TF-IDF alone")
 
# ======================================================================
# Live Predictor
# ======================================================================
with tab_predict:
    section("Check a message",
            "Paste any text. The model returns a verdict, a probability and the words behind it.")
 
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        b1, b2 = st.columns(2)
        b1.button("Load a spam example", on_click=set_example, args=("spam",))
        b2.button("Load a ham example", on_click=set_example, args=("ham",))
        message = st.text_area("Message", key="msg", height=150,
                               label_visibility="collapsed",
                               placeholder="Paste an SMS message here")
        threshold = st.slider(
            "Flag as spam at or above this probability",
            0.05, 0.95, 0.50, 0.05,
            help="Lower catches more spam but may flag real messages. "
                 "Higher causes fewer false alarms but may miss spam.")
 
    with right:
        if message.strip():
            words, p, contrib = word_contributions(model, message)
            is_spam = p >= threshold
            kind = "spam" if is_spam else "ham"
            label = "Spam" if is_spam else "Ham"
            shown_p = p if is_spam else 1 - p
            render(f'<div class="verdict {kind}"><span class="label">{label}</span>'
                   f'<span class="prob">{shown_p:.1%} likely</span></div>')
            render(f'<div class="sms">{highlight(words, contrib)}</div>')
            render('<div class="legend-note">Red words pushed the model towards spam, '
                   'green words towards ham. Darker means a stronger push.</div>')
        else:
            callout("Type or paste a message on the left to see the prediction.")
 
    if message.strip():
        g1, g2 = st.columns(2, gap="large")
        with g1:
            show(gauge(p, threshold))
        with g2:
            show(contribution_chart(words, contrib))
 
        section("Message signals", "Simple measurements the model also uses.")
        f = add_features(pd.DataFrame({"text": [message]})).iloc[0]
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("URL detected", "Yes" if f["has_url"] else "No")
        s2.metric("Digits", int(f["digit_count"]))
        s3.metric("Currency symbol", "Yes" if f["has_currency"] else "No")
        s4.metric("Length", int(f["length"]))
        s5.metric("Capitals", f"{f['caps_ratio']:.0%}")
 
# ======================================================================
# Error Analysis
# ======================================================================
with tab_errors:
    section("Where does the model go wrong?",
            f"Mistakes made by {best} on the test messages it had never seen.")
 
    err_threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, key="err_thr")
    preds_all = get_predictions()
    errors = error_table(preds_all, best, err_threshold)
 
    n_fn = int((errors["type"] == "False negative (spam missed)").sum())
    n_fp = int((errors["type"] == "False positive (ham flagged)").sum())
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test messages", len(preds_all))
    m2.metric("Total errors", len(errors))
    m3.metric("Spam missed", n_fn)
    m4.metric("Ham flagged as spam", n_fp)
 
    if errors.empty:
        callout("No errors at this threshold.", "ham")
    else:
        e1, e2 = st.columns([1, 2], gap="large")
        with e1:
            show(error_counts(errors))
        with e2:
            choice = st.radio("Show", ["All errors", "Spam missed", "Ham flagged as spam"],
                              horizontal=True)
            shown = errors.copy()
            if choice == "Spam missed":
                shown = shown[shown["type"].str.startswith("False negative")]
            elif choice == "Ham flagged as spam":
                shown = shown[shown["type"].str.startswith("False positive")]
 
            shown["wrong_confidence"] = np.where(
                shown["type"].str.startswith("False negative"),
                1 - shown["spam_probability"], shown["spam_probability"]).round(3)
            shown = shown.sort_values("wrong_confidence", ascending=False)
            st.dataframe(shown[["type", "text", "spam_probability",
                                "wrong_confidence", "reason"]])
            st.caption("wrong_confidence is how sure the model was about its mistake. "
                       "Near 0.5 is a borderline case, near 1 is a confident mistake.")
 
    section("Threshold trade-off",
            "Lowering the threshold catches more spam but flags more real messages. "
            "Raising it does the opposite.")
    thresholds = np.arange(0.05, 0.96, 0.05)
    y_true = preds_all["y_true"].values
    proba = preds_all[best].values
    rows = []
    for t in thresholds:
        pred = proba >= t
        rows.append({"threshold": round(float(t), 2),
                     "Spam missed": int(((~pred) & (y_true == 1)).sum()),
                     "Ham flagged as spam": int((pred & (y_true == 0)).sum())})
    trade = pd.DataFrame(rows).melt(id_vars="threshold", var_name="error type",
                                    value_name="count")
    show(px.line(trade, x="threshold", y="count", color="error type", markers=True,
                 title="Errors at different thresholds",
                 color_discrete_map={"Spam missed": SPAM, "Ham flagged as spam": HAM}))
 
    section("What the mistakes have in common", "")
    callout("<b>Spam missed</b> is usually spam written like casual conversation, "
            "with no URL, currency symbol or typical trigger words.", "spam")
    callout("<b>Ham flagged as spam</b> is usually a real message containing a phone number, "
            "a money amount, or words like <i>call</i> and <i>free</i>.", "ham")
    callout("The model relies on vocabulary, so it struggles when a message avoids "
            "spam-style wording.")
 
