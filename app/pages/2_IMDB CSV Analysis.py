import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import re
from fpdf import FPDF
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ============================================
# LOAD TRAINED MODEL + TRAINED VECTORIZER
# ============================================

VECTORIZER_PATH = "D:/Project1/Sentiment-Analysis/models/imdb_vectorizer.joblib"
MODEL_PATH = "D:/Project1/Sentiment-Analysis/models/imdb_model.joblib"

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


# ============================================
# CLEANING
# ============================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================
# MODEL PREDICTION
# ============================================

def get_sentiment(text):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    # ---- normalize output labels ----
    if pred in [1, "pos", "positive", "Positive"]:
        return "Positive"
    else:
        return "Negative"


# ============================================
# TF-IDF WORD EXTRACTION
# ============================================

def top_tfidf_words(texts, n=10):
    if len(texts) == 0:
        return []

    
    # Expanded stop words
    extra_stops = [
        'film', 'movie', 'movies', 'films', 'br', 'one', 'like', 
        'just', 'get', 'even', 'also', 'would', 'really', 'much',
        'make', 'made', 'characters', 'time', 'good', 'bad', 
        'plot', 'story', 'don', 'doesn', 'nan', 've', 'll'    
    ]
    
    all_stops = list(ENGLISH_STOP_WORDS) + extra_stops

    tfidf = TfidfVectorizer(
        stop_words=all_stops,
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,      # Increase from 2 to 3
        max_df=0.7     # Decrease from 0.8 to 0.7
    )

    tfidf_matrix = tfidf.fit_transform(texts)
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    words = tfidf.get_feature_names_out()

    ranking = sorted(zip(words, scores), key=lambda x: x[1], reverse=True)
    return ranking[:n]


# ============================================
# STREAMLIT APP
# ============================================

st.title("CSV Sentiment Analyzer + TF-IDF Keyword Explorer")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    text_col = st.selectbox(
        "Select the column containing text",
        df.columns
    )

    if st.button("Run Analysis"):

        # --------------------------
        # SENTIMENT ANALYSIS
        # --------------------------
        st.subheader("Sentiment Classification")

        df["cleaned"] = df[text_col].astype(str).apply(clean_text)
        df["sentiment"] = df[text_col].apply(get_sentiment)

        st.write(df[["cleaned", "sentiment"]].head())

        # Debug safety check
        st.write("Sentiment counts:")
        st.write(df["sentiment"].value_counts())

        # --------------------------
        # SENTIMENT DISTRIBUTION
        # --------------------------
        st.subheader("Sentiment Distribution")

        fig, ax = plt.subplots()
        df["sentiment"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)
        fig.savefig("sentiment_plot.png", dpi=300, bbox_inches="tight")



        # --------------------------
        # TF-IDF TOP WORDS
        # --------------------------
        st.subheader("Top TF-IDF Keywords")

        positive_texts = df[df.sentiment == "Positive"]["cleaned"].tolist()
        negative_texts = df[df.sentiment == "Negative"]["cleaned"].tolist()

        pos_len = len(positive_texts)
        neg_len = len(negative_texts)
        st.write("Positive samples:", pos_len)
        st.write("Negative samples:", neg_len)
        
        positive_percentage = (pos_len/ (pos_len + neg_len)) * 100
        negative_percentage = (neg_len/ (pos_len + neg_len)) * 100

        # Positive keywords
        st.write("### Top Positive Keywords")
        pos_words = top_tfidf_words(positive_texts)
        pos_block = pd.DataFrame(pos_words, columns=["Word", "Score"])
        st.write(pos_block)

        # Negative keywords
        st.write("### Top Negative Keywords")
        neg_words = top_tfidf_words(negative_texts)
        neg_block = pd.DataFrame(neg_words, columns=["Word", "Score"])
        st.write(neg_block)

        summary_text = f"""
Sentiment Analysis Report
-------------------------

Total rows analyzed: {pos_len + neg_len}

Positive: {pos_len} ({positive_percentage:.2f}%)
Negative: {neg_len} ({negative_percentage:.2f}%)

Top Positive Words:
{pos_block}

Top Negative Words:
{neg_block}
"""


        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", size=16)
        pdf.cell(0, 10, "Sentiment Analysis Report", ln=True, align="C")

        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        pdf.ln(4)

        pdf.multi_cell(0, 6, summary_text)

        pdf.ln(5)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, "Sentiment Breakdown Chart:", ln=True)

        pdf.image("sentiment_plot.png", w=150)

        pdf_output = pdf.output(dest="S").encode("latin-1")

        st.download_button(
            label="Download PDF Report",
            data=pdf_output,
            file_name="sentiment_report.pdf",
            mime="application/pdf"
        )
