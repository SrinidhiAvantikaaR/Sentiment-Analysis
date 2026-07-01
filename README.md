# Sentiment Analysis Web App

An interactive machine learning application for **sentiment polarity classification** of text data such as movie reviews or social media posts. The project demonstrates the use of **classical NLP techniques** and a lightweight web interface for experimentation and visualization.

---

## Overview

This project implements an end-to-end sentiment analysis pipeline, including text preprocessing, feature extraction, and classification. It focuses on applying **Natural Language Processing (NLP)** fundamentals using traditional machine learning models and deploying them through an interactive web interface.

---

## Features

- Preprocessing of raw text data (cleaning, normalization, tokenization)
- Feature extraction using **TF-IDF vectorization**
- Sentiment polarity classification (e.g., Positive / Negative)
- Interactive Streamlit-based web interface
- Easily extensible to multiple text domains (IMDB reviews, tweets, etc.)
- Upload and batch processing of CSV files containing multiple reviews
- Generates sentiment predictions for all entries in the dataset
- Provides downloadable results/report in CSV or PDF format

---

## Tech Stack

- **Programming Language:** Python  
- **Libraries & Tools:**  
  - Scikit-learn  
  - Pandas  
  - NumPy  
  - Streamlit  
- **NLP Techniques:**  
  - TF-IDF Vectorization  
  - Classical Machine Learning models  

---

## Project Structure

```
├── app.py # Streamlit application
├── model/
│ ├── vectorizer.pkl # Trained TF-IDF vectorizer
│ └── classifier.pkl # Trained ML model
├── data/
│ └── dataset.csv # Training dataset (optional)
├── notebook/
│ └── training.ipynb # Model training and evaluation
├── requirements.txt
└── README.md
```

---

## How It Works

1. User inputs text through the Streamlit interface  
2. Text is preprocessed and transformed using a trained **TF-IDF vectorizer**  
3. The resulting features are passed to a trained machine learning classifier  
4. The predicted sentiment is displayed in real time  

---

## Dataset

The datasets used for training and evaluation (IMDB movie reviews and social media text) are **not included in this repository due to size constraints**.

---

## Installation & Usage

```bash
# Clone the repository (optional)
git clone <repository-url>
cd sentiment-analysis-web-app

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
