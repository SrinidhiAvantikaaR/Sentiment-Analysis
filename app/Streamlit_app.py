import streamlit as st

# Page config
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        height: 100%;
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.95rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze emotions in text with AI-powered sentiment detection</p>', unsafe_allow_html=True)

st.markdown("---")

# Feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">🎬</div>
            <div class="feature-title">IMDB Movie Reviews</div>
            <div class="feature-desc">
                Analyze sentiment in movie reviews and film critiques. 
                Perfect for understanding audience reactions and review trends.
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Social Media Posts</div>
            <div class="feature-desc">
                Evaluate sentiment in tweets, posts, and comments. 
                Ideal for social listening and brand monitoring.
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# How it works section
st.markdown("## 🚀 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div class="info-box">
            <h3 style = "color: black">1️⃣ Choose Dataset</h3>
            <p style = "color: black">Select IMDB reviews or social media posts from the sidebar</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="info-box">
            <h3 style = "color: black">2️⃣ Input Data</h3>
            <p style = "color: black">Type a single text or upload a CSV file for batch analysis</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="info-box">
            <h3 style = "color: black">3️⃣ Get Insights</h3>
            <p style = "color: black">View sentiment predictions and explore key phrases</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Features list
st.markdown("## ✨ Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    - ✅ **Real-time Analysis** - Instant sentiment predictions
    - 📊 **Batch Processing** - Analyze hundreds of texts at once
    - 🎯 **High Accuracy** - Machine learning-powered classification
    """)

with col2:
    st.markdown("""
    - 🔍 **Keyword Extraction** - Discover important phrases
    - 📈 **Visual Analytics** - Charts and sentiment distribution
    - 💾 **Export Results** - Download analyzed data as CSV
    """)

st.markdown("---")

# Call to action
st.markdown("### 👈 Get Started")
st.info("Use the **sidebar on the left** to select your dataset and begin analyzing!")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.9rem;'>Powered by Machine Learning | Built with Streamlit</p>",
    unsafe_allow_html=True
)