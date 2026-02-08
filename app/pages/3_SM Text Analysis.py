import streamlit as st
import joblib

def clear_text():
    st.session_state["input_text"] = ""

st.title("Social Media Text Analysis")

vectorizer = joblib.load("D:\Project1\Sentiment-Analysis\models\social_media_vectorizer.joblib")
model = joblib.load("D:\Project1\Sentiment-Analysis\models\social_media_model.joblib")

st.write("Text for analysis: ")
input = st.text_area("Enter text", key="input_text")

col1, col2 = st.columns(2)
with col1:
 if st.button("Predict", type = "primary"):
    if input.strip():
        vec = vectorizer.transform([input])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.success("Sentiment: **Positive**")
        elif pred == -1:
            st.error("Sentiment: **Negative**")
        else:
            st.info("Sentiment: **Neutral**")
    else:
        st.warning("Please enter text.")
with col2:
 st.button("Clear", on_click=clear_text)
