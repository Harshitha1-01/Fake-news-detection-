import streamlit as st
import joblib
import os

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")


@st.cache_resource
def load_artifacts():
    if not (os.path.exists("model.pkl") and os.path.exists("vectorizer.pkl")):
        return None, None
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

st.title("📰 Fake News Detection")
st.write("Enter a news headline or article to check whether it is REAL or FAKE.")

if model is None or vectorizer is None:
    st.error(
        "Model files not found. Run `python train_model.py` first "
        "to generate `model.pkl` and `vectorizer.pkl`, then restart this app."
    )
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(
        "This app uses a TF-IDF vectorizer with a Passive Aggressive "
        "Classifier trained on labeled news text to flag content as "
        "REAL or FAKE."
    )
    st.caption("For education/demo purposes — not a substitute for fact-checking.")

news = st.text_area("Enter News Text", height=180, placeholder="Paste a headline or article here...")

col1, col2 = st.columns([1, 1])
with col1:
    check_clicked = st.button("Check News", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear", use_container_width=True)

if clear_clicked:
    st.rerun()

if check_clicked:
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vector = vectorizer.transform([news])
        prediction = model.predict(news_vector)[0]

        # Decision function gives a confidence-like margin for PassiveAggressiveClassifier
        try:
            score = model.decision_function(news_vector)[0]
            confidence = min(abs(score) / 5, 1.0) * 100  # rough scaling for display only
        except Exception:
            confidence = None

        if prediction == "REAL":
            st.success("✅ This news appears to be REAL.")
        else:
            st.error("❌ This news appears to be FAKE.")

        if confidence is not None:
            st.progress(confidence / 100)
            st.caption(f"Model confidence (approximate): {confidence:.1f}%")
