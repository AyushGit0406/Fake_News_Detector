import streamlit as st
import pickle
import numpy as np
import re
import string

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


@st.cache_resource
def load_assets():
    model = load_model('model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('params.pkl', 'rb') as f:
        params = pickle.load(f)
    return model, tokenizer, params['MAX_LEN']

model, tokenizer, MAX_LEN = load_assets()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_news(title_text, body_text=""):
  
    combined = title_text.strip()
    if body_text.strip():
        combined = combined + " " + body_text.strip()

    cleaned = clean_text(combined)

    if len(cleaned.split()) < 3:
        return None, None, None, "too_short"

    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob   = float(model.predict(padded, verbose=0)[0][0])

    label      = 'REAL' if prob > 0.5 else 'FAKE'
    confidence = prob if prob > 0.5 else (1 - prob)
    return label, confidence, prob, "ok"

#  UI 
st.title("Fake News Detector")
st.markdown(
    "Enter a **headline** (required) and optionally the **article body**. "
    "The model was trained on both together, so providing more text gives a more accurate result."
)
st.markdown("---")


title_input = st.text_input(
    label=" Headline / Title (required)",
    placeholder="e.g. Scientists confirm that drinking coffee daily reduces risk of heart disease"
)

body_input = st.text_area(
    label="📄 Article Body ",
    placeholder="Paste the full article text here...",
    height=180
)

#  Detect button 
if st.button(" Detect", use_container_width=True):
    if not title_input.strip() and not body_input.strip():
        st.warning("Please enter at least a headline.")
    else:
        with st.spinner("Analysing..."):
            label, confidence, raw_prob, status = predict_news(title_input, body_input)

        if status == "too_short":
            st.warning("The text is too short for a reliable prediction. Please add more content.")
        else:
            st.markdown("---")

            #  Result card 
            if label == "FAKE":
                st.error("## FAKE NEWS")
                st.markdown(
                    f"The model predicts this article is **fake** "
                    f"with **{confidence * 100:.1f}% confidence**."
                )
            else:
                st.success("## REAL NEWS")
                st.markdown(
                    f"The model predicts this article is **real** "
                    f"with **{confidence * 100:.1f}% confidence**."
                )

            #  Confidence bar 
            st.markdown("#### Confidence Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="FAKE probability", value=f"{(1 - raw_prob) * 100:.1f}%")
            with col2:
                st.metric(label="REAL probability", value=f"{raw_prob * 100:.1f}%")

            st.progress(float(raw_prob))
            st.caption("◀ More FAKE  |  More REAL ▶")

            # Debug: show what went into the model 
            with st.expander("See cleaned text fed into the model"):
                combined_raw = title_input.strip()
                if body_input.strip():
                    combined_raw += " " + body_input.strip()
                st.write(clean_text(combined_raw))

            # Low confidence warning
            if confidence < 0.65:
                st.info(
                    "Confidence is below 65%. Consider adding more article text "
                    "or cross-checking with a trusted news source."
                )

st.markdown("---")


