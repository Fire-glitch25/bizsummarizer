import streamlit as st
from transformers import pipeline
from fpdf import FPDF
from streamlit_lottie import st_lottie
import requests
import base64

# --- Helper for Lottie animation ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Load Lottie animation (you can change the URL for a different animation) ---
lottie_json = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_j1adxtyb.json")

# --- Set Streamlit page config ---
st.set_page_config(page_title="Kelly Summarizer", page_icon="📝", layout="centered")

# --- Custom CSS for background and UI ---
st.markdown("""
    <style>
        body {
            background-image: url('https://images.unsplash.com/photo-1607746882042-944635dfe10e?auto=format&fit=crop&w=1470&q=80');
            background-size: cover;
            background-attachment: fixed;
        }
        .main {
            background-color: rgba(0,0,0,0.6);
            border-radius: 10px;
            padding: 2rem;
            color: white;
        }
        .stButton>button {
            background-color: #ff4e84;
            color: white;
            border-radius: 8px;
            padding: 0.5em 2em;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- App title and animation ---
st.markdown("<h1 style='text-align: center; color: white;'>📝 Kelly Summarizer</h1>", unsafe_allow_html=True)
st_lottie(lottie_json, height=180, key="animation")

# --- Summarizer pipeline ---
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summarizer()

# --- Main UI ---
with st.form("summarize_form"):
    text = st.text_area("Paste text here...", height=200)
    submitted = st.form_submit_button("Summarize")
    summary = ""
    if submitted and text.strip():
        with st.spinner("Summarizing..."):
            summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
        st.markdown(
            f"<div class='main'><h3>Summary:</h3><p>{summary}</p></div>",
            unsafe_allow_html=True
        )

        # --- PDF Download ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, summary)
        pdf_output = pdf.output(dest='S').encode('latin-1')
        b64 = base64.b64encode(pdf_output).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="summary.pdf" style="text-decoration:none;"><button>📥 Download PDF</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# --- Celebratory animation if summary is generated ---
if summary:
    st.balloons()
