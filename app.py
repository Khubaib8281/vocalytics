import streamlit as st
import google.generativeai as genai
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))   
from feature_extractor import analyze_bytes_and_show
from dotenv import load_dotenv
import json
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="🎙️ Vocalytics", layout="centered")

st.title("🎙️ Vocalytics")
st.markdown("Upload your voice sample to get a full acoustic analysis and AI-generated voice report.")

# --- Upload section ---
uploaded_file = st.file_uploader(
    "Upload an audio file (WAV, MP3, OGG, OPUS)",
    type=["wav", "mp3", "ogg", "opus"],
)    

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.info("Analyzing your voice... Please wait ⏳")

    # Read file bytes
    audio_bytes = uploaded_file.read()

    # Run your analysis
    metrics, outputs, fig = analyze_bytes_and_show(audio_bytes)

    # Show metrics & plots
    st.subheader("📊 Acoustic Visualization")
    st.pyplot(fig)

    st.subheader("📈 Metrics Summary")
    st.json(metrics)

    # Prepare Gemini prompt
    prompt = f"""
    You are a professional voice analysis AI.
    Analyze these extracted acoustic features and provide a human-readable voice report.
    Focus on tone, energy, clarity, emotional state, and vocal health hints.

    Metrics:
    {json.dumps(metrics, indent=2)}

    Additional outputs:
    {outputs}

    Be concise but insightful, 3-5 paragraphs.
    """

    # Generate report
    with st.spinner("Generating AI report..."):
        response = model.generate_content(prompt)
        report_text = response.text

    st.subheader("🧾 AI Voice Report")
    st.write(report_text)

else:
    st.warning("Please upload an audio file to begin.")




