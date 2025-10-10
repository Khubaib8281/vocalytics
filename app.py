import streamlit as st
import google.generativeai as genai
import os, sys
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
You are a professional voice analyst writing an easy-to-understand and empathetic
voice report for a non-technical person (like a singer, speaker, or everyday user).

Below is the extracted acoustic data:
{metrics} and outputs: {outputs}

Write a report titled: "🎙️ AI Voice Analysis Report"
    
Structure the report in this format:

1️⃣ **Overview** – Summarize what kind of voice this appears to be overall
   (e.g., calm, energetic, tense, soft).

2️⃣ **Tone & Emotion** – Explain the tone score, pitch, and wetness in simple language.
   Use intuitive emotional terms like “relaxed”, “tired”, “confident”, or “emotional”.
   Avoid technical jargon. Mention if the voice feels steady or fluctuating.

3️⃣ **Energy & Power** – Talk about loudness and energy values (energy_mean, energy_std, peak_energy)
   in everyday terms — e.g., “Your voice stays mostly gentle, with occasional bursts of energy.”

4️⃣ **Clarity & Smoothness** – Explain HNR, jitter, and shimmer (if available) without formulas.
   Say things like “a touch of breathiness” or “slight vocal tension” instead of numerical values.
    
5️⃣ **Vocal Health Tips** – Give friendly, practical advice (like hydration, breathing, relaxing the throat).
   Focus on clarity and warmth, not medical claims.

6️⃣ **Summary Line** – End with a one-line encouraging conclusion,
   e.g., “You have a warm, natural tone that could sound even more expressive with a bit more energy!”

Guidelines:
- Use short, clear sentences.
- Avoid numbers unless needed to support a point.
- Keep tone supportive, kind, and confidence-boosting.
"""


    # Generate report
    with st.spinner("Generating AI report..."):
        response = model.generate_content(prompt)
        report_text = response.text

    st.subheader("🧾 AI Voice Report")
    st.write(report_text)

else:
    st.warning("Please upload an audio file to begin.")
