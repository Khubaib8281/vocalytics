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

    # st.subheader("📈 Metrics Summary")
    # st.json(metrics)

    # Prepare Gemini prompt
    prompt = f"""
You are a friendly professional voice coach. Your job is to take the following
acoustic features and describe what they mean in **simple, natural human language**,
so that even a beginner with no technical knowledge can understand.

Here are the analyzed voice features:
{metrics}, output: {outputs}

Write a short, easy-to-read voice report titled:
🎙️ "Voice Analysis Report"

Follow this style guide:
- Use plain, everyday language.
- Avoid any technical or scientific terms (like Hz, dB, centroid, spectral, etc.)
- Instead, explain what the data means in how the voice **feels** or **sounds**.
- Be warm, conversational, and encouraging, but be clear and say the truth.
- If some features like jitter or shimmer are missing, just say “Some tiny voice stability details weren’t available.”
- Use bullet points or short paragraphs so it’s easy to read.
- End with a kind motivational line about the person’s voice.

Structure:
1️⃣ **Overall Impression:** Talk about how the voice generally feels (calm, energetic, confident, soft, etc.)
2️⃣ **Tone & Emotion:** Describe the emotional impression — cheerful, steady, gentle, serious, or expressive.
3️⃣ **Clarity & Smoothness:** Explain if the voice sounds clear, breathy, or slightly tense.
4️⃣ **Energy & Flow:** Mention if the voice feels steady, varies naturally, or gets louder/softer.
5️⃣ **Helpful Tips:** Give 2–3 simple, supportive tips for improving or maintaining a healthy tone.
6️⃣ **Positive Summary:** End with one uplifting compliment.

Keep it under 200 words, sound **human, kind, and natural** — as if you’re talking directly to the speaker.
"""



    # Generate report
    with st.spinner("Generating AI report..."):
        response = model.generate_content(prompt)
        report_text = response.text

    st.subheader("🧾 AI Voice Report")
    st.write(report_text)

else:
    st.warning("Please upload an audio file to begin.")



