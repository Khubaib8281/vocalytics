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
You are a **professional voice health analyst and speech specialist**. 
Your role is to interpret the following **acoustic features** to create a 
professional, yet easy-to-understand **Voice Health Report** that can help 
the person understand their **vocal condition**, possible **strain or fatigue**, 
and overall **vocal well-being**.

Here are the analyzed voice features:
{metrics}, output: {outputs}

Write a clear, human-readable report titled:
🧾 "Voice Health & Quality Report"

🧠 **Tone & Style Guidelines:**
- Use **plain language** anyone can understand.
- Keep a **calm, professional, and empathetic tone** (like a speech pathologist explaining results).
- Avoid raw numbers or units (Hz, dB, etc.), but interpret what they mean — e.g., “your voice sounds slightly breathy or strained”.
- Mention **possible throat health signs** like tension, dryness, breathiness, fatigue, or imbalance.
- If jitter or shimmer are missing, say: “Some fine voice stability measures weren’t available, so this report may be partially limited.”
- Keep the structure well-organized and professional.
- Use short paragraphs or bullet points for clarity.

📋 **Report Structure:**

1️⃣ **Overall Voice Summary:**  
Describe how the voice generally sounds — calm, tense, steady, tired, or expressive. Mention clarity, steadiness, and vocal balance.

2️⃣ **Vocal Health Insights:**  
Explain if there are signs of throat strain, breathiness, roughness, fatigue, or dryness.  
Use cues from jitter, shimmer, and HNR to assess vocal stability and health.

3️⃣ **Tone & Expression:**  
Discuss emotional impression — confident, tired, neutral, expressive, etc.  
If wetness or tone scores indicate emotional states (sad, energetic, relaxed), explain that in plain words.

4️⃣ **Potential Concerns or Red Flags:**  
Gently mention if the data shows signs that may suggest vocal tension, overuse, or possible early vocal fatigue — always phrased supportively, not alarmingly.

5️⃣ **Voice Care Recommendations:**  
Offer 2–3 practical suggestions to maintain or improve vocal health.  
For example: hydration, warm-ups, posture, rest, or reducing vocal strain.

6️⃣ **Positive Reinforcement:**  
End with a supportive line recognizing the person’s unique voice and encouraging them to take care of it.

🩺 **Output Goal:**  
Create a report that feels professional enough for a health or research setting, 
yet gentle and clear enough for non-technical users. 
Avoid overly technical or emotional exaggeration — balance expertise with empathy.

Keep it under 250 words.
"""




    # Generate report
    with st.spinner("Generating AI report..."):
        response = model.generate_content(prompt)
        report_text = response.text

    st.subheader("🧾 AI Voice Report")
    st.write(report_text)

else:
    st.warning("Please upload an audio file to begin.")




