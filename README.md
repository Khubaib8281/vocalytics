# 🎙️ SpeechSense: AI-Powered Voice Health & Tone Analyzer

## 🚀 Overview

**SpeechSense** is an advanced Speech AI tool designed to analyze human
voice recordings and provide deep insights into **vocal tone, tract
health, emotional expression, and energy dynamics**.   

It empowers users --- speakers, singers, therapists, and researchers ---
with a **comprehensive report** on vocal quality, health, and
expressiveness, complemented by **interactive visualizations** such as
**spectrograms, mel-spectrograms, and dB-power plots**.

------------------------------------------------------------------------

## 🧠 Key Features

-   🎧 Upload `.wav` or `.opus` voice recordings.

-   🔍 Extracts rich acoustic features using `librosa`, `parselmouth`,
    and signal analysis.

-   📊 Generates visualizations:

    -   Spectrogram
    -   Mel-Spectrogram
    -   Power (Energy-to-dB) plots

-   🤖 AI-driven Voice Report including: 1️⃣ **Overall Voice Summary** 2️⃣
    **Vocal Health Insights** 3️⃣ **Tone & Expression** 4️⃣ **Potential
    Concerns / Red Flags** 5️⃣ **Voice Care Recommendations** 6️⃣
    **Positive Reinforcement**

-   📈 Downloadable analysis reports and plots.

-   ⚡ Deployed using **Streamlit** + **ngrok** for instant web access
    in Colab.

------------------------------------------------------------------------

## 🧩 Example Voice Report

**1️⃣ Overall Voice Summary**\
Your voice recording presents a **moderate energy level**. The voiced
portion shows some breathiness, while pitch variation indicates
expressiveness --- though occasional strain is noticeable.

**2️⃣ Vocal Health Insights**\
A **slightly low harmonic-to-noise ratio (HNR)** suggests mild
roughness. The **strain score** points to possible tension in the vocal
folds.

**3️⃣ Tone & Expression**\
Expressive tone with **emotional undercurrents** --- the voice conveys
sensitivity or emotional depth.

**4️⃣ Potential Concerns**\
Moderate strain detected --- monitor vocal effort and hydration.

**5️⃣ Voice Care Recommendations** - Stay hydrated 💧
- Do light vocal warm-ups 🎵
- Maintain good posture 🧍‍♂️

**6️⃣ Positive Reinforcement**\
Every voice has its unique signature --- with mindful care, you can
maintain vocal longevity and strength.

------------------------------------------------------------------------

## 🖼️ Visualizations Generated

-   **Spectrogram**
-   **Mel-Spectrogram**
-   **Power-to-dB Curve**

All plots are **interactive and downloadable**, making it easy to
compare multiple samples or monitor vocal progress over time.

------------------------------------------------------------------------

## ⚙️ Tech Stack

  -----------------------------------------------------------------------
  Category                   Tools / Libraries
  -------------------------- --------------------------------------------
  Language                   Python 3

  Core Libraries             Librosa, Parselmouth, Numpy, Matplotlib,
                             Soundfile

  Web Framework              Streamlit

-----------------------------------------------------------------------

------------------------------------------------------------------------

## 💡 How to Run in Colab

``` python
!pip install streamlit pyngrok librosa soundfile parselmouth numpy matplotlib
!ngrok authtoken <YOUR_AUTH_TOKEN>

!streamlit run app.py &
from pyngrok import ngrok
public_url = ngrok.connect(8501).public_url
print("App running at:", public_url)
```

------------------------------------------------------------------------

## 🌍 Use Cases

-   🎤 Vocal Health Monitoring
-   🎶 Singing & Tone Analysis
-   🧑‍⚕️ Speech Therapy Aid
-   🧬 Research in Speech Acoustics
-   🧠 AI-based Emotional Voice Profiling

------------------------------------------------------------------------
    
## 💬 Connect

Created by **\[Muhammad Khubaib Ahmad\]** --- passionate about bridging **AI & Human
Expression**.\
Let's connect on [LinkedIn](https://linkedin.com/in/Muhammad-Khubaib-ahmad-) and collaborate on
**Voice Intelligence projects**!

------------------------------------------------------------------------

## 📜 License

MIT License © 2025 Muhammad Khubaib Ahmad\
Use with credit. Contributions welcome!

### ⭐ If you like SpeechSense, give it a star on GitHub and share your feedback!
