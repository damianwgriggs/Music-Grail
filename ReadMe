# 🎬 Creator Studio: Beat Blueprint & Shorts Maker

Welcome to the **Creator Studio toolkit**, a dual-app Streamlit ecosystem designed for independent musicians, filmmakers, and DIY creators. Submitted for **No Film School**, this project bridges the gap between AI music generation, vocal tracking, and music video production. 

This repository contains two powerful web-based applications that allow you to produce full music videos and social-media-ready Shorts directly from your browser—no expensive DAWs or NLEs required.

---

## 🛠️ The Tools

### 1. Beat Blueprint Studio (`app.py`)
A hybrid audio-video mixing console that turns raw vocal takes into exact synth stems, mixes them with AI beats, and sequences AI video clips into a final music video.

*   **🎙️ Vocal Tone Digitizer:** Record raw vocals in the browser. The app extracts exact continuous pitch (Hz) and volume dynamics.
*   **🎹 Synthesizer Engine:** Digitize your voice into one of four synth presets (e.g., Deep Moog, Haunting Cinematic Lead). It maps your exact human inflection onto pure synthesized waveforms—zero raw voice remains.
*   **🎚️ Hybrid Mixer:** Import AI instrumentals (like Suno beats), blend them with your synth stems, and overlay raw vocals featuring automated "Vocal Polish" (rumble cut, compression, and dark reverb).
*   **🎬 Music Video Sequencer:** Upload AI-generated video clips. The app strips the original AI sound, lets you loop and arrange scenes in a "Flow-style" timeline, and uses FFmpeg to merge the silent sequence seamlessly with your mixed track.

### 2. Shorts Maker Studio (`shortapp.py`)
A rapid-fire vertical video generator to turn your finished music videos (or comedy episodes) into TikTok/Reels/Shorts-ready content.

*   **✂️ Timestamp Slicer:** Easily extract highlights using simple `MM:SS` to `MM:SS` inputs.
*   **📱 Auto-Framing:** Automatically reframe widescreen landscape videos to 1080x1920 (9:16). Choose between *Center Crop* (for solo shots) or *Widescreen + Blurred Background* padding.
*   **💬 Dynamic Subtitles:** Paste your lyrics or dialogue. Use timestamped text (e.g., `[00:05] Lyrics here`) or paste plain text to have the app automatically distribute the words evenly across the clip. Burns captions directly into the video file.

---

## 🚀 Installation & Setup

### Prerequisites
1. **Python 3.9+**
2. **FFmpeg:** Crucial for video processing and audio stripping. 
   * *Note: The apps will attempt to use `imageio-ffmpeg` if a system-wide FFmpeg installation is not found.*

### Quick Start
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/creator-studio.git
cd creator-studio
pip install -r requirements.txt
```

*(If you don't have a `requirements.txt` yet, install the core libraries manually):*
```bash
pip install streamlit librosa numpy soundfile scipy imageio-ffmpeg
```

### Running the Apps
The tools are split into two separate Streamlit interfaces. Open two terminal windows to run them side-by-side:

**To launch Beat Blueprint Studio:**
```bash
streamlit run app.py
```

**To launch Shorts Maker Studio:**
```bash
streamlit run shortapp.py
```

---

## 📂 Folder Structure
The apps automatically generate local directories to manage your project assets safely:
*   `saved_takes/` - Raw audio recordings and pitch JSON data.
*   `saved_stems/` - Digitized synthesizer WAV files.
*   `suno_beats/` - Imported AI instrumentals.
*   `generated_beats/` - Your final mixed WAV audio tracks.
*   `uploaded_clips/` - Raw AI video clips before audio stripping.
*   `generated_videos/` - Final assembled horizontal music videos.
*   `generated_shorts/` - Sliced, vertical 9:16 videos with burned-in captions.

---

## 💡 Why This Matters for Indie Creators (No Film School)
For the modern independent creator, the workflow is often fragmented across a dozen expensive subscriptions: vocal recording in a DAW, AI beat generation in the cloud, video generation in another app, and final assembly in Premiere or Resolve. 

This toolkit consolidates the experimental "AI + Human" workflow into an open-source, accessible, and fast pipeline. It allows filmmakers and musicians to prototype music videos, translate vocal ideas into synth lines without knowing how to play piano, and instantly generate promotional shorts—all powered by Python.

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE). Feel free to fork, modify, and build upon it for your own studio setup!
