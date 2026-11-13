import io
import json
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import butter, lfilter, medfilt

# --- SETUP ---
st.set_page_config(page_title="Beat Blueprint Studio", page_icon="🎛️", layout="wide")
SAVE_DIR = Path("saved_takes")
STEM_DIR = Path("saved_stems")
BEAT_DIR = Path("generated_beats")

SAVE_DIR.mkdir(exist_ok=True)
STEM_DIR.mkdir(exist_ok=True)
BEAT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 44100


# --- AUDIO HELPERS ---
def make_metronome_wav(bpm: int, fs: int = 44100) -> bytes:
    sec_per_beat = 60.0 / bpm
    total_sec = 4 * sec_per_beat
    t = np.linspace(0, total_sec, int(fs * total_sec), endpoint=False)
    clicks = np.zeros_like(t, dtype=np.float32)

    for beat in range(4):
        start = int(beat * sec_per_beat * fs)
        length = int(0.05 * fs)
        pitch = 1000 if beat == 0 else 700
        beep_t = np.linspace(0, 0.05, length, endpoint=False)
        beep = 0.25 * np.sin(2 * np.pi * pitch * beep_t) * np.exp(-beep_t * 45)
        if start + length < len(clicks):
            clicks[start : start + length] += beep

    buf = io.BytesIO()
    sf.write(buf, clicks, fs, format="WAV")
    return buf.getvalue()


def to_wav_bytes(audio: np.ndarray, fs: int = 44100) -> bytes:
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak) * 0.90
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32), fs, format="WAV")
    return buf.getvalue()


def add_dark_reverb(audio: np.ndarray, fs: int, wet: float = 0.30) -> np.ndarray:
    if wet <= 0.01:
        return audio
    taps = [(0.043, 0.45), (0.067, 0.35), (0.097, 0.25), (0.137, 0.18), (0.193, 0.12)]
    reverb = np.zeros_like(audio)
    for delay_sec, gain in taps:
        d = int(delay_sec * fs)
        if d < len(audio):
            reverb[d:] += audio[:-d] * gain
    b, a = butter(2, 2200 / (fs / 2), btype="low")
    reverb = lfilter(b, a, reverb)
    return (1.0 - wet) * audio + wet * reverb


# --- STEP 1 & 2: EXACT TONE & DYNAMICS EXTRACTOR ---
def extract_exact_tones(y: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Extracts continuous pitch (Hz) and exact volume envelope frame-by-frame.
    Uses a tiny 5-frame median filter only to remove single-frame octave glitches
    while keeping every real vocal slide, vibrato, and inflection intact.
    """
    hop = 256  # ~5.8ms resolution for tight vocal tracking
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    max_rms = float(np.max(rms))
    if max_rms < 1e-6:
        return np.array([]), np.array([]), hop

    amp_frames = rms / max_rms
    f0 = librosa.yin(y, fmin=45, fmax=850, sr=fs, hop_length=hop)

    valid = np.isfinite(f0) & (f0 >= 45.0) & (f0 <= 850.0) & (amp_frames > 0.06)
    f0_clean = np.empty_like(f0, dtype=np.float64)

    if np.any(valid):
        last_val = float(f0[valid][0])
        for i in range(len(f0)):
            if valid[i]:
                last_val = float(f0[i])
            f0_clean[i] = last_val
    else:
        f0_clean.fill(110.0)

    f0_clean = medfilt(f0_clean, kernel_size=5)
    return f0_clean, amp_frames, hop


def analyze_recording(audio_bytes: bytes, bpm: int) -> dict:
    y, fs = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if fs != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=fs, target_sr=SAMPLE_RATE)
        fs = SAMPLE_RATE

    if len(y) < 2048:
        return {"error": "Recording was too short. Try recording at least 1–2 seconds!"}

    f0_clean, amp_frames, _ = extract_exact_tones(y, fs)
    if len(f0_clean) == 0:
        return {"error": "No audio detected—check your microphone input!"}

    voiced_pitches = f0_clean[amp_frames > 0.12]
    avg_hz = round(float(np.median(voiced_pitches)), 1) if len(voiced_pitches) > 0 else 0.0
    avg_note = librosa.hz_to_note(avg_hz) if avg_hz > 0 else "N/A"

    return {
        "bpm": bpm,
        "duration_sec": round(len(y) / fs, 2),
        "center_note": avg_note,
        "center_hz": avg_hz,
        "pitch_timeline": [round(float(p), 1) for p in f0_clean[::4]],
        "energy_timeline": [round(float(e * 100.0), 1) for e in amp_frames[::4]],
    }


# --- STEP 3: SINGLE SYNTHESIZER RESYNTHESIS ENGINE ---
def digitize_to_synth(
    wav_path: Path,
    synth_preset: str = "NF Dark Analog Synth",
    octave_shift: int = 0,
    noise_gate: float = 0.12,
    reverb_wet: float = 0.28,
) -> np.ndarray:
    """
    Drives ONE synthesizer directly with the exact pitch and volume curve
    of the recording. Zero raw voice, zero guessed chords or drums.
    """
    y, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    n_samples = len(y)
    f0_frames, amp_frames, hop = extract_exact_tones(y, SAMPLE_RATE)

    if len(f0_frames) == 0:
        return np.zeros(n_samples, dtype=np.float32)

    pitch_mult = 2.0 ** octave_shift
    target_f0 = f0_frames * pitch_mult

    gated_amp = np.where(amp_frames >= noise_gate, amp_frames, 0.0)
    if np.max(gated_amp) > 0:
        gated_amp = gated_amp / np.max(gated_amp)

    frame_idx = np.arange(len(target_f0)) * hop
    sample_idx = np.arange(n_samples)
    freq_audio = np.interp(sample_idx, frame_idx, target_f0)
    amp_audio = np.interp(sample_idx, frame_idx, gated_amp)

    vca_coeff = np.exp(-1.0 / (0.008 * SAMPLE_RATE))
    amp_smooth = lfilter([1.0 - vca_coeff], [1.0, -vca_coeff], amp_audio)

    phase = 2.0 * np.pi * np.cumsum(freq_audio) / SAMPLE_RATE

    if synth_preset == "Pure Digitized Tone":
        osc = 0.85 * np.sin(phase) + 0.15 * np.sin(2.0 * phase)
        cutoff_hz = 3200.0

    elif synth_preset == "Deep Moog 808 Bass":
        phase_sub = 2.0 * np.pi * np.cumsum(freq_audio * 0.5) / SAMPLE_RATE
        raw_bass = (
            0.65 * np.sin(phase)
            + 0.45 * np.sin(phase_sub)
            + 0.25 * np.sin(2.0 * phase)
        )
        osc = np.tanh(raw_bass * (1.5 + 1.8 * amp_smooth))
        cutoff_hz = 480.0

    elif synth_preset == "Haunting Cinematic Lead":
        p1 = 2.0 * np.pi * np.cumsum(freq_audio * 1.004) / SAMPLE_RATE
        p2 = 2.0 * np.pi * np.cumsum(freq_audio * 0.996) / SAMPLE_RATE
        osc = (
            0.45 * np.sin(phase)
            + 0.28 * np.tanh(1.6 * np.sin(p1))
            + 0.28 * np.tanh(1.6 * np.sin(p2))
            + 0.22 * np.sin(2.0 * phase)
            + 0.10 * np.sin(3.0 * phase)
        )
        cutoff_hz = 2600.0

    else:  # "NF Dark Analog Synth"
        p1 = 2.0 * np.pi * np.cumsum(freq_audio * 1.003) / SAMPLE_RATE
        p2 = 2.0 * np.pi * np.cumsum(freq_audio * 0.997) / SAMPLE_RATE
        p_sub = 2.0 * np.pi * np.cumsum(freq_audio * 0.5) / SAMPLE_RATE
        osc = (
            0.50 * np.sin(phase)
            + 0.25 * np.sin(p_sub)
            + 0.22 * np.tanh(1.7 * np.sin(p1))
            + 0.22 * np.tanh(1.7 * np.sin(p2))
            + 0.16 * np.sin(2.0 * phase)
        )
        cutoff_hz = 1850.0

    synth_raw = osc * amp_smooth
    nyq = SAMPLE_RATE / 2.0
    b, a = butter(2, min(cutoff_hz / nyq, 0.95), btype="low")
    synth_filtered = lfilter(b, a, synth_raw)

    return add_dark_reverb(synth_filtered, SAMPLE_RATE, wet=reverb_wet)


# --- UI LAYOUT ---
st.title("🎛️ Vocal Tone Digitizer")
st.caption("Record voice → Digitize into exact synth stems → Mix them together.")

tab_record, tab_synth, tab_mixer = st.tabs(
    ["🎙️ Step 1: Record Tones", "🎹 Step 2: Save as Synth Stem", "🎚️ Step 3: Multi-Track Mixer"]
)

# ==========================================
# TAB 1: RECORD & INSPECT EXACT TONES
# ==========================================
with tab_record:
    left_col, right_col = st.columns([1.3, 1], gap="large")

    with left_col:
        st.subheader("1. Record Your Vocal Tones")

        with st.expander("🎧 Optional Metronome Click (Use Headphones)", expanded=False):
            bpm = st.slider("Tempo (BPM)", min_value=60, max_value=160, value=90, step=1)
            st.audio(make_metronome_wav(bpm), format="audio/wav", loop=True)

        if "bpm" not in locals():
            bpm = 90

        recorded_audio = st.audio_input("Click the mic to Record, click Stop when done")

        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()
            analysis = analyze_recording(audio_bytes, bpm)

            if "error" in analysis:
                st.warning(analysis["error"])
            else:
                st.success(
                    f"Captured **{analysis['duration_sec']}s** | Center Pitch: **{analysis['center_note']}** ({analysis['center_hz']} Hz)"
                )
                c_p, c_e = st.columns(2)
                with c_p:
                    st.markdown("**Exact Pitch Curve (Hz)**")
                    st.line_chart(analysis["pitch_timeline"], height=150)
                with c_e:
                    st.markdown("**Exact Volume / Dynamics Curve**")
                    st.area_chart(analysis["energy_timeline"], height=150)

                st.divider()
                st.subheader("2. Save Raw Recording")
                
                # CLEAN UX: Empty by default, clear placeholder
                take_name = st.text_input("Name your take (optional):", placeholder="e.g., funky_bassline")

                if st.button("💾 Save Take", type="primary", use_container_width=True):
                    # If user typed something, clean it. If empty, generate a short timestamp.
                    if take_name.strip():
                        clean_name = "".join(c for c in take_name if c.isalnum() or c in ("-", "_")).strip()
                    else:
                        clean_name = f"take_{datetime.now().strftime('%H%M%S')}"

                    wav_path = SAVE_DIR / f"{clean_name}.wav"
                    json_path = SAVE_DIR / f"{clean_name}.json"

                    wav_path.write_bytes(audio_bytes)
                    json_path.write_text(json.dumps(analysis, indent=2))
                    st.toast(f"Saved raw take '{clean_name}'!", icon="✅")
                    st.rerun()

    with right_col:
        st.subheader("📂 Saved Raw Takes")
        saved_wavs = sorted(SAVE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not saved_wavs:
            st.info("No saved takes yet. Record something on the left and hit **Save**!")
        else:
            for wav_file in saved_wavs:
                json_file = wav_file.with_suffix(".json")
                with st.container(border=True):
                    top_row, del_row = st.columns([4, 1])
                    top_row.markdown(f"**{wav_file.stem}**")
                    if del_row.button("🗑️", key=f"del_raw_{wav_file.stem}"):
                        wav_file.unlink(missing_ok=True)
                        json_file.unlink(missing_ok=True)
                        st.rerun()

                    st.audio(str(wav_file), format="audio/wav")

                    if json_file.exists():
                        data = json.loads(json_file.read_text())
                        with st.expander(f"Tone Curve ({data.get('center_note', 'N/A')})"):
                            st.line_chart(data.get("pitch_timeline", []), height=100)


# ==========================================
# TAB 2: DIGITIZE INTO ONE SYNTHESIZER
# ==========================================
with tab_synth:
    st.subheader("🎹 Digitize Recording Into a Synth Stem")
    st.caption("Pick a raw take, turn it into a synthesized instrument, and save it as a Stem for the Mixer.")

    available_takes = sorted([p.stem for p in SAVE_DIR.glob("*.wav")])

    if not available_takes:
        st.info("Record and save a take in Tab 1 first!")
    else:
        options = ["(None)"] + available_takes
        selected_take = st.selectbox("🎙️ Select Raw Recording", options, index=0)

        if selected_take != "(None)":
            wav_file = SAVE_DIR / f"{selected_take}.wav"

            col1, col2 = st.columns(2)
            with col1:
                synth_preset = st.selectbox(
                    "🎛️ Synthesizer Voice",
                    options=[
                        "NF Dark Analog Synth",
                        "Deep Moog 808 Bass",
                        "Haunting Cinematic Lead",
                        "Pure Digitized Tone",
                    ],
                    index=0,
                )
            with col2:
                octave_label = st.selectbox(
                    "🎼 Octave Shift",
                    options=[
                        "0 (Exact Pitch)",
                        "+1 Octave",
                        "+2 Octaves",
                        "-1 Octave",
                    ],
                    index=0,
                )
                octave_map = {
                    "0 (Exact Pitch)": 0,
                    "+1 Octave": 1,
                    "+2 Octaves": 2,
                    "-1 Octave": -1,
                }
                octave_shift = octave_map[octave_label]

            col3, col4 = st.columns(2)
            with col3:
                noise_gate = st.slider("🔇 Silence Gate", 0.02, 0.40, 0.12, 0.02)
            with col4:
                reverb_wet = st.slider("🌌 Dark Hall Space", 0.0, 0.75, 0.28, 0.05)

            if st.button("⚡ Preview Synth", type="primary", use_container_width=True):
                with st.spinner("Digitizing exact tones into synthesizer..."):
                    synth_audio = digitize_to_synth(
                        wav_file,
                        synth_preset=synth_preset,
                        octave_shift=octave_shift,
                        noise_gate=noise_gate,
                        reverb_wet=reverb_wet,
                    )
                    st.session_state["digitized_wav"] = to_wav_bytes(synth_audio, SAMPLE_RATE)

        if "digitized_wav" in st.session_state:
            st.divider()
            st.markdown("### 🎧 Synth Preview")
            st.audio(st.session_state["digitized_wav"], format="audio/wav", loop=True)

            save_c1, save_c2 = st.columns([3, 1])
            
            # CLEAN UX: Empty by default, clear placeholder
            out_name = save_c1.text_input("Name your Stem (optional):", placeholder="e.g., moog_bounce")
            
            if save_c2.button("💾 Save as Stem", use_container_width=True):
                # If user typed something, clean it. If empty, generate a smart short name based on the synth.
                if out_name.strip():
                    clean_out = "".join(c for c in out_name if c.isalnum() or c in ("-", "_")).strip()
                else:
                    short_preset = synth_preset.split()[1].lower() # e.g. 'dark', 'moog', 'cinematic', 'digitized'
                    clean_out = f"{short_preset}_{datetime.now().strftime('%H%M%S')}"
                    
                out_path = STEM_DIR / f"{clean_out}.wav"
                out_path.write_bytes(st.session_state["digitized_wav"])
                st.toast(f"Saved stem '{clean_out}.wav' to Mixer!", icon="🔥")
                st.rerun()

    st.divider()
    st.subheader("💿 Saved Synth Stems")
    saved_stems = sorted(STEM_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not saved_stems:
        st.caption("Your exported synth stems will appear here.")
    else:
        for s_file in saved_stems:
            with st.container(border=True):
                b1, b2 = st.columns([5, 1])
                b1.markdown(f"**{s_file.stem}**")
                if b2.button("🗑️", key=f"del_stem_{s_file.stem}"):
                    s_file.unlink(missing_ok=True)
                    st.rerun()
                st.audio(str(s_file), format="audio/wav")


# ==========================================
# TAB 3: MULTI-TRACK MIXER
# ==========================================
with tab_mixer:
    st.subheader("🎚️ Multi-Track Studio Mixer")
    st.caption("Combine your digitized stems. Differing track lengths are handled automatically—shorter tracks will just naturally stop playing while the rest continue.")

    available_stems = sorted([p.stem for p in STEM_DIR.glob("*.wav")])
    if not available_stems:
        st.info("No saved stems yet. Digitize and save some instruments in Tab 2 first!")
    else:
        mixer_options = ["(None)"] + available_stems
        track_selections = []
        track_volumes = []

        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                st.markdown(f"**Track {i+1}**")
                choice = st.selectbox(f"Stem {i+1}", mixer_options, index=0, label_visibility="collapsed")
                vol = st.slider(f"Volume {i+1}", 0.0, 1.5, 1.0, 0.05, label_visibility="collapsed")
                
                if choice != "(None)":
                    st.audio(str(STEM_DIR / f"{choice}.wav"), format="audio/wav")
                
                track_selections.append(choice)
                track_volumes.append(vol)

        if st.button("🎛️ Mix Full Beat", type="primary", use_container_width=True):
            loaded_audio = []
            for choice, vol in zip(track_selections, track_volumes):
                if choice != "(None)":
                    y, _ = librosa.load(str(STEM_DIR / f"{choice}.wav"), sr=SAMPLE_RATE, mono=True)
                    loaded_audio.append(y * vol)

            if not loaded_audio:
                st.warning("Select at least one track to mix!")
            else:
                with st.spinner("Summing tracks..."):
                    # Find the longest track to accommodate varying lengths automatically
                    max_len = max(len(arr) for arr in loaded_audio)
                    master_mix = np.zeros(max_len, dtype=np.float32)

                    # Add each track into the master array (zero-pads shorter tracks automatically)
                    for arr in loaded_audio:
                        master_mix[:len(arr)] += arr

                    st.session_state["master_mix"] = to_wav_bytes(master_mix, SAMPLE_RATE)

        if "master_mix" in st.session_state:
            st.divider()
            st.markdown("### 🎧 Final Mixed Master")
            st.audio(st.session_state["master_mix"], format="audio/wav", loop=True)

            save_m1, save_m2 = st.columns([3, 1])
            
            # CLEAN UX: Empty by default, clear placeholder
            master_name = save_m1.text_input("Name your final beat (optional):", placeholder="e.g., oakland_funk_master")
            
            if save_m2.button("💾 Save Final Beat WAV", use_container_width=True):
                # If user typed something, clean it. If empty, generate a short master name.
                if master_name.strip():
                    clean_master = "".join(c for c in master_name if c.isalnum() or c in ("-", "_")).strip()
                else:
                    clean_master = f"master_{datetime.now().strftime('%H%M%S')}"
                    
                out_path = BEAT_DIR / f"{clean_master}.wav"
                out_path.write_bytes(st.session_state["master_mix"])
                st.toast(f"Saved '{clean_master}.wav' to generated_beats/!", icon="🔥")
                st.rerun()
