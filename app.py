import io
import json
import re
import shutil
import subprocess
import tempfile
import uuid
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
SUNO_DIR = Path("suno_beats")
BEAT_DIR = Path("generated_beats")
VIDEO_CLIPS_DIR = Path("uploaded_clips")
VIDEO_OUT_DIR = Path("generated_videos")

SAVE_DIR.mkdir(exist_ok=True)
STEM_DIR.mkdir(exist_ok=True)
SUNO_DIR.mkdir(exist_ok=True)
BEAT_DIR.mkdir(exist_ok=True)
VIDEO_CLIPS_DIR.mkdir(exist_ok=True)
VIDEO_OUT_DIR.mkdir(exist_ok=True)
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


def polish_vocals(y: np.ndarray, fs: int) -> np.ndarray:
    """
    Studio vocal chain for raw mic takes:
    1. Cuts low-end mic rumble (< 85 Hz)
    2. Gentle compression to balance quiet verses and loud punchlines
    3. Subtle room glue so vocals sit naturally inside a mastered beat
    """
    if len(y) == 0:
        return y

    b, a = butter(2, 85.0 / (fs / 2.0), btype="high")
    y_clean = lfilter(b, a, y)

    peak = float(np.max(np.abs(y_clean)))
    if peak < 1e-6:
        return y_clean
    y_norm = y_clean / peak

    abs_y = np.abs(y_norm)
    coeff = np.exp(-1.0 / (0.035 * fs))
    env = lfilter([1.0 - coeff], [1.0, -coeff], abs_y)

    threshold = 0.22
    ratio = 3.0
    gain = np.ones_like(env)
    above = env > threshold
    compressed_env = threshold + (env[above] - threshold) / ratio
    gain[above] = compressed_env / (env[above] + 1e-6)

    y_comp = np.tanh((y_norm * gain) * 1.8) * 0.85
    return add_dark_reverb(y_comp, fs, wet=0.14)


def apply_offset(y: np.ndarray, offset_sec: float, fs: int) -> np.ndarray:
    """Shifts track forward (+ seconds delay) or trims start (- seconds)."""
    shift_samples = int(round(offset_sec * fs))
    if shift_samples > 0:
        return np.concatenate([np.zeros(shift_samples, dtype=np.float32), y])
    elif shift_samples < 0:
        trim = abs(shift_samples)
        if trim >= len(y):
            return np.zeros(1, dtype=np.float32)
        return y[trim:]
    return y


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


# --- STEP 4: VIDEO & AUDIO MERGE HELPERS ---
def get_ffmpeg_bin() -> str | None:
    """Finds ffmpeg on system PATH or via imageio-ffmpeg if installed."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def probe_media_info(ffmpeg_bin: str, media_path: Path) -> dict:
    """Extracts duration (sec) and video dimensions (width, height) via ffmpeg."""
    cmd = [ffmpeg_bin, "-i", str(media_path)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = proc.stderr

    dur_sec = 0.0
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if dur_match:
        dur_sec = (
            int(dur_match.group(1)) * 3600
            + int(dur_match.group(2)) * 60
            + float(dur_match.group(3))
        )

    width, height = 1280, 720
    res_match = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", stderr)
    if res_match:
        w = int(res_match.group(1))
        h = int(res_match.group(2))
        width = w - (w % 2)
        height = h - (h % 2)

    return {"duration": round(dur_sec, 2), "width": width, "height": height}


def build_music_video(
    ffmpeg_bin: str,
    sequence: list[dict],
    audio_path: Path,
    out_path: Path,
    match_audio_length: bool = True,
) -> tuple[bool, str]:
    """
    1. Strips ALL original sound (-an) from every AI video clip.
    2. Normalizes resolution, framerate, and pixel format across clips.
    3. Concatenates clips in Flow sequence order (respecting each clip's loop count).
    4. Merges the silent video sequence with the selected music track.
    """
    if not sequence:
        return False, "No video clips in the sequence."

    first_clip_path = VIDEO_CLIPS_DIR / sequence[0]["filename"]
    if not first_clip_path.exists():
        return False, f"Missing clip file: {sequence[0]['filename']}"

    base_info = probe_media_info(ffmpeg_bin, first_clip_path)
    target_w = base_info["width"] or 1280
    target_h = base_info["height"] or 720
    audio_info = probe_media_info(ffmpeg_bin, audio_path)
    audio_dur = audio_info["duration"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        normalized_map: dict[str, Path] = {}
        clip_durations: dict[str, float] = {}

        # 1. Strip audio (-an) and normalize each unique clip once
        unique_names = list(dict.fromkeys(item["filename"] for item in sequence))
        for idx, fname in enumerate(unique_names):
            src_path = VIDEO_CLIPS_DIR / fname
            if not src_path.exists():
                return False, f"Clip not found: {fname}"

            info = probe_media_info(ffmpeg_bin, src_path)
            clip_durations[fname] = max(info["duration"], 0.5)

            norm_path = tmp / f"norm_{idx}.mp4"
            vf = (
                f"fps=30,scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,format=yuv420p"
            )
            cmd_norm = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(src_path),
                "-an",  # Delete all original AI video sound
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                str(norm_path),
            ]
            res = subprocess.run(cmd_norm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                return False, f"Error stripping audio/normalizing '{fname}': {res.stderr[-400:]}"
            normalized_map[fname] = norm_path

        # 2. Build ordered concat playlist with loops
        playlist_entries: list[Path] = []
        total_video_dur = 0.0
        for item in sequence:
            fname = item["filename"]
            loops = max(1, int(item.get("loops", 1)))
            for _ in range(loops):
                playlist_entries.append(normalized_map[fname])
                total_video_dur += clip_durations[fname]

        # If matching audio duration and video is shorter than audio, repeat last scene clip
        if match_audio_length and audio_dur > 0 and total_video_dur < audio_dur:
            last_fname = sequence[-1]["filename"]
            last_dur = clip_durations[last_fname]
            while total_video_dur < audio_dur:
                playlist_entries.append(normalized_map[last_fname])
                total_video_dur += last_dur

        concat_txt = tmp / "concat_list.txt"
        lines = [
            f"file '{p.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
            for p in playlist_entries
        ]
        concat_txt.write_text("\n".join(lines), encoding="utf-8")

        # 3. Concatenate silent video and merge with music track
        cmd_merge = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ]
        if match_audio_length and audio_dur > 0:
            cmd_merge.append("-shortest")
        cmd_merge.append(str(out_path))

        res_merge = subprocess.run(
            cmd_merge, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if res_merge.returncode != 0:
            return False, f"Error merging video and audio: {res_merge.stderr[-400:]}"

    return True, "Music video built successfully!"


# --- UI LAYOUT ---
st.title("🎛️ Vocal Tone Digitizer")
st.caption("Record voice → Digitize into exact synth stems → Mix beats → Assemble silent AI videos with music.")

tab_record, tab_synth, tab_mixer, tab_video = st.tabs(
    [
        "🎙️ Step 1: Record Tones",
        "🎹 Step 2: Save as Synth Stem",
        "🎚️ Step 3: Multi-Track Mixer",
        "🎬 Step 4: Music Video Studio",
    ]
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

                take_name = st.text_input(
                    "Name your take (optional):", placeholder="e.g., funky_bassline or verse_vocals"
                )

                if st.button("💾 Save Take", type="primary", use_container_width=True):
                    if take_name.strip():
                        clean_name = "".join(
                            c for c in take_name if c.isalnum() or c in ("-", "_")
                        ).strip()
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
    st.caption(
        "Pick a raw take, turn it into a synthesized instrument, and save it as a Stem for the Mixer."
    )

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

            out_name = save_c1.text_input(
                "Name your Stem (optional):", placeholder="e.g., moog_bounce"
            )

            if save_c2.button("💾 Save as Stem", use_container_width=True):
                if out_name.strip():
                    clean_out = "".join(
                        c for c in out_name if c.isalnum() or c in ("-", "_")
                    ).strip()
                else:
                    short_preset = synth_preset.split()[1].lower()
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
# TAB 3: HYBRID MULTI-TRACK MIXER
# ==========================================
with tab_mixer:
    st.subheader("🎚️ Hybrid Studio Mixer (Suno Beats + Synth Stems + Raw Vocals)")
    st.caption(
        "1) Mix a Synth Stem + Raw Vocal to export a Guide Track for Suno.  "
        "2) Import Suno's instrumental below and align your polished raw vocals over it."
    )

    # --- UPLOAD SUNO BEAT ---
    with st.container(border=True):
        up_col1, up_col2 = st.columns([2, 1])
        with up_col1:
            uploaded_suno = st.file_uploader(
                "📥 Import Suno Beat (.mp3 or .wav)",
                type=["mp3", "wav"],
                help="Upload the instrumental track Suno generated around your guide.",
            )
            if uploaded_suno is not None:
                dest_path = SUNO_DIR / uploaded_suno.name
                if not dest_path.exists() or dest_path.stat().st_size != uploaded_suno.size:
                    dest_path.write_bytes(uploaded_suno.getvalue())
                    st.toast(f"Imported '{uploaded_suno.name}' into Mixer!", icon="📥")

        with up_col2:
            suno_files = sorted(
                [p for p in SUNO_DIR.iterdir() if p.suffix.lower() in (".mp3", ".wav")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            st.markdown(f"**Imported Suno Beats ({len(suno_files)})**")
            if suno_files:
                del_target = st.selectbox(
                    "Manage imports",
                    ["(Select to delete)"] + [p.name for p in suno_files],
                    label_visibility="collapsed",
                )
                if del_target != "(Select to delete)":
                    if st.button("🗑️ Delete Selected Import", use_container_width=True):
                        (SUNO_DIR / del_target).unlink(missing_ok=True)
                        st.rerun()
            else:
                st.caption("No Suno beats imported yet.")

    # Build unified source list across Suno Beats, Synth Stems, and Raw Vocals
    suno_options = [f"[Suno] {p.name}" for p in suno_files]
    stem_options = [f"[Stem] {p.stem}" for p in sorted(STEM_DIR.glob("*.wav"))]
    vocal_options = [f"[Vocal] {p.stem}" for p in sorted(SAVE_DIR.glob("*.wav"))]

    mixer_options = ["(None)"] + suno_options + stem_options + vocal_options

    if len(mixer_options) == 1:
        st.info("Record a take in Tab 1 or import a Suno beat above to start mixing!")
    else:
        st.divider()
        track_configs = []

        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                with st.container(border=True):
                    st.markdown(f"**Track {i+1}**")
                    choice = st.selectbox(
                        f"Source {i+1}",
                        mixer_options,
                        index=0,
                        key=f"mix_src_{i}",
                        label_visibility="collapsed",
                    )

                    vol = st.slider(
                        "Volume",
                        min_value=0.0,
                        max_value=1.5,
                        value=1.0,
                        step=0.05,
                        key=f"mix_vol_{i}",
                    )

                    nudge_sec = st.number_input(
                        "Timing Nudge (sec)",
                        min_value=-15.0,
                        max_value=30.0,
                        value=0.0,
                        step=0.05,
                        format="%.2f",
                        key=f"mix_nudge_{i}",
                        help="+ delays start (e.g. wait for Suno intro); - trims dead air at start",
                    )

                    is_vocal = choice.startswith("[Vocal]")
                    polish = st.checkbox(
                        "✨ Vocal Polish",
                        value=is_vocal,
                        key=f"mix_pol_{i}_{choice}",
                        help="Cuts low rumble, evens out volume, and adds subtle room reverb",
                    )

                    if choice != "(None)":
                        if choice.startswith("[Suno] "):
                            preview_path = SUNO_DIR / choice.removeprefix("[Suno] ")
                        elif choice.startswith("[Stem] "):
                            preview_path = STEM_DIR / f"{choice.removeprefix('[Stem] ')}.wav"
                        else:
                            preview_path = SAVE_DIR / f"{choice.removeprefix('[Vocal] ')}.wav"

                        if preview_path.exists():
                            st.audio(str(preview_path))

                    track_configs.append(
                        {
                            "choice": choice,
                            "vol": vol,
                            "nudge": nudge_sec,
                            "polish": polish,
                        }
                    )

        if st.button("🎛️ Mix Full Track", type="primary", use_container_width=True):
            loaded_tracks = []
            for cfg in track_configs:
                choice = cfg["choice"]
                if choice == "(None)":
                    continue

                if choice.startswith("[Suno] "):
                    file_path = SUNO_DIR / choice.removeprefix("[Suno] ")
                elif choice.startswith("[Stem] "):
                    file_path = STEM_DIR / f"{choice.removeprefix('[Stem] ')}.wav"
                else:
                    file_path = SAVE_DIR / f"{choice.removeprefix('[Vocal] ')}.wav"

                if file_path.exists():
                    y, _ = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
                    if cfg["polish"]:
                        y = polish_vocals(y, SAMPLE_RATE)
                    y = apply_offset(y, cfg["nudge"], SAMPLE_RATE)
                    loaded_tracks.append(y * cfg["vol"])

            if not loaded_tracks:
                st.warning("Select at least one track to mix!")
            else:
                with st.spinner("Applying vocal polish, aligning timing, and summing tracks..."):
                    max_len = max(len(arr) for arr in loaded_tracks)
                    master_mix = np.zeros(max_len, dtype=np.float32)

                    for arr in loaded_tracks:
                        master_mix[: len(arr)] += arr

                    st.session_state["master_mix"] = to_wav_bytes(master_mix, SAMPLE_RATE)

        if "master_mix" in st.session_state:
            st.divider()
            st.markdown("### 🎧 Mixed Output")
            st.audio(st.session_state["master_mix"], format="audio/wav", loop=True)

            save_m1, save_m2 = st.columns([3, 1])

            master_name = save_m1.text_input(
                "Name your export (optional):",
                placeholder="e.g., suno_guide_mix or final_song",
            )

            if save_m2.button("💾 Save Export WAV", use_container_width=True):
                if master_name.strip():
                    clean_master = "".join(
                        c for c in master_name if c.isalnum() or c in ("-", "_")
                    ).strip()
                else:
                    clean_master = f"mix_{datetime.now().strftime('%H%M%S')}"

                out_path = BEAT_DIR / f"{clean_master}.wav"
                out_path.write_bytes(st.session_state["master_mix"])
                st.toast(f"Saved '{clean_master}.wav' to generated_beats/!", icon="🔥")
                st.rerun()

    # Show previously exported mixes
    st.divider()
    st.subheader("💿 Saved Mixes & Final Songs")
    saved_beats = sorted(BEAT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not saved_beats:
        st.caption("Your exported guide tracks and finished songs will appear here.")
    else:
        for b_file in saved_beats:
            with st.container(border=True):
                b1, b2 = st.columns([5, 1])
                b1.markdown(f"**{b_file.stem}**")
                if b2.button("🗑️", key=f"del_beat_{b_file.stem}"):
                    b_file.unlink(missing_ok=True)
                    st.rerun()
                st.audio(str(b_file), format="audio/wav")


# ==========================================
# TAB 4: AI VIDEO MUTE & MUSIC MERGER (FLOW-STYLE)
# ==========================================
with tab_video:
    st.subheader("🎬 Step 4: AI Video Sequencer & Music Merger")
    st.caption(
        "Upload your AI video clips in order (like Google Flow) → Automatically delete all AI video sound → "
        "Loop scenes & merge with your music track into a YouTube-ready MP4."
    )

    ffmpeg_bin = get_ffmpeg_bin()
    if not ffmpeg_bin:
        st.error(
            "⚠️ **FFmpeg is not installed on this machine yet.** "
            "Run `pip install imageio-ffmpeg` in your terminal and refresh this page to enable video processing."
        )

    if "video_sequence" not in st.session_state:
        st.session_state["video_sequence"] = []
    if "processed_video_uploads" not in st.session_state:
        st.session_state["processed_video_uploads"] = set()

    # --- 1. UPLOAD CLIPS IN ORDER ---
    with st.container(border=True):
        st.markdown("### 1. Upload AI Video Clips (Adds Next in Order & Strips Sound)")
        uploaded_videos = st.file_uploader(
            "Upload one or more AI video clips (.mp4, .mov, .webm, .mkv) — each new upload appends to the end of your sequence",
            type=["mp4", "mov", "webm", "mkv"],
            accept_multiple_files=True,
            key="flow_video_uploader",
        )

        if uploaded_videos:
            added_any = False
            for up_vid in uploaded_videos:
                sig = (up_vid.name, up_vid.size)
                dest_clip = VIDEO_CLIPS_DIR / up_vid.name
                if not dest_clip.exists() or dest_clip.stat().st_size != up_vid.size:
                    dest_clip.write_bytes(up_vid.getvalue())

                if sig not in st.session_state["processed_video_uploads"]:
                    st.session_state["processed_video_uploads"].add(sig)
                    st.session_state["video_sequence"].append(
                        {
                            "id": str(uuid.uuid4()),
                            "filename": up_vid.name,
                            "loops": 1,
                        }
                    )
                    added_any = True

            if added_any:
                st.toast("Added uploaded clip(s) to sequence (sound will be stripped)!", icon="🎬")
                st.rerun()

        # Allow reusing already-uploaded clips (e.g., repeating Chorus clips)
        available_clips = sorted(
            [
                p.name
                for p in VIDEO_CLIPS_DIR.iterdir()
                if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")
            ]
        )
        if available_clips:
            reuse_c1, reuse_c2, reuse_c3 = st.columns([3, 1.5, 1.5])
            clip_to_append = reuse_c1.selectbox(
                "Add another saved clip to the end of the sequence (e.g. reuse Chorus clip):",
                available_clips,
                key="reuse_clip_select",
            )
            if reuse_c2.button("➕ Add Clip Next in Order", use_container_width=True):
                st.session_state["video_sequence"].append(
                    {
                        "id": str(uuid.uuid4()),
                        "filename": clip_to_append,
                        "loops": 1,
                    }
                )
                st.rerun()

            if reuse_c3.button("🧹 Clear Sequence", use_container_width=True):
                st.session_state["video_sequence"] = []
                st.rerun()

    # --- 2. FLOW-STYLE SEQUENCE TIMELINE ---
    st.divider()
    st.markdown("### 2. Scene Sequence Order (Original AI Audio Deleted Automatically)")

    seq = st.session_state["video_sequence"]
    total_seq_sec = 0.0

    if not seq:
        st.info(
            "Your sequence is empty. Upload your first AI video clip above, then upload the next one to build your timeline!"
        )
    else:
        for idx, item in enumerate(list(seq)):
            clip_path = VIDEO_CLIPS_DIR / item["filename"]
            clip_dur = 8.0
            if ffmpeg_bin and clip_path.exists():
                clip_dur = probe_media_info(ffmpeg_bin, clip_path)["duration"] or 8.0

            seg_total = round(clip_dur * item["loops"], 1)
            total_seq_sec += seg_total

            with st.container(border=True):
                c_info, c_loop, c_order = st.columns([3.2, 1.3, 2.5])

                with c_info:
                    st.markdown(
                        f"**Scene {idx + 1}:** `{item['filename']}`  \n"
                        f"🔇 *AI sound removed* • Clip: **{clip_dur:.1f}s** × **{item['loops']}x** = **{seg_total:.1f}s**"
                    )
                    with st.expander("Preview raw clip"):
                        if clip_path.exists():
                            st.video(str(clip_path))

                with c_loop:
                    new_loops = st.number_input(
                        "Loop Count",
                        min_value=1,
                        max_value=30,
                        value=int(item["loops"]),
                        step=1,
                        key=f"loop_{item['id']}",
                        help="Repeat this clip back-to-back before moving to the next scene",
                    )
                    if new_loops != item["loops"]:
                        st.session_state["video_sequence"][idx]["loops"] = int(new_loops)
                        st.rerun()

                with c_order:
                    st.markdown("**Order**")
                    b_up, b_dn, b_dup, b_del = st.columns(4)
                    if b_up.button("⬆️", key=f"up_{item['id']}", disabled=(idx == 0), help="Move earlier"):
                        seq[idx - 1], seq[idx] = seq[idx], seq[idx - 1]
                        st.rerun()
                    if b_dn.button(
                        "⬇️", key=f"dn_{item['id']}", disabled=(idx == len(seq) - 1), help="Move later"
                    ):
                        seq[idx + 1], seq[idx] = seq[idx], seq[idx + 1]
                        st.rerun()
                    if b_dup.button("📋", key=f"dup_{item['id']}", help="Duplicate scene"):
                        seq.insert(
                            idx + 1,
                            {
                                "id": str(uuid.uuid4()),
                                "filename": item["filename"],
                                "loops": item["loops"],
                            },
                        )
                        st.rerun()
                    if b_del.button("🗑️", key=f"del_seq_{item['id']}", help="Remove from sequence"):
                        seq.pop(idx)
                        st.rerun()

        st.info(
            f"🎞️ **Total Silent Video Sequence Length:** **{total_seq_sec:.1f} seconds** "
            f"({int(total_seq_sec // 60)}m {int(total_seq_sec % 60):02d}s) across **{len(seq)}** scene slot(s)."
        )

    # --- 3. MUSIC TRACK SELECTION & MERGE ---
    st.divider()
    st.markdown("### 3. Upload or Pick Music Track & Merge")

    with st.container(border=True):
        m_col1, m_col2 = st.columns([1.2, 1])

        with m_col1:
            uploaded_song = st.file_uploader(
                "🎵 Upload Your Music Track (.wav, .mp3, .m4a, .ogg, .flac)",
                type=["wav", "mp3", "m4a", "ogg", "flac"],
                key="video_music_uploader",
            )
            if uploaded_song is not None:
                song_dest = BEAT_DIR / uploaded_song.name
                if not song_dest.exists() or song_dest.stat().st_size != uploaded_song.size:
                    song_dest.write_bytes(uploaded_song.getvalue())
                    st.toast(f"Loaded music track '{uploaded_song.name}'!", icon="🎵")

        with m_col2:
            all_songs = sorted(
                [
                    p
                    for p in list(BEAT_DIR.iterdir()) + list(SUNO_DIR.iterdir()) + list(SAVE_DIR.iterdir())
                    if p.suffix.lower() in (".wav", ".mp3", ".m4a", ".ogg", ".flac")
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            song_labels = [f"{p.parent.name}/{p.name}" for p in all_songs]
            default_idx = 0
            if uploaded_song is not None:
                target_lbl = f"{BEAT_DIR.name}/{uploaded_song.name}"
                if target_lbl in song_labels:
                    default_idx = song_labels.index(target_lbl)

            selected_song_label = st.selectbox(
                "Or select any saved song/mix from your machine:",
                ["(None)"] + song_labels,
                index=(default_idx + 1) if song_labels else 0,
                key="video_song_select",
            )

        chosen_audio_path: Path | None = None
        if selected_song_label != "(None)":
            chosen_audio_path = Path(selected_song_label)
            if chosen_audio_path.exists():
                st.audio(str(chosen_audio_path))
                if ffmpeg_bin:
                    song_dur = probe_media_info(ffmpeg_bin, chosen_audio_path)["duration"]
                    st.caption(
                        f"🎵 Selected Music Track Duration: **{song_dur:.1f}s** "
                        f"({int(song_dur // 60)}m {int(song_dur % 60):02d}s)"
                    )

        opt_col1, opt_col2 = st.columns([1.5, 1.5])
        match_audio = opt_col1.checkbox(
            "Auto-fit video length to match music track (loops final scene if short, trims at song end)",
            value=True,
        )
        video_out_name = opt_col2.text_input(
            "Output Video Filename (optional):",
            placeholder="e.g., to_be_me_music_video",
        )

        if st.button(
            "🎬 Strip Video Sound & Merge With Music",
            type="primary",
            use_container_width=True,
            disabled=(not ffmpeg_bin or not seq or chosen_audio_path is None),
        ):
            if video_out_name.strip():
                clean_vname = "".join(
                    c for c in video_out_name if c.isalnum() or c in ("-", "_")
                ).strip()
            else:
                clean_vname = f"music_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            final_mp4_path = VIDEO_OUT_DIR / f"{clean_vname}.mp4"

            with st.spinner(
                "Deleting original AI video audio, stitching sequence in order, and merging your music track..."
            ):
                ok, msg = build_music_video(
                    ffmpeg_bin=ffmpeg_bin,
                    sequence=seq,
                    audio_path=chosen_audio_path,
                    out_path=final_mp4_path,
                    match_audio_length=match_audio,
                )

            if ok:
                st.session_state["latest_music_video"] = str(final_mp4_path)
                st.toast("Music video ready!", icon="🔥")
                st.rerun()
            else:
                st.error(msg)

    # --- 4. PREVIEW & DOWNLOAD FINISHED VIDEOS ---
    if "latest_music_video" in st.session_state:
        latest_path = Path(st.session_state["latest_music_video"])
        if latest_path.exists():
            st.divider()
            st.markdown("### 🎉 Latest Rendered Music Video")
            st.video(str(latest_path))
            st.download_button(
                label=f"⬇️ Download {latest_path.name}",
                data=latest_path.read_bytes(),
                file_name=latest_path.name,
                mime="video/mp4",
                type="primary",
                use_container_width=True,
            )

    st.divider()
    st.subheader("🎥 Saved Music Videos")
    saved_vids = sorted(
        VIDEO_OUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not saved_vids:
        st.caption("Your finished music videos will appear here.")
    else:
        for v_file in saved_vids:
            with st.container(border=True):
                v1, v2, v3 = st.columns([4, 1.5, 0.8])
                v1.markdown(f"**{v_file.name}**")
                v2.download_button(
                    "⬇️ Download",
                    data=v_file.read_bytes(),
                    file_name=v_file.name,
                    mime="video/mp4",
                    key=f"dl_vid_{v_file.name}",
                    use_container_width=True,
                )
                if v3.button("🗑️", key=f"del_vid_{v_file.name}"):
                    v_file.unlink(missing_ok=True)
                    st.rerun()
                st.video(str(v_file))
