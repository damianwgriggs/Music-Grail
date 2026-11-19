import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

# --- SETUP ---
st.set_page_config(page_title="Shorts Maker Studio", page_icon="📱", layout="wide")

VIDEO_OUT_DIR = Path("generated_videos")
SHORTS_DIR = Path("generated_shorts")

VIDEO_OUT_DIR.mkdir(exist_ok=True)
SHORTS_DIR.mkdir(exist_ok=True)


# --- FFMPEG & MEDIA HELPERS ---
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
    cmd = [ffmpeg_bin, "-i", str(media_path.resolve())]
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


def parse_timestamp(ts_str: str) -> float | None:
    """Converts 'MM:SS', 'HH:MM:SS', or raw seconds into float seconds."""
    ts_str = ts_str.strip()
    if not ts_str:
        return None
    try:
        parts = [float(p) for p in ts_str.split(":")]
        if len(parts) == 1:
            return max(0.0, parts[0])
        elif len(parts) == 2:
            return max(0.0, parts[0] * 60.0 + parts[1])
        elif len(parts) == 3:
            return max(0.0, parts[0] * 3600.0 + parts[1] * 60.0 + parts[2])
    except ValueError:
        return None
    return None


def format_srt_time(seconds: float) -> str:
    """Formats float seconds into SRT timestamp HH:MM:SS,mmm."""
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def build_srt_from_lyrics(
    lyrics_text: str, srt_out_path: Path, clip_start_sec: float, clip_dur: float
) -> bool:
    """
    Converts pasted lyrics into an SRT subtitle file.
    Supports lines prefixed with [MM:SS] (relative to the full video) OR plain lines
    (auto-distributed across the clip duration).
    """
    raw_lines = [line.strip() for line in lyrics_text.splitlines() if line.strip()]
    if not raw_lines:
        return False

    timed_entries: list[tuple[float, str]] = []
    plain_entries: list[str] = []

    for line in raw_lines:
        m = re.match(r"^\[(\d{1,2}:\d{2}(?:\.\d+)?)\]\s*(.+)$", line)
        if m:
            abs_sec = parse_timestamp(m.group(1))
            if abs_sec is not None:
                rel_sec = max(0.0, abs_sec - clip_start_sec)
                timed_entries.append((rel_sec, m.group(2).strip()))
        else:
            plain_entries.append(line)

    srt_blocks = []

    if timed_entries:
        timed_entries.sort(key=lambda x: x[0])
        for idx, (start_t, text) in enumerate(timed_entries):
            if start_t >= clip_dur:
                continue
            if idx + 1 < len(timed_entries):
                end_t = min(clip_dur, timed_entries[idx + 1][0])
            else:
                end_t = clip_dur
            if end_t <= start_t:
                end_t = min(clip_dur, start_t + 2.5)
            srt_blocks.append(
                f"{idx + 1}\n{format_srt_time(start_t)} --> {format_srt_time(end_t)}\n{text}\n"
            )
    else:
        # Auto-space plain lines proportionally by character count
        total_chars = sum(max(len(line), 1) for line in plain_entries)
        cursor = 0.0
        for idx, text in enumerate(plain_entries):
            seg_dur = (max(len(text), 1) / total_chars) * clip_dur
            start_t = cursor
            end_t = min(clip_dur, cursor + seg_dur)
            cursor = end_t
            srt_blocks.append(
                f"{idx + 1}\n{format_srt_time(start_t)} --> {format_srt_time(end_t)}\n{text}\n"
            )

    if not srt_blocks:
        return False

    srt_out_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    return True


def build_vertical_short(
    ffmpeg_bin: str,
    src_video: Path,
    out_path: Path,
    start_sec: float,
    end_sec: float,
    framing_mode: str = "center_crop",
    lyrics_text: str = "",
) -> tuple[bool, str]:
    """Slices a timestamp range, converts to 1080x1920 vertical 9:16, and optionally burns pasted lyrics."""
    clip_dur = end_sec - start_sec
    if clip_dur <= 0.5:
        return False, "End time must be at least 0.5 seconds after start time."

    abs_src_video = str(src_video.resolve())
    abs_out_path = str(out_path.resolve())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        srt_file = tmp / "captions.srt"
        has_srt = build_srt_from_lyrics(lyrics_text, srt_file, start_sec, clip_dur)

        sub_filter = ""
        if has_srt:
            sub_filter = (
                ",subtitles=captions.srt:force_style="
                "'FontName=Arial,FontSize=14,Bold=1,PrimaryColour=&H00FFFFFF,"
                "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=65'"
            )

        if framing_mode == "blur_pad":
            filter_complex = (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,boxblur=25:5[bg];"
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p{sub_filter}[outv]"
            )
        else:
            filter_complex = (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"crop=1080:1920,setsar=1,format=yuv420p{sub_filter}[outv]"
            )

        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss",
            str(start_sec),
            "-i",
            abs_src_video,
            "-t",
            str(clip_dur),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            abs_out_path,
        ]

        res = subprocess.run(cmd, cwd=tmpdir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            return False, f"FFmpeg Short render failed: {res.stderr[-400:]}"

    return True, "Vertical Short created successfully!"


# --- UI LAYOUT ---
st.title("📱 Automated 9:16 Shorts Maker")
st.caption(
    "Slice highlights from your music videos or comedy episodes, reframe to 1080x1920 vertical video, "
    "and paste lyrics or dialogue to burn in mobile captions."
)

ffmpeg_bin = get_ffmpeg_bin()
if not ffmpeg_bin:
    st.error(
        "⚠️ **FFmpeg is not installed on this machine yet.** "
        "Run `pip install imageio-ffmpeg` in your terminal and refresh this page."
    )

# --- 1. SELECT OR UPLOAD SOURCE VIDEO ---
with st.container(border=True):
    st.markdown("### 1. Choose Source Video")
    s_col1, s_col2 = st.columns([1.2, 1])

    with s_col1:
        uploaded_for_short = st.file_uploader(
            "Upload a Video to Clip (.mp4, .mov, .webm, .mkv)",
            type=["mp4", "mov", "webm", "mkv"],
            key="shorts_video_uploader",
        )
        if uploaded_for_short is not None:
            dest_short_src = VIDEO_OUT_DIR / uploaded_for_short.name
            if not dest_short_src.exists() or dest_short_src.stat().st_size != uploaded_for_short.size:
                dest_short_src.write_bytes(uploaded_for_short.getvalue())
                st.toast(f"Saved '{uploaded_for_short.name}' into generated_videos/!", icon="📥")

    with s_col2:
        video_sources = sorted(
            [
                p
                for p in VIDEO_OUT_DIR.iterdir()
                if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        source_names = [p.name for p in video_sources]
        default_s_idx = 0
        if uploaded_for_short is not None and uploaded_for_short.name in source_names:
            default_s_idx = source_names.index(uploaded_for_short.name)

        selected_short_src = st.selectbox(
            "Or pick any video already in generated_videos/:",
            ["(None)"] + source_names,
            index=(default_s_idx + 1) if source_names else 0,
            key="shorts_src_select",
        )

    src_video_path: Path | None = None
    if selected_short_src != "(None)":
        src_video_path = VIDEO_OUT_DIR / selected_short_src
        if src_video_path.exists() and ffmpeg_bin:
            info = probe_media_info(ffmpeg_bin, src_video_path)
            total_vid_dur = info["duration"]
            st.caption(
                f"🎞️ Source Length: **{total_vid_dur:.1f}s** ({int(total_vid_dur // 60)}:{int(total_vid_dur % 60):02d}) "
                f"| Original Resolution: {info['width']}x{info['height']}"
            )

# --- 2. SLICE TIMESTAMPS, FRAMING & LYRICS ---
st.divider()
st.markdown("### 2. Clip Range, Framing & Lyrics")

t_col1, t_col2, t_col3 = st.columns([1, 1, 1.5])
with t_col1:
    start_str = st.text_input("Start Time (MM:SS)", value="00:00", placeholder="e.g., 00:00")
with t_col2:
    end_str = st.text_input("End Time (MM:SS)", value="00:34", placeholder="e.g., 00:34")
with t_col3:
    framing_choice = st.selectbox(
        "Vertical Framing Style",
        options=[
            "Center Crop (Full-Screen 9:16 — Best for Solo Scenes)",
            "Widescreen + Blurred Background (Best for Wide / Group Scenes)",
        ],
        index=1,
    )

lyrics_input = st.text_area(
    "💬 Paste Lyrics or Dialogue for Captions (Optional — leave blank for no captions)",
    height=170,
    placeholder=(
        "Paste lines with timestamps like:\n"
        "[00:02] Yeah, I'm blind, got that visual disability\n"
        "[00:05] Let me walk you through what it's like to be me\n\n"
        "Or paste plain lines without timestamps to auto-space them across the clip."
    ),
)

short_out_name = st.text_input(
    "Output Short Filename (optional):",
    placeholder="e.g., to_be_me_intro_short",
)

start_sec = parse_timestamp(start_str)
end_sec = parse_timestamp(end_str)

if start_sec is not None and end_sec is not None and end_sec > start_sec:
    dur_preview = end_sec - start_sec
    if dur_preview > 60:
        st.warning(f"Clip is {dur_preview:.1f}s long. Keep YouTube Shorts under 60 seconds.")
    else:
        cap_status = "with burned-in captions" if lyrics_input.strip() else "without captions"
        st.info(f"Ready to slice **{dur_preview:.1f} seconds** (`{start_str}` to `{end_str}`) {cap_status}.")

if st.button(
    "📱 Render Vertical Short (1080x1920)",
    type="primary",
    use_container_width=True,
    disabled=(
        not ffmpeg_bin
        or src_video_path is None
        or start_sec is None
        or end_sec is None
        or end_sec <= start_sec
    ),
):
    if short_out_name.strip():
        clean_sname = "".join(c for c in short_out_name if c.isalnum() or c in ("-", "_")).strip()
    else:
        clean_sname = f"short_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    short_mp4_path = SHORTS_DIR / f"{clean_sname}.mp4"
    mode_key = "blur_pad" if framing_choice.startswith("Widescreen") else "center_crop"

    with st.spinner("Slicing clip, reframing to 1080x1920 vertical, and rendering..."):
        ok_s, msg_s = build_vertical_short(
            ffmpeg_bin=ffmpeg_bin,
            src_video=src_video_path,
            out_path=short_mp4_path,
            start_sec=start_sec,
            end_sec=end_sec,
            framing_mode=mode_key,
            lyrics_text=lyrics_input,
        )

    if ok_s:
        st.session_state["latest_short"] = str(short_mp4_path)
        st.toast("Vertical Short ready!", icon="🔥")
        st.rerun()
    else:
        st.error(msg_s)

# --- 3. PREVIEW & SAVED SHORTS ---
if "latest_short" in st.session_state:
    latest_s_path = Path(st.session_state["latest_short"])
    if latest_s_path.exists():
        st.divider()
        st.markdown("### 🎉 Latest Rendered Short")
        st.video(str(latest_s_path))
        st.download_button(
            label=f"⬇️ Download {latest_s_path.name}",
            data=latest_s_path.read_bytes(),
            file_name=latest_s_path.name,
            mime="video/mp4",
            type="primary",
            use_container_width=True,
        )

st.divider()
st.subheader("📲 Saved Vertical Shorts")
saved_shorts = sorted(SHORTS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
if not saved_shorts:
    st.caption("Your exported vertical Shorts will appear here.")
else:
    for s_vid in saved_shorts:
        with st.container(border=True):
            sv1, sv2, sv3 = st.columns([4, 1.5, 0.8])
            sv1.markdown(f"**{s_vid.name}**")
            sv2.download_button(
                "⬇️ Download",
                data=s_vid.read_bytes(),
                file_name=s_vid.name,
                mime="video/mp4",
                key=f"dl_short_{s_vid.name}",
                use_container_width=True,
            )
            if sv3.button("🗑️ Delete", key=f"del_short_{s_vid.name}"):
                s_vid.unlink(missing_ok=True)
                st.rerun()
            st.video(str(s_vid))
