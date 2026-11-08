"""FFmpeg helpers: probing, audio extraction, time-stretching and remuxing.

Audio is passed around as mono float32 NumPy arrays. FFmpeg does all decoding,
resampling and tempo changes, so no extra audio libraries are needed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise FFmpegError(
            f"{' and '.join(missing)} not found on PATH. Install FFmpeg "
            "(macOS: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`)."
        )


def _run(cmd: list[str], input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise FFmpegError(f"Command failed: {' '.join(cmd[:4])} ...\n" + "\n".join(err))
    return proc.stdout


def probe(path: str | Path) -> dict:
    out = _run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]
    )
    return json.loads(out)


def duration(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


def has_video(path: str | Path) -> bool:
    streams = probe(path).get("streams", [])
    return any(
        s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )


def load_audio(
    path: str | Path, sample_rate: int = 16000, start: float | None = None, end: float | None = None
) -> np.ndarray:
    """Decode (a part of) any audio/video file to mono float32 at ``sample_rate``."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    return np.frombuffer(_run(cmd), dtype=np.float32).copy()


def save_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono 16-bit PCM WAV."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def atempo_chain(factor: float) -> str:
    """Build an FFmpeg ``atempo`` filter chain for any positive speed factor.

    A single ``atempo`` accepts 0.5–2.0 (older FFmpeg builds), so larger or
    smaller factors are split into several stages.
    """
    if factor <= 0:
        raise ValueError("tempo factor must be positive")
    stages = []
    remaining = factor
    while remaining > 2.0:
        stages.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def time_stretch(audio: np.ndarray, sample_rate: int, factor: float) -> np.ndarray:
    """Change tempo without changing pitch (FFmpeg ``atempo``, a WSOLA-style method).

    ``factor`` > 1 speeds up (shorter output), < 1 slows down.
    """
    if abs(factor - 1.0) < 1e-3 or audio.size == 0:
        return audio
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "-",
        "-filter:a", atempo_chain(factor),
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-",
    ]  # fmt: skip
    out = _run(cmd, input_bytes=audio.astype(np.float32).tobytes())
    return np.frombuffer(out, dtype=np.float32).copy()


def remux(
    video_path: str | Path,
    audio_path: str | Path,
    out_path: str | Path,
    extend_video_to: float | None = None,
) -> None:
    """Replace the audio track of a video.

    The picture is copied untouched. If the new audio runs longer than the video,
    pass ``extend_video_to`` (seconds) and the last frame is frozen so no speech is
    cut off; this needs a video re-encode.
    """
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(video_path), "-i", str(audio_path)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    video_len = duration(video_path)
    if extend_video_to is not None and extend_video_to > video_len + 0.05:
        pad = extend_video_to - video_len
        cmd += [
            "-filter:v", f"tpad=stop_mode=clone:stop_duration={pad:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        ]  # fmt: skip
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)]
    _run(cmd)
