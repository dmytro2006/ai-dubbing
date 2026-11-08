"""Speech recognition with Whisper (English transcript + segment timestamps)."""

from __future__ import annotations

import logging

import numpy as np

from .segments import Segment

log = logging.getLogger(__name__)

ASR_SAMPLE_RATE = 16000
DEFAULT_WHISPER = "openai/whisper-large-v3-turbo"


class WhisperASR:
    def __init__(self, model_id: str = DEFAULT_WHISPER, device: str = "cpu"):
        import torch
        import transformers

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = transformers.pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=device,
            dtype=dtype,
        )

    def transcribe(self, audio: np.ndarray, language: str = "en") -> list[Segment]:
        """Transcribe 16 kHz mono audio into timestamped segments."""
        total = len(audio) / ASR_SAMPLE_RATE
        out = self.pipe(
            {"raw": audio, "sampling_rate": ASR_SAMPLE_RATE},
            return_timestamps=True,
            chunk_length_s=30,
            generate_kwargs={"language": language, "task": "transcribe"},
        )
        segments = []
        for chunk in out.get("chunks", []):
            start, end = chunk["timestamp"]
            text = chunk["text"].strip()
            if not text:
                continue
            start = 0.0 if start is None else float(start)
            end = total if end is None else float(end)
            segments.append(Segment(min(start, total), min(max(end, start), total), text))
        return segments


def frame_energy_db(audio: np.ndarray, sample_rate: int, frame_s: float = 0.02) -> np.ndarray:
    hop = max(1, int(sample_rate * frame_s))
    n = len(audio) // hop
    if n == 0:
        return np.full(1, -120.0)
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def refine_boundaries(
    segments: list[Segment],
    audio: np.ndarray,
    sample_rate: int = ASR_SAMPLE_RATE,
    threshold_db: float = -35.0,
    frame_s: float = 0.02,
    pad_s: float = 0.05,
) -> list[Segment]:
    """Tighten Whisper timestamps to where speech energy actually starts and stops.

    Whisper segment times are coarse and often include leading/trailing silence.
    The timing method measures the dubbed speech against these slots, so they
    should match the real speech. ``threshold_db`` is relative to the loudest frame
    of the whole recording.
    """
    if not segments:
        return segments
    db = frame_energy_db(audio, sample_rate, frame_s)
    voiced = db > (db.max() + threshold_db)
    refined = []
    for seg in segments:
        a = int(seg.start / frame_s)
        b = max(a + 1, int(np.ceil(seg.end / frame_s)))
        idx = np.flatnonzero(voiced[a:b])
        if idx.size == 0:
            refined.append(seg)
            continue
        start = max(seg.start, (a + idx[0]) * frame_s - pad_s)
        end = min(seg.end, (a + idx[-1] + 1) * frame_s + pad_s)
        refined.append(Segment(round(start, 3), round(end, 3), seg.text_en))
    return refined
