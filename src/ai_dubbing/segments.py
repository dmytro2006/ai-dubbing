"""Speech segments: the unit that flows through the whole pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Sentence-final punctuation (also covers "..." and the ellipsis character).
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")
# Split after sentence-final punctuation followed by whitespace and an upper-case
# letter, a digit or an opening quote/bracket.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+(?=[\"'“(\[]?[A-Z0-9ĄĆĘŁŃÓŚŹŻ])")


@dataclass
class Segment:
    """One stretch of speech with its original time slot.

    ``start``/``end`` are the slot in the original video (seconds). Everything
    else is filled in by later stages.
    """

    start: float
    end: float
    text_en: str
    text_pl: str = ""
    tts_path: str = ""
    tts_duration: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


def ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences (simple rule-based splitter, good enough for ASR output)."""
    text = " ".join(text.split())
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def merge_into_sentences(
    segments: list[Segment], max_gap: float = 0.8, max_duration: float = 15.0
) -> list[Segment]:
    """Join ASR chunks that split a sentence in the middle.

    Whisper often cuts a sentence into two chunks. Translating half a sentence
    gives a bad translation, so a chunk is glued to the previous one when the
    previous one does not end a sentence, the pause between them is short and the
    result is not too long.
    """
    merged: list[Segment] = []
    for seg in segments:
        if merged:
            prev = merged[-1]
            gap = seg.start - prev.end
            if (
                not ends_sentence(prev.text_en)
                and gap <= max_gap
                and seg.end - prev.start <= max_duration
            ):
                prev.end = seg.end
                prev.text_en = f"{prev.text_en.strip()} {seg.text_en.strip()}"
                continue
        merged.append(Segment(seg.start, seg.end, seg.text_en.strip()))
    return merged


def save_segments(segments: list[Segment], path: str | Path) -> None:
    data = [asdict(s) for s in segments]
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_segments(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Segment(**d) for d in data]


def _srt_time(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[Segment], path: str | Path, lang: str = "en") -> None:
    lines = []
    for i, seg in enumerate(segments, 1):
        text = seg.text_en if lang == "en" else seg.text_pl
        lines += [str(i), f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}", text, ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
