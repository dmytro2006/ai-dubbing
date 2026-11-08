"""Timing synchronisation: build the dubbed audio track for methods M0 and M1.

M0 – no correction. Each synthesised segment starts at its original start time.
     If the previous segment is still playing, it waits until that one ends, so
     the delay accumulates (speech never overlaps).
M1 – post-synthesis time-stretching. Each segment is sped up / slowed down
     (pitch preserved) to the length of its original slot, optionally within
     ``[min_factor, max_factor]``, then placed like in M0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .media import time_stretch

METHODS = ("m0", "m1")


@dataclass
class Placement:
    start: float  # where the dubbed segment actually starts in the new track (s)
    end: float
    factor: float  # tempo factor applied (1.0 = unchanged, >1 = faster)


def stretch_factor(
    tts_duration: float,
    slot_duration: float,
    min_factor: float | None = None,
    max_factor: float | None = None,
) -> float:
    """Tempo factor that makes ``tts_duration`` last ``slot_duration``, clamped to limits."""
    if tts_duration <= 0 or slot_duration <= 0:
        return 1.0
    k = tts_duration / slot_duration
    if max_factor is not None:
        k = min(k, max_factor)
    if min_factor is not None:
        k = max(k, min_factor)
    return k


def assemble_track(
    clips: list[np.ndarray],
    starts: list[float],
    sample_rate: int,
    min_length: float = 0.0,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Lay clips on a silent track at the requested start times without overlap.

    Returns the track and the actual (start, end) of every clip.
    """
    placed: list[tuple[int, int]] = []
    cursor = 0
    for clip, start in zip(clips, starts):
        s = max(int(round(start * sample_rate)), cursor)
        placed.append((s, s + len(clip)))
        cursor = s + len(clip)
    length = max(cursor, int(round(min_length * sample_rate)))
    track = np.zeros(length, dtype=np.float32)
    for clip, (s, e) in zip(clips, placed):
        track[s:e] += clip
    return track, [(s / sample_rate, e / sample_rate) for s, e in placed]


def build_track(
    method: str,
    clips: list[np.ndarray],
    slots: list[tuple[float, float]],
    sample_rate: int,
    total_duration: float,
    min_factor: float | None = None,
    max_factor: float | None = None,
) -> tuple[np.ndarray, list[Placement]]:
    """Build the dubbed track for one method.

    ``clips`` are the synthesised segments and ``slots`` their original
    (start, end) times.
    """
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; available: {', '.join(METHODS)}")

    factors = [1.0] * len(clips)
    if method == "m1":
        new_clips = []
        for i, (clip, (s, e)) in enumerate(zip(clips, slots)):
            k = stretch_factor(len(clip) / sample_rate, e - s, min_factor, max_factor)
            factors[i] = k
            new_clips.append(time_stretch(clip, sample_rate, k))
        clips = new_clips

    track, spans = assemble_track(clips, [s for s, _ in slots], sample_rate, total_duration)
    placements = [Placement(round(a, 3), round(b, 3), round(k, 4)) for (a, b), k in zip(spans, factors)]
    return track, placements
