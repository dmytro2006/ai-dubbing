"""Fast tests for the model-free parts of the pipeline (no downloads, no GPU)."""

import shutil

import numpy as np
import pytest

from ai_dubbing.asr import refine_boundaries
from ai_dubbing.media import atempo_chain, time_stretch
from ai_dubbing.metrics import timing_metrics
from ai_dubbing.segments import Segment, merge_into_sentences, split_sentences
from ai_dubbing.sync import Placement, assemble_track, build_track, stretch_factor
from ai_dubbing.tts import trim_silence

SR = 16000
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds: float, sr: int = SR, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --- segments ----------------------------------------------------------------


def test_split_sentences_keeps_every_sentence():
    text = "First one. Second one? Third, with 3.5 km! Done"
    assert split_sentences(text) == ["First one.", "Second one?", "Third, with 3.5 km!", "Done"]


def test_merge_joins_half_sentences_only():
    segs = [
        Segment(0.0, 2.0, "We went to the"),
        Segment(2.2, 4.0, "Moon in 1969."),
        Segment(4.1, 6.0, "It was great."),
        Segment(9.0, 10.0, "Long pause before"),  # gap too long to merge with the next
        Segment(11.5, 12.0, "this."),
    ]
    merged = merge_into_sentences(segs, max_gap=0.8)
    assert [(s.start, s.end) for s in merged] == [(0.0, 4.0), (4.1, 6.0), (9.0, 10.0), (11.5, 12.0)]
    assert merged[0].text_en == "We went to the Moon in 1969."


def test_refine_boundaries_trims_silence():
    audio = np.concatenate([np.zeros(SR), tone(2.0), np.zeros(SR)])  # speech 1.0-3.0 s
    (seg,) = refine_boundaries([Segment(0.0, 4.0, "x")], audio, SR, pad_s=0.0)
    assert seg.start == pytest.approx(1.0, abs=0.03)
    assert seg.end == pytest.approx(3.0, abs=0.03)


def test_trim_silence():
    audio = np.concatenate([np.zeros(SR // 2), tone(1.0), np.zeros(SR)])
    assert len(trim_silence(audio, SR, pad_s=0.0)) / SR == pytest.approx(1.0, abs=0.02)


# --- tempo -------------------------------------------------------------------


@pytest.mark.parametrize("factor", [0.3, 0.5, 0.8, 1.0, 1.7, 2.0, 3.5])
def test_atempo_chain_multiplies_to_factor(factor):
    stages = [float(s.split("=")[1]) for s in atempo_chain(factor).split(",")]
    assert all(0.5 <= s <= 2.0 for s in stages)
    assert np.prod(stages) == pytest.approx(factor, rel=1e-4)


def test_stretch_factor_limits():
    assert stretch_factor(6.0, 4.0) == pytest.approx(1.5)
    assert stretch_factor(6.0, 4.0, max_factor=1.2) == pytest.approx(1.2)
    assert stretch_factor(2.0, 4.0, min_factor=0.8) == pytest.approx(0.8)


@needs_ffmpeg
@pytest.mark.parametrize("factor", [0.75, 1.25, 2.5])
def test_time_stretch_changes_length(factor):
    out = time_stretch(tone(2.0), SR, factor)
    assert len(out) / SR == pytest.approx(2.0 / factor, rel=0.03)


# --- track assembly & methods -------------------------------------------------


def test_assemble_track_never_overlaps():
    clips = [tone(3.0), tone(1.0)]
    track, spans = assemble_track(clips, [0.0, 2.0], SR, min_length=5.0)
    assert spans[0] == pytest.approx((0.0, 3.0))
    assert spans[1] == pytest.approx((3.0, 4.0))  # pushed back until the first one ends
    assert len(track) == 5 * SR


def test_m0_keeps_length_and_drifts():
    clips = [tone(3.0), tone(2.0)]
    slots = [(0.0, 2.0), (2.5, 4.0)]
    _, pl = build_track("m0", clips, slots, SR, total_duration=5.0)
    assert [p.factor for p in pl] == [1.0, 1.0]
    assert pl[1].start == pytest.approx(3.0)  # delayed by the overlong first clip


@needs_ffmpeg
def test_m1_fits_slots():
    clips = [tone(3.0), tone(1.0)]
    slots = [(0.0, 2.0), (2.5, 4.0)]
    _, pl = build_track("m1", clips, slots, SR, total_duration=5.0)
    for p, (s, e) in zip(pl, slots):
        assert p.start == pytest.approx(s, abs=0.02)
        assert p.end == pytest.approx(e, abs=0.05)
    assert pl[0].factor == pytest.approx(1.5, abs=1e-3)
    assert pl[1].factor == pytest.approx(1 / 1.5, abs=1e-3)


def test_unknown_method():
    with pytest.raises(ValueError):
        build_track("m9", [], [], SR, 1.0)


# --- metrics -------------------------------------------------------------------


def test_timing_metrics():
    segs = [Segment(0.0, 2.0, "a"), Segment(3.0, 4.0, "b")]
    pls = [Placement(0.0, 3.0, 1.0), Placement(3.0, 4.5, 1.0)]
    m = timing_metrics(segs, pls, video_duration=4.0)
    s = m["summary"]
    assert m["segments"][0]["rel_duration_error"] == pytest.approx(0.5)
    assert s["total_rel_duration_error"] == pytest.approx(0.5)  # 4.5 s of speech vs 3 s
    assert s["mean_abs_end_error_s"] == pytest.approx(0.75)
    assert s["overflow_beyond_video_s"] == pytest.approx(0.5)


def test_chunk_text_respects_limit():
    from ai_dubbing.tts import chunk_text

    text = ("Pierwsze zdanie jest krótkie. " + "Drugie zdanie, które jest długie, ma wiele części, "
            "a każda część dodaje kolejne słowa, aż przekroczy limit znaków modelu.")
    pieces = chunk_text(text, limit=50)
    assert all(len(p) <= 50 for p in pieces)
    assert " ".join(pieces).split() == text.split()
