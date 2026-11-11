"""Objective timing metrics for one dubbed clip.

WER (jiwer) and voice similarity (ECAPA-TDNN) come in a later milestone; this
module covers the timing side of the evaluation plan.
"""

from __future__ import annotations

import numpy as np

from .segments import Segment
from .sync import Placement


def timing_metrics(
    segments: list[Segment], placements: list[Placement], video_duration: float
) -> dict:
    """Per-segment and aggregate timing errors of the dubbed track vs. original slots.

    - ``rel_duration_error``: (dubbed length − slot length) / slot length
    - ``start_error`` / ``end_error``: dubbed boundary − original boundary (s)
    - ``stretch_factor``: tempo change applied (k_stretch, 1.0 = none)
    """
    rows = []
    for seg, pl in zip(segments, placements):
        slot = seg.end - seg.start
        dub = pl.end - pl.start
        rows.append(
            {
                "slot_start": seg.start,
                "slot_end": seg.end,
                "dub_start": pl.start,
                "dub_end": pl.end,
                "rel_duration_error": round((dub - slot) / slot, 4) if slot > 0 else None,
                "start_error": round(pl.start - seg.start, 3),
                "end_error": round(pl.end - seg.end, 3),
                "stretch_factor": pl.factor,
            }
        )
    if not rows:
        return {"segments": [], "summary": {}}

    rel = np.array([abs(r["rel_duration_error"]) for r in rows if r["rel_duration_error"] is not None])
    starts = np.array([abs(r["start_error"]) for r in rows])
    ends = np.array([abs(r["end_error"]) for r in rows])
    k = np.array([r["stretch_factor"] for r in rows])
    slot_total = sum(s.end - s.start for s in segments)
    dub_total = sum(p.end - p.start for p in placements)
    summary = {
        "n_segments": len(rows),
        "mean_abs_rel_duration_error": round(float(rel.mean()), 4) if rel.size else None,
        "total_rel_duration_error": round((dub_total - slot_total) / slot_total, 4)
        if slot_total > 0
        else None,
        "mean_abs_start_error_s": round(float(starts.mean()), 3),
        "mean_abs_end_error_s": round(float(ends.mean()), 3),
        "max_abs_end_error_s": round(float(ends.max()), 3),
        "overflow_beyond_video_s": round(max(0.0, placements[-1].end - video_duration), 3),
        "mean_stretch_factor": round(float(k.mean()), 4),
        "max_stretch_factor": round(float(k.max()), 4),
        "min_stretch_factor": round(float(k.min()), 4),
        "mean_abs_log_stretch": round(float(np.abs(np.log(k)).mean()), 4),
    }
    return {"segments": rows, "summary": summary}
