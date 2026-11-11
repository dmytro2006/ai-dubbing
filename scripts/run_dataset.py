"""Dub every clip of a dataset split (each with its speaker's voice sample) and
collect the timing metrics of all methods into one CSV.

Usage:
    uv run python scripts/run_dataset.py --split dev
    uv run python scripts/run_dataset.py --split test --max-stretch 1.3

The test split is meant to stay untouched until the method parameters are
frozen (blind evaluation), so run it only when that point is reached.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from ai_dubbing.pipeline import DubbingConfig, run
from ai_dubbing.sync import METHODS

DATA = Path(__file__).resolve().parent.parent / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("output"))
    ap.add_argument("-m", "--method", dest="methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--max-stretch", type=float, default=None)
    ap.add_argument("--min-stretch", type=float, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("ai_dubbing").setLevel(logging.INFO)

    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    clips = [c for c in meta["clips"] if args.split == "all" or c["split"] == args.split]
    missing = [c["file"] for c in clips if not (DATA / c["file"]).exists()]
    if missing:
        print(f"Missing clips ({len(missing)}); run scripts/build_dataset.py first.", file=sys.stderr)
        return 1

    out_dir = args.output_dir / args.split
    by_speaker = defaultdict(list)
    for c in clips:
        by_speaker[c["speaker_id"]].append(c)

    rows = []
    for spk, spk_clips in by_speaker.items():
        cfg = DubbingConfig(
            methods=tuple(args.methods),
            output_dir=out_dir,
            voice_sample=DATA / meta["speakers"][spk]["voice_sample"],
            device=args.device,
            min_stretch=args.min_stretch,
            max_stretch=args.max_stretch,
            reuse=args.reuse,
        )
        jobs = run([DATA / c["file"] for c in spk_clips], cfg)
        for c, job in zip(spk_clips, jobs):
            for method, m in job.metrics.items():
                rows.append({"clip_id": c["clip_id"], "split": c["split"], "speaker_id": spk,
                             "method": method, **m["summary"], **job.timings})  # fmt: skip

    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        path = out_dir / "summary.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} rows -> {path}")
        for method in args.methods:
            r = [x for x in rows if x["method"] == method]
            mean = {k: sum(abs(x[k]) for x in r) / len(r)
                    for k in ("total_rel_duration_error", "mean_abs_end_error_s", "mean_abs_log_stretch")}
            print(f"  {method.upper()}: mean |total duration error| {mean['total_rel_duration_error']:.1%}, "
                  f"mean |end error| {mean['mean_abs_end_error_s']:.2f} s, "
                  f"mean |log k| {mean['mean_abs_log_stretch']:.3f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
