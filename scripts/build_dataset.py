"""Download the source videos and build the evaluation dataset.

Reads data/dataset.csv (clips) and data/speakers.csv (voice samples), downloads
the public-domain NASA source videos (cached in data/raw/), and writes:

    data/clips/<clip_id>.mp4        10-30 s test/dev clips (H.264 + AAC)
    data/voices/<speaker_id>.wav    voice sample for cloning, cut from a part of
                                    the source that is NOT used in any clip
    data/references/<clip_id>.json  reference transcript + segment boundaries
                                    from NASA's closed captions (to be checked by hand)
    data/metadata.json              everything above in one file

Usage:  uv run python scripts/build_dataset.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSET_URL = "https://images-assets.nasa.gov/video/{id}/{id}{suffix}"
DETAILS_URL = "https://images.nasa.gov/details/{id}"
LICENSE = (
    "Public domain (NASA-produced media are not subject to US copyright; "
    "see https://www.nasa.gov/nasa-brand-center/images-and-media/). "
    "Do not use to imply NASA endorsement."
)
USER_AGENT = "ai-dubbing-dataset/0.1 (AGH University research project)"


def fetch(url: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            dest.unlink(missing_ok=True)
            return False
        raise


def download_source(nasa_id: str, raw_dir: Path) -> dict:
    """Download the video (medium, 1280x720), captions and NASA metadata, once."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    q = urllib.parse.quote(nasa_id)
    video = raw_dir / f"{nasa_id}.mp4"
    if not video.exists():
        print(f"  downloading {nasa_id} ...")
        if not fetch(ASSET_URL.format(id=q, suffix="~medium.mp4"), video):
            raise RuntimeError(f"no medium.mp4 for {nasa_id}")
    captions = None
    for ext in (".vtt", ".srt"):
        path = raw_dir / f"{nasa_id}{ext}"
        if path.exists() or fetch(ASSET_URL.format(id=q, suffix=ext), path):
            captions = path
            break
    meta_path = raw_dir / f"{nasa_id}.metadata.json"
    if not meta_path.exists():
        fetch(f"https://images-assets.nasa.gov/video/{q}/metadata.json", meta_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return {
        "video": video,
        "captions": captions,
        "title": meta.get("AVAIL:Title") or meta.get("XMP:Title", ""),
        "center": meta.get("AVAIL:Center", ""),
        "date": (meta.get("AVAIL:DateCreated") or "")[:10],
        "url": DETAILS_URL.format(id=q),
    }


# --- captions ------------------------------------------------------------------

_TS = r"(\d+:\d\d:\d\d[.,]\d+|\d\d:\d\d[.,]\d+)"


def _seconds(ts: str) -> float:
    parts = ts.replace(",", ".").split(":")
    return sum(float(p) * 60**i for i, p in enumerate(reversed(parts)))


def parse_captions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\n\s*\n", text):
        m = re.search(_TS + r"\s*-->\s*" + _TS, block)
        if not m:
            continue
        body = block[m.end():].split("\n", 1)
        body = body[1] if len(body) > 1 else ""
        body = html.unescape(re.sub(r"<[^>]+>", "", body))
        body = re.sub(r"\[[^\]]*\]|&gt;&gt;|>>", " ", body)
        body = " ".join(body.split())
        if body:
            cues.append({"start": _seconds(m.group(1)), "end": _seconds(m.group(2)), "text": body})
    # Auto-generated SRT files repeat text in overlapping "roll-up" cues; ours are
    # clean enough for the clips we picked, but drop exact consecutive duplicates.
    out = []
    for c in cues:
        if out and c["text"] == out[-1]["text"]:
            out[-1]["end"] = c["end"]
        else:
            out.append(c)
    return out


def normalise_case(text: str) -> str:
    """ALL-CAPS broadcast captions → sentence case (proper nouns need a manual fix)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters or sum(ch.isupper() for ch in letters) / len(letters) < 0.8:
        return text
    text = text.lower()
    text = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bi\b", "I", text)
    for pattern, repl in CASE_FIXES.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


# Proper nouns and acronyms lost by lower-casing ALL-CAPS captions.
CASE_FIXES = {r"\bacet-9\b": "ACET-9", r"\bearth\b": "Earth", r"\b3d\b": "3D",
              r"\bdyna beads\b": "Dynabeads", r"\bnasa\b": "NASA"}


def cue_is_spoken(video: Path, cue: dict, threshold_db: float = -45.0) -> bool:
    """False for captions shown over silence (e.g. on-screen interview questions)."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{cue['start']:.3f}", "-to", f"{cue['end']:.3f}",
         "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    import numpy as np

    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    if x.size == 0:
        return False
    hop = 320
    n = max(1, len(x) // hop)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    # spoken if at least 30% of the frames are above the threshold
    return float((20 * np.log10(rms) > threshold_db).mean()) > 0.3


def clip_reference(cues: list[dict], start: float, end: float) -> dict:
    inside = [c for c in cues if c["end"] > start + 0.2 and c["start"] < end - 0.2]
    rel = [
        {
            "start": round(max(c["start"] - start, 0.0), 3),
            "end": round(min(c["end"], end) - start, 3),
            "text": normalise_case(c["text"]),
        }
        for c in inside
    ]
    text = normalise_case(" ".join(c["text"] for c in inside))
    return {"text_en": text, "cues": rel}


# --- audio/video cutting ---------------------------------------------------------


def snap_to_pause(video: Path, t: float, window: float = 0.25) -> float:
    """Move a cut point to the quietest 20 ms frame within ±window seconds."""
    a = max(0.0, t - window)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{2 * window:.3f}", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    import numpy as np

    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    hop = 320
    n = len(x) // hop
    if n == 0:
        return t
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return round(a + (int(rms.argmin()) + 0.5) * hop / 16000, 3)


def cut_clip(video: Path, start: float, end: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
         "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "160k", "-ac", "1", "-movflags", "+faststart", str(out)],
        check=True,
    )  # fmt: skip


def cut_voice(video: Path, start: float, end: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "22050", "-c:a", "pcm_s16le", str(out)],
        check=True,
    )  # fmt: skip


def overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        print("ffmpeg not found on PATH", file=sys.stderr)
        return 1

    with open(DATA / "dataset.csv", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    with open(DATA / "speakers.csv", encoding="utf-8") as f:
        speakers = {r["speaker_id"]: r for r in csv.DictReader(f)}
    for d in ("clips", "voices", "references"):
        (DATA / d).mkdir(exist_ok=True)

    sources = {}
    for nasa_id in sorted({c["nasa_id"] for c in clips} | {s["nasa_id"] for s in speakers.values()}):
        sources[nasa_id] = download_source(nasa_id, DATA / "raw")

    # Voice samples must not overlap any clip of the same source.
    for spk in speakers.values():
        v = (float(spk["voice_start"]), float(spk["voice_end"]))
        for c in clips:
            if c["nasa_id"] == spk["nasa_id"] and overlaps(v, (float(c["start"]), float(c["end"]))):
                raise SystemExit(f"voice sample of {spk['speaker_id']} overlaps clip {c['clip_id']}")

    meta = {"license": LICENSE, "sources": {}, "speakers": {}, "clips": []}
    for nasa_id, src in sources.items():
        meta["sources"][nasa_id] = {k: str(v) if isinstance(v, Path) else v for k, v in src.items() if k != "video"}
        meta["sources"][nasa_id].pop("captions", None)

    for spk_id, spk in speakers.items():
        src = sources[spk["nasa_id"]]
        out = DATA / "voices" / f"{spk_id}.wav"
        if args.force or not out.exists():
            cut_voice(src["video"], float(spk["voice_start"]), float(spk["voice_end"]), out)
        meta["speakers"][spk_id] = {
            "name": spk["name"], "gender": spk["gender"], "role": spk["role"],
            "source": spk["nasa_id"], "voice_sample": f"voices/{spk_id}.wav",
            "voice_start": float(spk["voice_start"]), "voice_end": float(spk["voice_end"]),
        }  # fmt: skip
        print(f"voice  {spk_id:10s} {float(spk['voice_end']) - float(spk['voice_start']):5.1f} s")

    for c in clips:
        src = sources[c["nasa_id"]]
        start = snap_to_pause(src["video"], float(c["start"]))
        end = snap_to_pause(src["video"], float(c["end"]))
        out = DATA / "clips" / f"{c['clip_id']}.mp4"
        if args.force or not out.exists():
            cut_clip(src["video"], start, end, out)
        ref = {}
        if src["captions"]:
            cues = [q for q in parse_captions(src["captions"]) if cue_is_spoken(src["video"], q)]
            ref = clip_reference(cues, start, end)
        ref_doc = {
            "clip_id": c["clip_id"],
            "source": "NASA closed captions, cut to the clip; NOT yet verified by hand",
            "verified": False,
            **ref,
        }
        (DATA / "references" / f"{c['clip_id']}.json").write_text(
            json.dumps(ref_doc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        meta["clips"].append({
            "clip_id": c["clip_id"], "split": c["split"], "speaker_id": c["speaker_id"],
            "gender": speakers[c["speaker_id"]]["gender"], "source": c["nasa_id"],
            "source_url": src["url"], "source_start": start, "source_end": end,
            "duration": round(end - start, 3), "file": f"clips/{c['clip_id']}.mp4",
            "reference": f"references/{c['clip_id']}.json", "notes": c.get("notes", ""),
        })  # fmt: skip
        print(f"clip   {c['clip_id']:12s} {c['split']:4s} {end - start:5.1f} s")

    (DATA / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    n_dev = sum(c["split"] == "dev" for c in clips)
    print(f"\n{len(clips)} clips ({n_dev} dev / {len(clips) - n_dev} test), "
          f"{len(speakers)} speakers -> {DATA / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
