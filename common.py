"""Shared helpers for the fdb-v1-gen stages: config, jsonl, LLM, text, audio."""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
TASKS = ("pause_handling", "smooth_turn_taking", "user_interruption", "backchannel")
# Folder names of the final dataset (FDB v1 layout, synthetic_ prefix as in the paper).
TASK_DIRS = {t: f"synthetic_{t}" for t in TASKS}
# Text parts rendered per task, in playback order (part index = <p> in unit id u_<p>_<k>).
TASK_PARTS = {
    "pause_handling": ("text_a", "text_b"),
    "smooth_turn_taking": ("text",),
    "user_interruption": ("context", "interrupt"),
    "backchannel": ("text",),
}


# ------------------------------------------------------------------ config
def stage_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(HERE / "config.yaml"))
    p.add_argument("--force", action="store_true", help="redo samples that are already done")
    p.add_argument("--limit", type=int, default=None, help="override config limit (samples per task)")
    return p.parse_args()


def load_config(path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if limit is not None:
        cfg["limit"] = limit
    base = cfg_path.parent
    for key in ("data_dir", "api_md", "voice_pool_dir", "work_dir", "unit_audio_dir", "out_dir"):
        cfg["paths"][key] = (base / cfg["paths"][key]).resolve()
    return cfg


def batch_dir(cfg) -> Path:
    return cfg["paths"]["work_dir"] / "_batch"


# ------------------------------------------------------------------- jsonl
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_num(sample_id: str) -> int:
    """'pause_handling_0007' -> 7."""
    return int(sample_id.rsplit("_", 1)[1])


# --------------------------------------------------------------------- LLM
def router_key(cfg) -> str:
    key = os.environ.get("ROUTER_API_KEY", "")
    if key:
        return key
    api_md = cfg["paths"]["api_md"]
    m = re.search(r"sk-[A-Za-z0-9-]+", api_md.read_text(encoding="utf-8")) if api_md.is_file() else None
    if not m:
        raise SystemExit("9router key missing: set ROUTER_API_KEY or paths.api_md")
    return m.group(0)


def chat(cfg, prompt: str, temperature: Optional[float] = None) -> str:
    """One OpenAI-compatible chat call through 9router, retried on any failure."""
    import requests

    llm = cfg["llm"]
    url = llm["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {router_key(cfg)}", "Content-Type": "application/json"}
    payload = {
        "model": llm["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": llm["temperature"] if temperature is None else temperature,
        "stream": False,  # 9router streams SSE by default
    }
    last = None
    for attempt in range(int(llm["max_retries"])):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=llm["timeout"])
            if r.status_code in (401, 402, 403):
                raise SystemExit(f"9router auth/credit error {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except SystemExit:
            raise
        except Exception as exc:  # network, 429/5xx, malformed body
            last = exc
            time.sleep(min(60, 4 * 2**attempt))
    raise RuntimeError(f"LLM failed after {llm['max_retries']} tries: {last}")


def parse_json_reply(text: str) -> Any:
    """Parse a JSON value from an LLM reply, tolerating ```json fences."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


# -------------------------------------------------------------------- text
def split_sentences_vi(text: str) -> List[str]:
    """Same rule as Vilex stage5_lib.split_sentences_vi: split after . ? !"""
    chunks = [c.strip() for c in re.split(r"(?<=[\.\?!])\s+", text) if c.strip()]
    return chunks or [text.strip()]


def words(text: str) -> List[str]:
    return text.split()


def norm_chars(s: str) -> str:
    """Lowercase letters/digits only: used to line up tokens with aligner words."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


# ------------------------------------------------------------------ voices
def voice_pool(cfg) -> List[str]:
    """Sorted names (stem) of pool voices that have a .txt reference transcript."""
    pool = cfg["paths"]["voice_pool_dir"]
    names = sorted(w.stem for w in pool.glob("*.wav") if w.with_suffix(".txt").is_file())
    if not names:
        raise SystemExit(f"voice pool {pool} has no <name>.wav + <name>.txt pairs")
    return names


def voice_ref_text(cfg, name: str) -> str:
    return (cfg["paths"]["voice_pool_dir"] / f"{name}.txt").read_text(encoding="utf-8").strip()


# ------------------------------------------------------------------- audio
def load_mono(path: Path, sr: int) -> np.ndarray:
    import soundfile as sf

    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), file_sr, sr)


def resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr or x.size == 0:
        return x.astype(np.float32, copy=False)
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), int(target_sr))
    return resample_poly(x, int(target_sr) // g, int(sr) // g).astype(np.float32)


def trim_bounds(x: np.ndarray, sr: int, db: float, margin_sec: float) -> Tuple[int, int]:
    """[start, end) sample bounds of the speech in x (10 ms frames above peak+db dB)."""
    if x.size == 0:
        return 0, 0
    hop = max(1, int(0.01 * sr))
    n = int(np.ceil(x.size / hop))
    pad = np.zeros(n * hop, dtype=np.float32)
    pad[: x.size] = np.abs(x)
    frame_peak = pad.reshape(n, hop).max(axis=1)
    thr = frame_peak.max() * 10 ** (db / 20)
    active = np.flatnonzero(frame_peak > thr)
    if active.size == 0:
        return 0, 0
    margin = int(margin_sec * sr)
    return max(0, active[0] * hop - margin), min(x.size, (active[-1] + 1) * hop + margin)


def write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")


# ------------------------------------------------------------------- align
def read_align_txt(path: Path) -> List[Dict[str, Any]]:
    """Parse '<start> <end> <word>' lines (vilex align dump format)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3:
            out.append({"word": parts[2], "start": float(parts[0]), "end": float(parts[1])})
    return out


def write_align_txt(path: Path, ws: List[Dict[str, Any]]) -> None:
    lines = [f"{w['start']:.4f} {w['end']:.4f} {w['word']}" for w in ws]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def even_words(text: str, start: float, end: float) -> List[Dict[str, Any]]:
    """Fallback timing: split [start, end] evenly over the words (1 VI word = 1 syllable)."""
    ws = words(text)
    if not ws:
        return []
    step = (end - start) / len(ws)
    return [
        {"word": w, "start": start + i * step, "end": start + (i + 1) * step}
        for i, w in enumerate(ws)
    ]


def token_end_times(tokens: List[str], aligned: List[Dict[str, Any]]) -> List[float]:
    """End time of each text token, matched to aligner words by character position.

    The aligner may drop punctuation or split tokens differently, so both sides
    are reduced to alphanumeric characters and a token ends where the aligned
    word covering its last character ends.
    """
    spans, pos = [], 0
    for w in aligned:
        n = len(norm_chars(w["word"]))
        spans.append((pos, pos + n, w["end"]))
        pos += n
    total = max(pos, 1)
    out, cur, j = [], 0, 0
    tok_total = max(sum(len(norm_chars(t)) for t in tokens), 1)
    for t in tokens:
        cur += len(norm_chars(t))
        # rescale when the two character streams differ in length (e.g. numbers read out)
        target = cur * total / tok_total
        while j < len(spans) - 1 and spans[j][1] < target:
            j += 1
        out.append(spans[j][2] if spans else 0.0)
    return out
