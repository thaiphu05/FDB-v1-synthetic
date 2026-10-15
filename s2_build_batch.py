#!/usr/bin/env python3
"""Stage 2 — S1 samples -> OmniVoice batch jsonl for B200 (vilex format).

One row per sentence, same 6 fields as
vilex_5.1a_gen/stage5_work/_batch/omnivoice_units_benchmark.jsonl:

    {"id", "text", "ref_audio", "ref_text", "language_id", "speed"}

id follows the vilex pattern ``<rel_path joined by __>__var00__<unit_id>`` with
rel_path = fdb_v1/<task>/<NNNN> and unit_id = u_<part>_<sentence>. Pauses are
not rendered (OmniVoice cannot read [PAUSE]); s4 inserts the silence.

Writes work/_batch/omnivoice_units_fdb.jsonl and work/_batch/samples.jsonl
(the S1 sample plus the unit ids per part, read back by s3/s4).

Usage: python s2_build_batch.py [--config config.yaml] [--limit N]
"""

from common import (
    TASK_PARTS,
    TASKS,
    batch_dir,
    load_config,
    read_jsonl,
    sample_num,
    split_sentences_vi,
    stage_args,
    voice_ref_text,
    write_jsonl,
)


def unit_prefix(task: str, sample_id: str) -> str:
    return f"fdb_v1__{task}__{sample_num(sample_id):04d}__var00"


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    cfg = load_config(args.config, args.limit)
    limit = int(cfg.get("limit") or 0)
    prefix = cfg["paths"]["ref_audio_prefix"].rstrip("/")
    ref_texts = {}
    rows, samples = [], []
    for task in TASKS:
        task_samples = read_jsonl(cfg["paths"]["work_dir"] / "s1_dialogue" / f"{task}.jsonl")
        if limit:
            task_samples = task_samples[:limit]
        for s in task_samples:
            voice = s["voice"]
            if voice not in ref_texts:
                ref_texts[voice] = voice_ref_text(cfg, voice)
            parts = []
            for p, field in enumerate(TASK_PARTS[task]):
                ids = []
                for k, sentence in enumerate(split_sentences_vi(s[field])):
                    uid = f"{unit_prefix(task, s['sample_id'])}__u_{p}_{k}"
                    ids.append(uid)
                    rows.append(
                        {
                            "id": uid,
                            "text": sentence,
                            "ref_audio": f"{prefix}/{voice}.wav",
                            "ref_text": ref_texts[voice],
                            "language_id": cfg["tts"]["language_id"],
                            "speed": cfg["tts"]["speed"],
                        }
                    )
                parts.append(ids)
            samples.append({**s, "parts": parts})
        print(f"[s2] {task}: {len(task_samples)} samples")

    out = batch_dir(cfg)
    write_jsonl(out / "omnivoice_units_fdb.jsonl", rows)
    write_jsonl(out / "samples.jsonl", samples)
    print(f"[s2] {len(rows)} units, {len(ref_texts)} voices -> {out / 'omnivoice_units_fdb.jsonl'}")
    print("[s2] on B200: render this jsonl with the same omnivoice batch command as vilex,")
    print(f"     then put the {{id}}.wav files in {cfg['paths']['unit_audio_dir']}")


if __name__ == "__main__":
    main()
