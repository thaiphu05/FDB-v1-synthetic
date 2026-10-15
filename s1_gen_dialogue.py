#!/usr/bin/env python3
"""Stage 1 — select user turns per task and generate the task text with the LLM.

Paper (FDB v1, Sec. III-B/C) builds its synthetic sets from GPT-4o dialogues;
here the user turns come from raw_data_gen/data and the 9router LLM adds what
each task needs:

  pause_handling      one [PAUSE] where the speaker clearly still holds the turn
                      (prefix unfinished), the ChatTTS [uv_break] analogue
  smooth_turn_taking  the turn as-is (no LLM)
  user_interruption   a self-contained context request + an interrupting turn
                      (paper: GPT-4o writes both); intent cycles over the
                      paper's "clarify, disagree, or shift the discussion"
  backchannel         per word, the share of listeners who would backchannel
                      right after it (ICC: ~59 listeners responding at TRPs)

Selection is seeded and round-robins over topics, and a turn is used by at
most one task. LLM results are cached per sample together with a hash of the
task prompt, so a rerun only calls the LLM for missing samples or samples made
with an older prompt (--force recomputes everything).

Writes work/s1_dialogue/{task}.jsonl (and _cache/, failed.jsonl).

Usage: python s1_gen_dialogue.py [--config config.yaml] [--limit N] [--force]
"""

import json
import random
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from common import (
    TASKS,
    chat,
    load_config,
    norm_chars,
    parse_json_reply,
    read_jsonl,
    sample_num,
    stage_args,
    voice_pool,
    words,
    write_jsonl,
)

# Scarcest candidates first, so the long turns go to backchannel.
SELECTION_ORDER = ("backchannel", "pause_handling", "smooth_turn_taking", "user_interruption")
LLM_TRIES = 3
SENT_END = (".", "?", "!")
# Words after which a Vietnamese utterance is clearly unfinished (rule fallback for
# pauses). Only the connectors named in PAUSE_PROMPT: words like "cho", "về" also
# start compounds ("cho thấy", "về phía") that a pause must not split.
CONNECTORS = {"nhưng", "vì", "nên", "mà", "là", "thì", "và", "hoặc", "của", "để", "với", "rằng", "nếu"}
# A comma after these closes a question ("... năm nay không, hay ..."): a turn end.
QUESTION_PARTICLES = {"không", "nhỉ", "chứ", "à", "hả", "sao", "nhé", "ạ"}

# Paper III-B.1: pauses "are often not intended to yield the turn"; the research
# question is whether the model sees "the other speaker is still holding the turn".
# So the text before the pause must be unfinished: a pause after a complete
# clause is a legitimate turn end and would make a takeover look like an error.
PAUSE_PROMPT = """Bạn đang tạo dữ liệu cho bài đánh giá "Pause Handling" (Full-Duplex-Bench): người dùng đang nói thì ngừng khoảng 0.4–1 giây rồi nói tiếp; trợ lý giọng nói phải nhận ra người dùng vẫn đang giữ lượt và không được chen vào.

Hãy chèn đúng MỘT ký hiệu [PAUSE] vào lượt nói dưới đây, tại chỗ người nói ngập ngừng tự nhiên MÀ phần trước [PAUSE] rõ ràng chưa trọn ý, để không người nghe nào nghĩ là đã tới lượt mình.

Chỗ nên đặt:
- Ngay sau một từ nối hoặc giới từ: "nhưng", "vì", "nên", "mà", "là", "thì", "và", "hoặc", "của", "để", "với", "rằng", "nếu"...
- Giữa động từ và tân ngữ, hoặc ngay trước một danh từ, tên riêng, con số quan trọng (như đang tìm từ).
- Sau một vế phụ còn dang dở như "Nếu ...," hoặc "Vì ...,".

Chỗ không được đặt:
- Sau một câu hoặc một vế đã trọn ý, kể cả khi có dấu phẩy. Ví dụ SAI: "Con số đó nghe ấn tượng thật, [PAUSE] nhưng so với ..."
- Sau một câu hỏi hoặc tiểu từ kết thúc câu hỏi ("nhỉ", "chứ", "không" ở cuối câu hỏi), sau dấu chấm, chấm hỏi, chấm than.
- Trong 20% đầu hoặc 20% cuối lượt nói.

Giữ nguyên toàn bộ từ ngữ và dấu câu, không thêm, bớt hay sửa chữ nào. Chỉ trả về lượt nói đã chèn [PAUSE], không giải thích.

Lượt nói:
{text}"""

# Paper III-B.4: a listener "interjects mid-turn to clarify, disagree, or shift the
# discussion"; III-C: GPT-4o generates both the contextual turn and the
# interruption, which plays ~7 s after the context (the model is mid-answer).
# The model under test hears only these two turns, so the context must stand on
# its own and ask for a long answer, and the interruption must not depend on
# what the model happened to say.
INTENTS = {
    "clarify": "làm rõ: người dùng nhận ra mình hỏi chưa đúng ý nên nói rõ hoặc sửa lại chính yêu cầu ban đầu (thu hẹp phạm vi, đổi chi tiết, nói rõ điều thật sự muốn biết).",
    "disagree": "phản đối: người dùng không đồng ý với hướng trả lời mà yêu cầu ban đầu dẫn tới, hoặc phản bác một tiền đề của câu hỏi, rồi đòi một hướng khác cụ thể.",
    "shift": "chuyển hướng: người dùng bỏ dở câu trả lời hiện tại để hỏi sang một khía cạnh hoặc câu hỏi khác có liên quan tới chủ đề.",
}
INTENT_ORDER = ("clarify", "disagree", "shift")
# Openers are assigned per sample: left to itself the LLM opened 187/200
# interruptions with "Khoan đã". About 1/3 start directly with the content, like
# the paper's example.
OPENERS = {
    "clarify": ("À không,", "Ý tôi là", "Xin lỗi, tôi hỏi chưa rõ,", "Không phải,", "Ơ, ý tôi là"),
    "disagree": ("Không, không,", "Nhưng mà", "Khoan,", "Thôi,", "Không cần đâu,"),
    "shift": ("À mà", "Thôi, khoan đã,", "Mà này,", "Nhân tiện,", "Chuyện đó để sau,"),
}
DIRECT_SHARE = 1 / 3
INTERJECTIONS = {"khoan", "à", "ơ", "ờ", "thôi", "ừ", "này", "ê", "ủa", "ấy"}
OPENER_RULE = 'Bắt đầu câu đúng bằng "{opener}" rồi nói tiếp tự nhiên.'
DIRECT_RULE = (
    'Vào thẳng câu hỏi hoặc yêu cầu mới; không mở đầu bằng từ đệm, cụm ngắt lời (như "Khoan", "À", "Thôi") '
    'hay câu dẫn nói về việc đổi chủ đề (như "Chuyển sang", "Tôi muốn hỏi").'
)

INTERRUPT_PROMPT = """Bạn đang tạo dữ liệu cho bài đánh giá "User Interruption" (Full-Duplex-Bench): người dùng nói một yêu cầu với trợ lý giọng nói; trợ lý bắt đầu trả lời; khoảng 7 giây sau, khi trợ lý vẫn đang nói, người dùng chen ngang. Mô hình được chấm theo việc nó có dừng lại và trả lời đúng vào câu chen ngang hay không.

Lượt nói gốc của người dùng (trích từ giữa một cuộc hội thoại dài hơn, có thể nhắc tới những gì đã nói trước đó):
"{source}"

Hãy viết 2 lượt nói tiếng Việt văn nói tự nhiên của người dùng.

1. "context": yêu cầu ban đầu gửi trợ lý.
- Giữ chủ đề và các chi tiết cụ thể (tên, số liệu, sự kiện) của lượt nói gốc.
- Tự đứng được một mình: trợ lý không nghe được gì trước đó, nên không nhắc tới lời trước (như "ý đó", "như bạn nói", "bước 3", "hiểu rồi", "vậy còn").
- Là câu hỏi hoặc yêu cầu cần câu trả lời dài, nhiều ý (liệt kê, giải thích, hướng dẫn), để trợ lý vẫn đang nói khi bị chen ngang.
- 8 đến 25 từ.

2. "interrupt": câu người dùng chen ngang.
- Ý định: {intent}
- {opener_rule}
- Chỉ dựa vào yêu cầu ban đầu và chủ đề, không dựa vào lời trợ lý (vì không biết trợ lý đã nói gì).
- Chứa một câu hỏi hoặc yêu cầu mới, cụ thể, sao cho câu trả lời đúng khác hẳn việc tiếp tục trả lời yêu cầu ban đầu.
- Ngắn như lời chen ngang thật: 6 đến 15 từ.

Ví dụ trong paper (tiếng Anh): context "Could you suggest some books to help with personal development?", interrupt "Can we talk about something else? How about fiction books?"

Chỉ trả về JSON: {{"context": "...", "interrupt": "..."}}"""

# Paper III-C: the backchannel GT comes from ICC, where ~59 listeners per
# stimulus gave a brief backchannel "whenever they found it appropriate" (TRPs),
# counted per 200 ms window. Ask for that share of listeners per word.
BC_PROMPT = """Mô phỏng thí nghiệm backchannel của ICC corpus: khoảng 60 người nghe bản ngữ tiếng Việt cùng nghe lượt nói dưới đây. Mỗi người được dặn chỉ chêm một phản hồi ngắn (như "ừ", "vâng", "ừm", "à", "thế à") bất cứ khi nào họ thấy phù hợp, tức là tại các điểm chuyển lượt tiềm năng (Transition Relevance Place): cuối một ý hoặc một vế câu, cuối câu, chỗ người nói ngừng để chờ xác nhận, và cuối lượt nói.

Lượt nói (mỗi từ được đánh số):
{numbered}

Với MỖI từ i, ước lượng tỉ lệ người nghe (0 đến 1) sẽ phản hồi ngay sau từ i.
- Giữa một cụm từ (giữa động từ và tân ngữ, ngay sau từ nối như "và", "của", "nhưng"): gần 0.
- Cuối một vế câu mà ý còn dang dở: thấp. Cuối một ý trọn vẹn: cao hơn. Cuối câu và cuối lượt nói: cao nhất.
- Mỗi người có thể phản hồi nhiều lần trong một lượt nên các giá trị không cần cộng lại bằng 1.

Trả về DUY NHẤT một mảng JSON gồm đúng {n} số thực, phần tử thứ i ứng với từ i. Ví dụ: [0.0, 0.02, ...]"""

PROMPT_SPECS = {
    "pause_handling": PAUSE_PROMPT + json.dumps([sorted(CONNECTORS), sorted(QUESTION_PARTICLES)], ensure_ascii=False),
    "user_interruption": INTERRUPT_PROMPT
    + json.dumps([INTENTS, OPENERS, DIRECT_SHARE, OPENER_RULE, DIRECT_RULE], ensure_ascii=False),
    "backchannel": BC_PROMPT,
}


def prompt_version(task: str) -> Optional[str]:
    """Hash of the task prompt; cached samples made with another prompt are regenerated."""
    spec = PROMPT_SPECS.get(task)
    return f"{zlib.crc32(spec.encode('utf-8')):08x}" if spec else None


# ---------------------------------------------------------------- selection
def load_user_turns(cfg) -> List[Dict[str, Any]]:
    turns = []
    for path in sorted(cfg["paths"]["data_dir"].glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        hist = d.get("history", [])
        style = (d.get("meta") or {}).get("dialogue_type", "")
        for i, t in enumerate(hist):
            if t.get("role") != "user" or not t.get("content", "").strip():
                continue
            nxt = hist[i + 1]["content"] if i + 1 < len(hist) and hist[i + 1]["role"] == "assistant" else ""
            turns.append(
                {
                    "file": path.name,
                    "turn_idx": i,
                    "topic": path.stem.split("_")[0],
                    "style": style,
                    "text": " ".join(t["content"].split()),
                    "next_assistant": nxt,
                }
            )
    return turns


def eligible(task: str, t: Dict[str, Any], sel: Dict[str, Any]) -> bool:
    n = len(words(t["text"]))
    if task == "backchannel":
        return n >= sel["backchannel_min_words"]
    if task == "pause_handling":
        return n >= sel["pause_min_words"] and "," in t["text"]
    if task == "smooth_turn_taking":
        return n >= sel["turn_taking_min_words"] and t["text"].endswith(SENT_END)
    if task == "user_interruption":
        return t["text"].endswith("?") or not sel["interrupt_context_question"]
    raise ValueError(task)


def select(cfg, turns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Seeded round-robin over topics; each turn goes to at most one task."""
    rng = random.Random(cfg["seed"])
    used, picked = set(), {}
    for task in SELECTION_ORDER:
        by_topic: Dict[str, List[Dict[str, Any]]] = {}
        for t in turns:
            if eligible(task, t, cfg["selection"]):
                by_topic.setdefault(t["topic"], []).append(t)
        for group in by_topic.values():
            rng.shuffle(group)
        topics = sorted(by_topic)
        rng.shuffle(topics)
        need, out = cfg["counts"][task], []
        while len(out) < need and any(by_topic.values()):
            for topic in topics:
                group = by_topic[topic]
                while group and (group[0]["file"], group[0]["turn_idx"]) in used:
                    group.pop(0)
                if group and len(out) < need:
                    t = group.pop(0)
                    used.add((t["file"], t["turn_idx"]))
                    out.append(t)
        if len(out) < need:
            print(f"WARNING: {task}: only {len(out)}/{need} eligible turns")
        picked[task] = out
    return picked


# ------------------------------------------------------------- LLM per task
def split_pause_rule(text: str) -> Optional[int]:
    """Word index where text_b starts, closest to the middle (30–70 %).

    Prefers a boundary right after a connector ("nhưng", "của", ...), where the
    speaker is clearly unfinished; then a comma; a bare word boundary is the
    last resort, since in Vietnamese it can fall inside a compound ("xe khách").
    """
    ws = words(text)
    n = len(ws)
    cands = [i for i in range(1, n) if 0.3 <= i / n <= 0.7 and not ws[i - 1].endswith(SENT_END)]
    connector = [i for i in cands if ws[i - 1].lower() in CONNECTORS]
    commas = [i for i in cands if ws[i - 1].endswith(",") and ws[i - 1][:-1].lower() not in QUESTION_PARTICLES]
    pool = connector or commas or cands
    return min(pool, key=lambda i: abs(i / n - 0.5)) if pool else None


def gen_pause(cfg, text: str) -> Dict[str, Any]:
    ws = words(text)
    for _ in range(LLM_TRIES):
        reply = chat(cfg, PAUSE_PROMPT.format(text=text)).strip().strip('"')
        if reply.count("[PAUSE]") != 1:
            continue
        a, b = (s.strip() for s in reply.split("[PAUSE]"))
        wa, wb = words(a), words(b)
        if wa + wb != ws or not wa or not wb:
            continue  # LLM changed the words
        if wa[-1].endswith(SENT_END) or not 0.2 <= len(wa) / len(ws) <= 0.8:
            continue
        return {"text_a": a, "text_b": b, "pause_by": "llm"}
    i = split_pause_rule(text)
    if i is None:
        raise RuntimeError("no valid pause position")
    return {"text_a": " ".join(ws[:i]), "text_b": " ".join(ws[i:]), "pause_by": "rule"}


def opener_ok(interrupt: str, opener: str) -> bool:
    if opener:
        return norm_chars(interrupt).startswith(norm_chars(opener))
    return norm_chars(words(interrupt)[0]) not in INTERJECTIONS


def gen_interrupt(cfg, source: str, intent: str, opener: str) -> Dict[str, Any]:
    rule = OPENER_RULE.format(opener=opener) if opener else DIRECT_RULE
    prompt = INTERRUPT_PROMPT.format(source=source, intent=INTENTS[intent], opener_rule=rule)
    for _ in range(LLM_TRIES):
        try:
            reply = parse_json_reply(chat(cfg, prompt))
        except ValueError:
            continue
        if not isinstance(reply, dict):
            continue
        context = " ".join(str(reply.get("context", "")).split()).strip('"“” ')
        interrupt = " ".join(str(reply.get("interrupt", "")).split()).strip('"“” ')
        if (
            8 <= len(words(context)) <= 30
            and 5 <= len(words(interrupt)) <= 18
            and norm_chars(interrupt) != norm_chars(context)
            and opener_ok(interrupt, opener)
        ):
            return {"context": context, "interrupt": interrupt, "intent": intent, "opener": opener}
    raise RuntimeError("no valid context/interruption pair")


def gen_bc_probs(cfg, text: str) -> Dict[str, Any]:
    ws = words(text)
    numbered = " ".join(f"{i + 1}:{w}" for i, w in enumerate(ws))
    for _ in range(LLM_TRIES):
        try:
            probs = parse_json_reply(chat(cfg, BC_PROMPT.format(numbered=numbered, n=len(ws)), temperature=0))
        except ValueError:
            continue
        if (
            isinstance(probs, list)
            and len(probs) == len(ws)
            and all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in probs)
        ):
            return {"bc_probs": [round(float(p), 4) for p in probs]}
    raise RuntimeError("no valid bc_probs")


def voice_picks(cfg, file: str, voices: List[str]) -> Dict[str, str]:
    """2 distinct random voices per source dialogue (like vilex voice_picks)."""
    user, assistant = random.Random(f"{cfg['seed']}-{file}").sample(voices, 2)
    return {"user": user, "assistant": assistant}


def build_sample(cfg, task: str, sample_id: str, t: Dict[str, Any]) -> Dict[str, Any]:
    rng = random.Random(f"{cfg['seed']}-{sample_id}")
    s = {
        "sample_id": sample_id,
        "task": task,
        "source": {k: t[k] for k in ("file", "turn_idx", "topic", "style")},
        "next_assistant": t["next_assistant"],
        "prompt_version": prompt_version(task),
    }
    text = t["text"]
    if task == "pause_handling":
        lo, hi = cfg["paper"]["pause_sec"]
        s.update(text=text, pause_sec=round(rng.uniform(lo, hi), 3), **gen_pause(cfg, text))
    elif task == "smooth_turn_taking":
        s.update(text=text)
    elif task == "user_interruption":
        # intents cycle so the three paper categories stay balanced
        intent = INTENT_ORDER[(sample_num(sample_id) - 1) % len(INTENT_ORDER)]
        opener = "" if rng.random() < DIRECT_SHARE else rng.choice(OPENERS[intent])
        s.update(source_text=text, **gen_interrupt(cfg, text, intent, opener))
    elif task == "backchannel":
        s.update(text=text, words=words(text), **gen_bc_probs(cfg, text))
    return s


def is_fresh(hit: Optional[Dict[str, Any]], task: str, t: Dict[str, Any]) -> bool:
    """Cached sample made from this source turn with the current prompt."""
    return bool(
        hit
        and hit["source"]["file"] == t["file"]
        and hit["source"]["turn_idx"] == t["turn_idx"]
        and hit.get("prompt_version") == prompt_version(task)
    )


def main() -> None:
    args = stage_args(__doc__.splitlines()[0])
    cfg = load_config(args.config, args.limit)
    out_dir = cfg["paths"]["work_dir"] / "s1_dialogue"
    voices = voice_pool(cfg)
    picked = select(cfg, load_user_turns(cfg))
    limit = int(cfg.get("limit") or 0)

    jobs, cached, failed = [], {}, []
    sel = {task: picked[task][:limit] if limit else picked[task] for task in TASKS}
    for task in TASKS:
        cached[task] = {r["sample_id"]: r for r in read_jsonl(out_dir / "_cache" / f"{task}.jsonl")}
        for k, t in enumerate(sel[task], 1):
            sid = f"{task}_{k:04d}"
            if args.force or not is_fresh(cached[task].get(sid), task, t):
                cached[task].pop(sid, None)  # a failed retry must not fall back to the stale sample
                jobs.append((task, sid, t))

    print(f"[s1] {len(voices)} voices in {cfg['paths']['voice_pool_dir']}")
    print(f"[s1] {len(jobs)} samples to generate ({cfg['llm']['model']})")
    with ThreadPoolExecutor(max_workers=int(cfg["llm"]["workers"])) as ex:
        futs = {ex.submit(build_sample, cfg, task, sid, t): (task, sid, t) for task, sid, t in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            task, sid, t = futs[fut]
            try:
                cached[task][sid] = fut.result()
            except Exception as exc:
                failed.append({"sample_id": sid, "source": t["file"], "turn_idx": t["turn_idx"], "error": str(exc)})
                print(f"  FAIL {sid}: {exc}")
            if i % 20 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}")
            if i % 50 == 0:  # checkpoint the cache so a crash keeps finished LLM work
                for tk in TASKS:
                    write_jsonl(out_dir / "_cache" / f"{tk}.jsonl", sorted(cached[tk].values(), key=lambda r: r["sample_id"]))

    for task in TASKS:
        write_jsonl(out_dir / "_cache" / f"{task}.jsonl", sorted(cached[task].values(), key=lambda r: r["sample_id"]))
        n_sel = len(sel[task])
        rows = [
            cached[task][sid]
            for sid, t in ((f"{task}_{k:04d}", t) for k, t in enumerate(sel[task], 1))
            if is_fresh(cached[task].get(sid), task, t)
        ]
        for r in rows:  # voices are not LLM output: always reassigned from the current pool
            r["voice_picks"] = voice_picks(cfg, r["source"]["file"], voices)
            r["voice"] = r["voice_picks"]["user"]  # FDB input is user-only, only this voice is rendered
        write_jsonl(out_dir / f"{task}.jsonl", rows)
        extra = ""
        if task == "pause_handling":
            extra = f" (rule fallback: {sum(r['pause_by'] == 'rule' for r in rows)})"
        print(f"[s1] {task}: {len(rows)}/{n_sel}{extra}")
    write_jsonl(out_dir / "failed.jsonl", failed)
    if failed:
        print(f"[s1] {len(failed)} failed -> {out_dir / 'failed.jsonl'} (rerun s1 to retry)")


if __name__ == "__main__":
    main()
