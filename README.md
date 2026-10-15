# fdb-v1-gen

Tạo bộ Full-Duplex-Bench v1 tiếng Việt từ hội thoại text 2 người trong `raw_data_gen/data`,
theo cách tạo dữ liệu của paper (FullDuplexBenchv1.pdf, mục III-C). Output theo layout dataset FDB v1,
nên `v1_v1.5/evaluation` chạy được mà không phải sửa.

| Task | `input.wav` (chỉ có giọng user, mono 16 kHz) | Annotation | Số mẫu |
|---|---|---|---|
| pause handling | lượt nói + 1 pause 0.4–1.0 s ở giữa câu | `pause.json` | 137 |
| smooth turn-taking | lượt nói (> 4 s) + 5 s im lặng | `turn_taking.json` | 119 |
| user interruption | yêu cầu + 7 s im lặng + câu ngắt lời (làm rõ / phản đối / chuyển hướng) + 15 s im lặng | `interrupt.json`, `context.wav`, `interrupt.wav` | 200 |
| backchannel | lượt nói dài + 2 s im lặng | `icc_gt_distribution.json` (GT theo cửa sổ 200 ms) | 55 |

## Các stage

Mọi tham số nằm trong `config.yaml` (đường dẫn, số mẫu, route LLM, timing của paper, aligner).
Mỗi stage chỉ đọc output của stage trước, chạy lại được. `--limit N` để chạy thử N mẫu mỗi task.

```bash
P=~/miniconda3/envs/duplexgen/bin/python
cd fdb-v1-gen

# 1. [local] chọn lượt user + gọi LLM (9router, ag/gemini-3.8-flash-low)
$P s1_gen_dialogue.py            # -> work/s1_dialogue/{task}.jsonl

# 2. [local] xuất batch OmniVoice (cùng format omnivoice_units_benchmark.jsonl của vilex)
$P s2_build_batch.py             # -> work/_batch/omnivoice_units_fdb.jsonl, samples.jsonl

# 3. [B200] render work/_batch/omnivoice_units_fdb.jsonl bằng lệnh omnivoice batch như vilex,
#    rồi chép các file {id}.wav về work/units/ (paths.unit_audio_dir)

# 4. [B200 hoặc local] forced align (Qwen3-ForcedAligner-0.6B-hf) -> {id}.txt cạnh {id}.wav
$P s3_align.py                   # aligner.device: cuda trên B200, tự chuyển sang cpu nếu không có GPU

# 5. [local] ghép dataset + kiểm tra
$P s4_assemble.py                # -> out/fdb_v1_vi/synthetic_*/{ID}/
$P s5_check.py                   # -> out/fdb_v1_vi/report.md
```

- **LLM**: route 9router trong `llm.base_url`. Key lấy từ biến môi trường `ROUTER_API_KEY`, nếu không có thì đọc từ `paths.api_md`.
  Kết quả LLM được cache trong `work/s1_dialogue/_cache/` kèm hash của prompt (`prompt_version`). Chạy lại S1 thì
  chỉ gọi LLM cho mẫu còn thiếu, mẫu lỗi (xem `failed.jsonl`) hoặc mẫu sinh bằng prompt cũ; `--force` để gọi lại toàn bộ.
- **Prompt theo paper** (mục III-B/C):
  - pause: `[PAUSE]` chỉ đặt ở chỗ người nói rõ ràng còn giữ lượt (vế trước chưa trọn ý), không đặt sau một vế đã trọn ý.
  - interruption: như paper (GPT-4o viết cả 2 lượt), LLM viết lại lượt user gốc thành yêu cầu tự đứng được, cần trả lời dài,
    rồi viết câu chen ngang. Ý định xoay vòng theo 3 loại của paper (clarify / disagree / shift); cụm mở đầu được gán
    ngẫu nhiên theo seed (khoảng 1/3 vào thẳng nội dung) để không dồn về một mẫu câu.
    Lượt gốc được lưu ở `source_text`.
  - backchannel: mô phỏng thí nghiệm ICC: tỉ lệ người nghe (trong ~60 người) sẽ chêm backchannel ngay sau mỗi từ.
- **Giọng**: dùng mọi voice trong `paths.voice_pool_dir` (cặp `<name>.wav` + `<name>.txt`). Mỗi hội thoại nguồn bốc
  2 giọng random (`voice_picks.user` / `voice_picks.assistant`). Input FDB chỉ có kênh user nên chỉ render giọng user.
  `ref_audio` trong batch = `paths.ref_audio_prefix/<name>.wav`, tức là đường dẫn của pool trên B200.
- **Timestamp**: mốc trong `pause.json`, `turn_taking.json`, `interrupt.json` tính từ số sample lúc ghép audio.
  Thời điểm từng từ (`transcription.json`, GT backchannel) lấy từ kết quả align.
  Unit align lỗi sẽ dùng fallback chia đều và được liệt kê trong `work/units/align_fallback.txt`.
  Mẫu backchannel có unit fallback bị bỏ (xem `out/fdb_v1_vi/skipped.jsonl`).

## Eval

Không cần sửa `v1_v1.5/`. Model chạy trên `input.wav` sinh ra `output.wav` trong từng folder mẫu,
rồi chạy `get_transcript/asr.py` để có `output.json`, sau đó:

```bash
E=<repo>/v1_v1.5/evaluation
python $E/evaluate.py --task pause_handling     --root_dir out/fdb_v1_vi/synthetic_pause_handling
python $E/evaluate.py --task smooth_turn_taking --root_dir out/fdb_v1_vi/synthetic_smooth_turn_taking
python $E/evaluate.py --task user_interruption  --root_dir out/fdb_v1_vi/synthetic_user_interruption

# backchannel đọc ./icc_gt_distribution.json theo thư mục hiện tại -> cd vào folder task
cd out/fdb_v1_vi/synthetic_backchannel && python $E/evaluate.py --task backchannel --root_dir .
```
