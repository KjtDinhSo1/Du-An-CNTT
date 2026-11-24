
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".hf_cache"))       # mô hình nằm ở ổ D:, không tải lại
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
SAVE_EVERY = 480                                                 # số bài giữa hai lần lưu (~5 phút trên CPU)

def other_instances():
    """Số bản phobert_embed.py khác đang chạy. Hai bản cùng ghi một file vector sẽ giẫm lên nhau (FileNotFoundError, mất tiến độ)."""
    import subprocess
    cmd = (f"@(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -like 'python*' -and $_.ProcessId -ne {os.getpid()} "
           f"-and $_.CommandLine -like '*phobert_embed.py*' }}).Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


if other_instances():
    sys.exit("Đã có một bản phobert_embed.py đang chạy (xem tab Terminal). Chờ nó xong hoặc bấm Ctrl+C ở tab đó rồi chạy lại; "
             "KHÔNG chạy hai bản cùng lúc. Tiến độ đã lưu được giữ nguyên.")

# ---- nạp các ô cần thiết của notebook: cấu hình, từ khóa, làm sạch, đọc dữ liệu, tách từ, hàm PhoBERT ----
cells = ["".join(c["source"]) for c in json.load(open(HERE / "source.ipynb", encoding="utf8"))["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "phobert_embed"}
for src in cells:
    if "uni_vec = CountVectorizer" in src:                       # ô tách từ: chỉ lấy phần định nghĩa segment()
        exec(src.split("# Tách theo TỪNG DÒNG")[0], ns)
        break
    exec(src, ns)
pb_cell = next(s for s in cells if "def phobert_encode" in s)
exec(pb_cell.split("\nif CFG.PHOBERT_ENABLED:")[0], ns)          # các hàm PhoBERT, không chạy phần chính
CFG, df, np = ns["CFG"], ns["df_meaning"], ns["np"]
assert ns["OUT_DIR"] is not None, "Không tìm thấy dữ liệu thật trong thư mục data"
CFG.PHOBERT_BATCH = 16                                           # lô nhỏ hợp với máy 8 GB RAM

cache_path = ns["OUT_DIR"] / "phobert_embeddings.npz"
tmp_path = ns["OUT_DIR"] / f"phobert_embeddings.{os.getpid()}.tmp.npz"   # riêng cho từng tiến trình
cache = {}
if cache_path.exists():
    with np.load(cache_path) as z:                               # đóng file ngay: Windows không cho ghi đè file đang mở
        cache = dict(zip(z["keys"].tolist(), z["vecs"]))         # giữ float16 để tiết kiệm RAM
keys = [ns["doc_key"](t, l) for t, l in zip(df["title"], df["lyrics"])]
todo = [i for i, k in enumerate(keys) if k not in cache]
if "--limit" in sys.argv:
    todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]
print(f"\nTổng {len(df):,} bài | đã có vector: {len(df) - len([k for k in keys if k not in cache]):,} | lần này tính: {len(todo):,}", flush=True)


def save():
    """Ghi ra file tạm rồi đổi tên: tắt máy đúng lúc đang ghi cũng không làm hỏng file cũ."""
    np.savez_compressed(tmp_path, keys=np.array(list(cache), dtype=np.int64), vecs=np.vstack(list(cache.values())).astype(np.float16))
    for attempt in range(30):                                    # notebook có thể đang đọc file: chờ rồi thử lại, không bỏ dở
        try:
            os.replace(tmp_path, cache_path)
            return
        except PermissionError:
            time.sleep(2)
    raise PermissionError(f"Không ghi được {cache_path}: file đang bị chương trình khác giữ")


segmenter, seg_name = ns["make_phobert_segmenter"]()
print("Tách từ bằng:", seg_name, flush=True)
t0, done = time.time(), 0
try:
    for start in range(0, len(todo), SAVE_EVERY):
        ids = todo[start:start + SAVE_EVERY]
        texts, owner = [], []
        for n, i in enumerate(ids):
            for chunk in ns["doc_chunks"](df.at[i, "lyrics"], CFG.PHOBERT_CHUNK_LINES, CFG.PHOBERT_MAX_CHUNKS):
                texts.append(segmenter(chunk))
                owner.append(n)
        vecs, owner = ns["phobert_encode"](texts), np.array(owner)
        for n, i in enumerate(ids):
            cache[keys[i]] = vecs[owner == n].mean(axis=0).astype(np.float16)
        save()
        done += len(ids)
        rate = done / (time.time() - t0)
        print(f"[{time.strftime('%H:%M')}] đã lưu {done:,}/{len(todo):,} bài của lần chạy này | {rate:.2f} bài/giây | "
              f"còn khoảng {(len(todo) - done) / rate / 3600:.1f} giờ", flush=True)
except KeyboardInterrupt:
    print("\nĐã dừng theo yêu cầu. Phần đã lưu được giữ nguyên; chạy lại lệnh này để tính tiếp.", flush=True)
    sys.exit(0)
print("XONG: mọi bài đã có vector." if "--limit" not in sys.argv else "Xong phần thử.", flush=True)
