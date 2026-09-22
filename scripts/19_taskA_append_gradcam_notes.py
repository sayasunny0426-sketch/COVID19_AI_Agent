"""Append the researcher's qualitative reading to an existing gradcam_notes.md.

Append-only by construction: the file is read, the new text is added at the end (under the
existing "## 総括（読影）" heading when there is one), and the result is written back. Nothing
is deleted or rewritten, and no other Grad-CAM artefact is touched -- gradcam_run_meta.json,
gradcam_selection.csv, the panels and the individual images are not opened for writing.

No model is retrained, no prediction is recomputed and no threshold is re-derived here.

Usage:
    python scripts/19_taskA_append_gradcam_notes.py \
        --notes "<...>/06_GradCAM/AIagent_taskA_primary_test/gradcam_notes.md"
    # to append a different text:
    python scripts/19_taskA_append_gradcam_notes.py --notes <...> --text-file my_reading.md
"""
import argparse
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

HEADING = "## 総括（読影）"
MARKER = "### 読影総括（研究者記入）"

DEFAULT_TEXT = """事前に規定したTest症例15例（TP 4例、FP 4例、TN 4例、FN 3例）についてGrad-CAMを確認した。

TPでは、肺門周囲から下肺野を含む胸郭内に比較的強いattentionが認められ、モデルが高リスク症例の判定において臨床的に妥当な胸部領域を参照している可能性が示唆された。

FPでも胸郭内に強いattentionを認めたが、その分布は必ずしも死亡リスクに関連する肺病変に特異的とは限らなかった。

TNでは明瞭な局所的attentionは乏しく、全体に弱い反応を示す症例が多かった。

一方、FNの一部では肺野中央よりも画像辺縁付近に強いattentionが認められ、モデルが病変とは直接関係しない周辺構造や高コントラスト領域を利用した可能性が考えられた。

以上より、本モデルは特にTP症例では妥当な胸郭内領域に注意を向ける傾向を示した一方、誤分類症例の一部では非病変領域へのattentionも認められた。

なお、Grad-CAMは定性的な説明手法であり、強調領域が予測の因果的根拠であることを示すものではない。また、本所見は4×4パネルおよび生成されたGrad-CAM画像に基づく定性的評価であり、特定の画像診断所見との厳密な対応を示すものではない。"""

PANEL_NOTE = """本所見は、TP 4例・FP 4例・TN 4例・FN 3例の計15例からなる4×4パネル（FNはTest setに3例しか存在しなかったため全3例を採用し、他群からの補充は行っていない。1セルは空欄）に基づく。"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", required=True, help="existing gradcam_notes.md")
    ap.add_argument("--text-file", default=None, help="file with the text to append")
    ap.add_argument("--keep-heading", action="store_true",
                    help="append the text with its own '## 総括（読影）' heading even if the "
                         "file already has that heading (default: add it as a subsection)")
    ap.add_argument("--no-footer", action="store_true",
                    help="do not add the traceability footer line")
    ap.add_argument("--allow-duplicate", action="store_true",
                    help="append again even though a reading summary is already present")
    args = ap.parse_args()

    notes = Path(args.notes)
    if not notes.exists():
        print(f"STOP: notes file not found: {notes}")
        return 1
    original = notes.read_text(encoding="utf-8")
    if not original.strip():
        print(f"STOP: {notes} is empty; refusing to write into it")
        return 1
    before_hash = sha256(notes)
    before_lines = len(original.splitlines())

    if MARKER in original and not args.allow_duplicate:
        print(f"STOP: a reading summary ({MARKER}) is already present in {notes.name}. "
              f"Re-run with --allow-duplicate only if a second entry is intended.")
        return 1

    text = Path(args.text_file).read_text(encoding="utf-8") if args.text_file else DEFAULT_TEXT
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    block = []
    if HEADING in original and not args.keep_heading:
        block.append(f"{MARKER}")          # the H2 already exists; add a dated subsection
    else:
        block.append(HEADING)
        block.append("")
        block.append(MARKER)
    block += ["", PANEL_NOTE, "", text.strip()]
    if not args.no_footer:
        block += ["", f"（追記日時 {timestamp}／追記前 gradcam_notes.md SHA256 "
                      f"`{before_hash}`／追記のみで既存記載は削除していない）"]
    appended = "\n".join(block) + "\n"

    updated = original + ("" if original.endswith("\n") else "\n") + "\n" + appended
    if not updated.startswith(original.rstrip("\n")):
        print("STOP: internal check failed (existing content would not be preserved)")
        return 1
    notes.write_text(updated, encoding="utf-8")

    after = notes.read_text(encoding="utf-8")
    ok = after.startswith(original.rstrip("\n")) and MARKER in after
    after_hash = sha256(notes)
    print(f"file            : {notes}")
    print(f"updated (local) : {timestamp}")
    print(f"lines           : {before_lines} -> {len(after.splitlines())} (+{len(after.splitlines()) - before_lines})")
    print(f"sha256 before   : {before_hash}")
    print(f"sha256 after    : {after_hash}")
    print(f"existing content preserved : {ok}")
    print("\nTask A Grad-CAM読影総括追記完了" if ok else "\nWARNING: verification failed")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
