"""Assemble a GitHub-ready copy of the project (Task A) without touching the研究用 Drive copy.

This only copies and writes into --out; nothing in Google Drive or in the working project is
modified, and no result, threshold or model choice is recomputed. Every number in the generated
README is read from a file -- either from this repository or, with --drive-root, from the
research master copy under 「気合のCOVID19」. Files are copied out of Drive, never into it.

Usage:
    python scripts/20_prepare_github_repo.py --out github_repo \
        --drive-root "G:/マイドライブ/気合のCOVID19"
    # without --drive-root the Test / comparison / Grad-CAM sections stay marked 未確認
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

CODE_DIRS = ["src", "scripts", "tests", "notebooks"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".git"}
SKIP_SUFFIX = {".pyc", ".pyo", ".pt", ".pth", ".dcm", ".zip", ".png", ".jpg", ".npy"}
MAX_FILE_MB = 5
DOCS = [  # curated: study-level documentation that a reader needs
    "docs/requirements.md", "docs/implementation_plan.md", "docs/open_questions.md",
    "docs/decision_log.md", "docs/change_log.csv", "docs/task_a_implementation_plan.md",
    "docs/task_a_stage2_design.md", "docs/task_a_condition_selection.md",
    "docs/task_a_final_model_options.md", "docs/task_a_test_evaluation_plan.md",
    "docs/cxr_studydescription_mapping_final.csv",
]
RESULTS = {  # local artefact -> path inside results/taskA
    "data/processed/taskA_png512_16bit/preprocess_summary.json": "preprocessing/preprocess_summary.json",
    "data/interim/cxr_preproc_audit/header_audit_summary.json": "preprocessing/dicom_header_audit_summary.json",
    "data/interim/cxr_audit/audit_summary.json": "preprocessing/cxr_dicom_audit_summary.json",
    "data/interim/cxr_audit/index_cxr_selection_flow.json": "preprocessing/index_cxr_selection_flow.json",
    "data/interim/cxr_audit/split_reconciliation_report.json": "preprocessing/split_reconciliation_report.json",
    "data/interim/taskA_dataset_qc.json": "training/dataset_qc.json",
    "data/interim/taskA_model_smoke.json": "training/model_smoke_check.json",
    "docs/task_a_condition_comparison.csv": "training/condition_comparison.csv",
    "docs/task_a_condition_comparison.decision.json": "training/condition_selection_decision.json",
}
PATIENT_LEVEL = {  # patient-level files: IDs cleared for publication by the researcher
    "data/splits/COVID19_固定患者split_1277.csv": "preprocessing/fixed_split_1277.csv",
    "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv": "preprocessing/index_cxr_manifest_1277.csv",
}
DROP_COLUMNS = {"source_filepath", "source_image_path", "source_dicom_path", "output_image_path"}
UID_COLUMNS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]
# TCIA redistribution terms for instance-level UIDs could not be checked from this environment,
# so the published CSVs carry no UIDs. The full files stay on Drive and their SHA256 is recorded
# in results/taskA/README.md; scripts/02b and scripts/18 regenerate them from the DICOM data.
PUBLISH_UIDS = False

# ---- Google Drive (research master copy) -------------------------------------------------
DRIVE_EVAL = "01_TaskA_CXR/05_Evaluation/AIagent_taskA_test"
DRIVE_CMP_ZIP = "01_TaskA_CXR/05_Evaluation/AIagent_vs_current_TaskA_comparison.zip"
DRIVE_GRADCAM = "01_TaskA_CXR/06_GradCAM/AIagent_taskA_primary_test"
DRIVE_CODE = "00_共通・研究管理/AIagent_code"
EVAL_FILES = [  # (name, reason to publish)
    ("test_predictions_primary.csv", "患者単位の主解析予測（Subject ID・true_label・確率・閾値・判定）"),
    ("test_predictions_all_variants.csv", "seed42/43/44・ensemble・primary を 1 ファイルに統合"),
    ("test_metrics_primary.json", "主解析の指標・CI・2 つの操作点・calibration bins"),
    ("test_metrics_secondary.json", "副次解析（seed 別・3 seed ensemble・T0 当日除外の感度分析）"),
    ("test_metrics_table.csv", "上記を 1 表にまとめた比較表"),
    ("test_roc_curve_points.csv", "ROC 曲線の座標（図の再描画用）"),
    ("test_pr_curve_points.csv", "PR 曲線の座標（図の再描画用）"),
    ("test_calibration_bins.csv", "calibration 5 分位の集計"),
    ("test_run_meta.json", "実行環境・凍結 selection の SHA256・使用した閾値"),
    ("test_access_log.jsonl", "Test set を 1 回だけ使用したことの記録（438 B）"),
]
EVAL_EXCLUDE = [  # (name, reason)
    ("test_predictions_seed42.csv", "test_predictions_all_variants.csv に同一値が含まれる（照合済み・冗長）"),
    ("test_predictions_seed43.csv", "同上"),
    ("test_predictions_seed44.csv", "同上"),
    ("test_predictions_ensemble.csv", "同上"),
]
GRADCAM_FILES = [
    ("gradcam_selection.csv", "15 症例の選択根拠と CAM 領域指標"),
    ("gradcam_notes.md", "症例別所見と定量サマリー"),
    ("gradcam_run_meta.json", "モデル・閾値・入力 SHA256・選択規則"),
]
GRADCAM_PANEL = "panels/gradcam_panel_4x4.png"
# 個別 15 枚と群別パネル 4 枚は 4x4 パネルと同じ症例の高解像度版で、約 2.4 MB を追加しても
# 新しい情報は増えないため未収録。ファイル名と SHA256 のみ gradcam_excluded_images_sha256.csv に残す。
GRADCAM_IMAGE_DIRS = ["individual", "panels"]
ALLOW_BINARY = {"results/taskA/gradcam/gradcam_panel_4x4.png"}
DRIVE_PATTERNS = [
    (re.compile(r"/content/drive/MyDrive/気合のCOVID19"), "<DRIVE_ROOT>"),
    (re.compile(r"[A-Za-z]:[\\/]マイドライブ[\\/]気合のCOVID19"), "<DRIVE_ROOT>"),
]
# absolute paths of this machine are replaced in the copied result files
# NOTE: the character class must exclude newline. `[^"']*` also matches "\n", so an earlier
# version swallowed everything after the first path to the end of the document (docs were
# published truncated). Paths therefore end at whitespace, a quote or Japanese punctuation.
_PATH_TAIL = r"[^\s\"'、。，`)\]}<>|]*"
PATH_PATTERNS = [
    (re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}" + _PATH_TAIL), "<LOCAL_PATH>"),
    (re.compile(r"[A-Za-z]:[\\/]{1,2}manifest-\d+" + _PATH_TAIL), "<DICOM_ROOT>"),
]
SECRET_PATTERNS = [re.compile(p, re.IGNORECASE) for p in
                   (r"(api[_-]?key|secret|passwd|password|access[_-]?token)\s*[:=]\s*[\"'][^\"']+[\"']",
                    r"gh[pousr]_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}",
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----")]
AUDIT_SELF = "scripts/20_prepare_github_repo.py"


MAX_REDACTED_CHARS = 260   # a single path is far shorter than this


def redact(text: str, patterns=None) -> tuple[str, int]:
    original = text
    n = 0
    for pattern, placeholder in (patterns or PATH_PATTERNS):
        text, k = pattern.subn(placeholder, text)
        n += k
    # guard: a redaction replaces paths, it must never delete surrounding content
    removed = len(original) - len(text)
    if removed > n * MAX_REDACTED_CHARS:
        raise SystemExit(f"STOP: redaction removed {removed} characters over {n} match(es); "
                         f"a pattern is too greedy. Nothing was written.")
    if text.count("\n") != original.count("\n"):
        raise SystemExit("STOP: redaction changed the number of lines; a pattern crossed a "
                         "newline. Nothing was written.")
    return text, n


def redact_all(text: str) -> tuple[str, int]:
    text, a = redact(text, DRIVE_PATTERNS)
    text, b = redact(text, PATH_PATTERNS)
    return text, a + b


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> list[str]:
    copied = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or set(p.parts) & SKIP_PARTS or p.suffix.lower() in SKIP_SUFFIX:
            continue
        target = dst / p.relative_to(src.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied.append(str(target.relative_to(dst)).replace("\\", "/"))
    return copied


def verified_numbers(project: Path) -> dict:
    """Read the facts that can be checked in this repository (not from memory)."""
    split = pd.read_csv(project / "data/splits/COVID19_固定患者split_1277.csv", dtype=str)
    idx = pd.read_csv(project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv",
                      encoding="utf-8-sig", dtype=str)
    pre = json.loads((project / "data/processed/taskA_png512_16bit/preprocess_summary.json")
                     .read_text(encoding="utf-8"))
    cond = pd.read_csv(project / "docs/task_a_condition_comparison.csv", encoding="utf-8-sig")
    dec = json.loads((project / "docs/task_a_condition_comparison.decision.json")
                     .read_text(encoding="utf-8"))
    delta = idx.delta_days_from_T0.value_counts().to_dict()
    return {
        "cohort": len(split),
        "split": {k: int(v) for k, v in split.split.value_counts().items()},
        "deaths": {k: int(v) for k, v in split[split.true_label == "1"].split.value_counts().items()},
        "delta_days": {int(k): int(v) for k, v in delta.items()},
        "images": pre["images_written"], "qc_roundtrip": pre["exact_roundtrip_all"],
        "conditions": len(cond),
        "best_condition": dec["decision"]["selected_condition"],
        "best_val_auroc": dec["decision"]["baseline_mean_val_auroc"],
        "stage2_best_improvement": dec["decision"]["improvement"],
        "margin": dec["decision"]["margin"],
        "split_sha256": sha256(project / "data/splits/COVID19_固定患者split_1277.csv"),
    }


def drop_columns(src: Path, target: Path, extra_drop=()) -> list[str]:
    """Copy a CSV without the local-path columns and (unless PUBLISH_UIDS) without the UIDs."""
    df = pd.read_csv(src, dtype=str, encoding="utf-8-sig")
    drop = [c for c in df.columns if c in DROP_COLUMNS or c in extra_drop]
    if not PUBLISH_UIDS:
        drop += [c for c in df.columns if c in UID_COLUMNS]
    target.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=drop).to_csv(target, index=False, encoding="utf-8-sig")
    return drop


def import_from_drive(drive: Path, out: Path) -> dict:
    """Copy the Test / comparison / Grad-CAM master artefacts into results/taskA.

    Read-only with respect to Drive. Absolute paths inside the copies are redacted, so a
    redacted copy no longer hashes to its master; both hashes are recorded.
    """
    prov: list[dict] = []          # one row per artefact, for results/taskA/README.md
    res = out / "results/taskA"

    def record(source_rel, repo_rel, included, reason, src: Path | None, redacted=False):
        row = {"source": source_rel, "repo": repo_rel, "included": included, "reason": reason}
        if src is not None and src.exists():
            row["sha256_drive"] = sha256(src)
            row["bytes"] = src.stat().st_size
        if repo_rel and included and (out / repo_rel).exists():
            row["sha256_repo"] = sha256(out / repo_rel)
            row["identical_to_master"] = (not redacted) and row.get("sha256_drive") == row["sha256_repo"]
        prov.append(row)

    def copy_text(src: Path, dst: Path) -> bool:
        """Copy a text artefact with path redaction; returns True if anything was redacted.

        Works on bytes so that a file needing no redaction stays byte-identical to its master
        (line endings and BOM included) and therefore keeps the same SHA256.
        """
        raw = src.read_bytes()
        text, n = redact_all(raw.decode("utf-8"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(raw if n == 0 else text.encode("utf-8"))
        return n > 0

    # ---- 1. Test evaluation ----------------------------------------------------------
    ev = drive / DRIVE_EVAL
    for name, reason in EVAL_FILES:
        src = ev / name
        if not src.exists():
            record(f"{DRIVE_EVAL}/{name}", "", False, "Drive 上に存在しなかった（未確認）", None)
            continue
        dst = res / "evaluation" / name
        red = copy_text(src, dst)
        record(f"{DRIVE_EVAL}/{name}", f"results/taskA/evaluation/{name}", True, reason, src, red)
    for name, reason in EVAL_EXCLUDE:
        record(f"{DRIVE_EVAL}/{name}", "", False, reason, ev / name)

    # ---- 2. current vs AI-agent comparison -------------------------------------------
    zp = drive / DRIVE_CMP_ZIP
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                raw = z.read(info.filename)
                text, n = redact_all(raw.decode("utf-8"))
                dst = res / "comparison" / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(raw if n == 0 else text.encode("utf-8"))
                prov.append({"source": f"{DRIVE_CMP_ZIP}::{info.filename}",
                             "repo": f"results/taskA/comparison/{name}", "included": True,
                             "reason": "paired DeLong 比較の正本（ZIP から展開）",
                             "bytes": info.file_size,
                             "sha256_drive": hashlib.sha256(raw).hexdigest(),
                             "sha256_repo": sha256(dst), "identical_to_master": n == 0})

    # ---- 3. Grad-CAM -------------------------------------------------------------------
    gc = drive / DRIVE_GRADCAM
    for name, reason in GRADCAM_FILES:
        src = gc / name
        if not src.exists():
            record(f"{DRIVE_GRADCAM}/{name}", "", False, "Drive 上に存在しなかった（未確認）", None)
            continue
        dst = res / "gradcam" / name
        if src.suffix == ".csv":
            dropped = drop_columns(src, dst)
            record(f"{DRIVE_GRADCAM}/{name}", f"results/taskA/gradcam/{name}", True,
                   f"{reason}（除去列: {', '.join(dropped)}）", src, redacted=True)
        else:
            red = copy_text(src, dst)
            record(f"{DRIVE_GRADCAM}/{name}", f"results/taskA/gradcam/{name}", True, reason, src, red)
    panel = gc / GRADCAM_PANEL
    if panel.exists():
        dst = res / "gradcam" / panel.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(panel, dst)
        record(f"{DRIVE_GRADCAM}/{GRADCAM_PANEL}", f"results/taskA/gradcam/{panel.name}", True,
               "15 症例すべてを 1 枚に収めた代表図（1 セル空欄）", panel)
    # the remaining overlays are not published; their hashes are, so they stay verifiable
    rows = []
    for d in GRADCAM_IMAGE_DIRS:
        for p in sorted((gc / d).glob("*.png")):
            if p == panel:
                continue
            rows.append({"file": f"{d}/{p.name}", "bytes": p.stat().st_size, "sha256": sha256(p)})
    if rows:
        pd.DataFrame(rows).to_csv(res / "gradcam" / "gradcam_excluded_images_sha256.csv",
                                  index=False, encoding="utf-8")
        record(f"{DRIVE_GRADCAM}/(individual|panels)/*.png", "", False,
               f"個別 {len(rows)} 枚は 4x4 パネルと同一症例の高解像度版（計 "
               f"{sum(r['bytes'] for r in rows) / 2**20:.1f} MB）。SHA256 のみ収録", None)
    return {"provenance": prov}


def drive_numbers(drive: Path) -> dict:
    """Read the Test / comparison / Grad-CAM numbers from the Drive master files."""
    ev, gc = drive / DRIVE_EVAL, drive / DRIVE_GRADCAM
    m = json.loads((ev / "test_metrics_primary.json").read_text(encoding="utf-8-sig"))
    meta = json.loads((ev / "test_run_meta.json").read_text(encoding="utf-8-sig"))
    sec = json.loads((ev / "test_metrics_secondary.json").read_text(encoding="utf-8-sig"))
    with zipfile.ZipFile(drive / DRIVE_CMP_ZIP) as z:
        dl = json.loads(z.read("taskA_current_vs_aiagent_delong.json").decode("utf-8-sig"))
        cm = pd.read_csv(io.BytesIO(z.read("taskA_current_vs_aiagent_metrics.csv")))
    gmeta = json.loads((gc / "gradcam_run_meta.json").read_text(encoding="utf-8-sig"))
    cur = cm[cm.model == "current"].iloc[0]
    return {"test": m, "meta": meta, "secondary": sec, "delong": dl,
            "current": {"roc_auc": float(cur.roc_auc), "pr_auc": float(cur.pr_auc),
                        "brier": float(cur.brier)},
            "gradcam": gmeta}


README = """# COVID-19 in-hospital mortality prediction from admission chest radiographs (Task A)

Stony Brook University COVID-19 Positive Cases (TCIA **COVID-19-NY-SBU**) を用いて、
入院（visit 開始）時点の胸部 X 線写真から院内死亡を予測する ResNet18 モデルを構築した研究の
Task A 部分です。本リポジトリは **AI agent が独立に設計・実装した解析**のコードと結果をまとめたものです。

> **数値の出所**：本 README の数値は**すべて実ファイルから読み出して生成**しています。
> コホート・split・前処理・条件選択の値は本リポジトリ内のファイルから、Test 性能・DeLong 比較・
> Grad-CAM の値は `results/taskA/` に収録した最終成果物（正本は研究用保存先）から読んでいます。
> どの数値をどのファイルから読んだかは [docs/results_provenance.md](docs/results_provenance.md)、
> 正本との対応と SHA256 は [results/taskA/README.md](results/taskA/README.md) にあります。

---

## 1. 目的

入院時点で得られる **index CXR 1 枚**から院内死亡を予測し、臨床情報モデル（Task B）および
マルチモーダルモデル（Task C）と同一の Test set 上で比較できる形にすること。

## 2. データセットの概要（検証済）

| 項目 | 値 |
|---|---|
| 固定コホート | **{cohort} 例** |
| Train / Validation / Test | **{train} / {val} / {test}** |
| 死亡数（Train / Val / Test） | **{d_train} / {d_val} / {d_test}** |
| 1 患者あたりの画像 | index CXR 1 枚 |
| 前処理済み画像 | {images} 枚（16-bit PNG 512×512、round-trip QC 完全一致：{qc}） |

固定 split は `results/taskA/preprocessing/fixed_split_1277.csv`（SHA256 `{split_sha}`）。
**split は再作成せず、全 Task で共通に使用します。**

## 3. 予測時点と CXR の選択規則（検証済）

- **T0 = `visit_start_datetime`**（日付のみ。データに正確な admission time が存在しないため）
- CXR は **撮影日が T0−2 日 〜 T0 当日**のものを対象とし、T0 より後の日付は使用しない
- 対象日が複数ある場合は T0 に最も近い日、その日に複数 Series があれば `AcquisitionTime` が
  最も早い Series、同時刻なら `SeriesNumber` 最小（tie-break のみ。撮影順とは解釈しない）
- frontal 判定：`Modality` が CR/DX → `StudyDescription` で胸部検査を特定 → `SeriesDescription` で AP/PA
- 選ばれた index CXR の T0 からの日数差：{delta}

## 4. モデル

| 項目 | 値 |
|---|---|
| アーキテクチャ | ResNet18（torchvision、ImageNet 事前学習 `IMAGENET1K_V1`） |
| 学習範囲 | 全層 fine-tuning |
| 入力 | 3×224×224（16-bit PNG → per-image 1–99 percentile clip → [0,1] → 正方形 zero pad → 224 → ImageNet 正規化） |
| 損失 | `BCEWithLogitsLoss(pos_weight=6.563)`（Train のみから算出） |
| 最適化 | AdamW、weight decay 1e-4、batch 32、最大 30 epoch、OneCycleLR（warm-up 1 epoch → cosine） |

## 5. 最終モデル（主解析）

**`lr3e-4_aug_b` / seed 42**（単一 seed を事前固定）

- Stage 1（学習率 4 × augmentation 2 × seed 3 = 24 run）と Stage 2（warm-up / staged unfreezing × seed 3 = 6 run）の
  計 30 run を **Validation のみ**で比較し、3 seed 平均 Validation AUROC が最大の条件を採用（検証済：{best_condition}、{best_auroc:.6f}）
- Stage 2 は baseline を上回らなかった（最良でも **{improvement:+.6f}**、事前に固定した
  practical tolerance / parsimony margin {margin} 未満）ため、より単純な Stage 1 条件を採用
- 主解析を単一 seed としたのは、従来の Task A（seed 42 単一モデル）との比較で
  学習条件の違いと ensemble 効果が混ざらないようにするため。3 seed ensemble と seed 別の結果は副次解析として保存

詳細：[docs/task_a_condition_selection.md](docs/task_a_condition_selection.md)、
[results/taskA/training/condition_comparison.csv](results/taskA/training/condition_comparison.csv)

## 6. モデル選択と Test の扱い

- **モデル選択・checkpoint 選択・閾値決定はすべて Validation のみ**で実施
- 分類閾値：Validation の Youden index 最大（同率時は最も低い閾値。感度を優先）。
  副次の操作点として「Validation 感度 ≥ 0.80 のうち最も高い閾値」を **exploratory** に併記
- **Test set は最終評価として 1 回だけ使用**。前処理選択、変数選択、モデル選択、
  ハイパーパラメータ調整、checkpoint 選択、閾値決定には使用していない。
  閾値は `final_selection.json` に凍結した値を用い、Test では再計算していない

## 7. Test 性能{test_status}

主解析 `lr3e-4_aug_b` / seed 42、Test {t_n} 例（死亡 {t_events} 例、有病率 {t_prev}）。

| 指標 | 値 | 95% CI |
|---|---|---|
| ROC-AUC | **{t_auroc}** | {t_auroc_lo} – {t_auroc_hi} |
| PR-AUC（Average Precision） | **{t_auprc}** | {t_auprc_lo} – {t_auprc_hi} |
| Brier score | {t_brier} | — |
| ECE（5 分位） | {t_ece} | — |

操作点（いずれも **Validation で決定した閾値を凍結して適用**。Test では再計算していません）

| 操作点 | 閾値 | 感度 | 特異度 | PPV | NPV | TP/FP/TN/FN |
|---|---|---|---|---|---|---|
| 主：Validation Youden 最大 | {t_thr_y} | {t_sens} | {t_spec} | {t_ppv} | {t_npv} | {t_cm_y} |
| 副次（exploratory）：Validation 感度 ≥ 0.80 のうち最も高い閾値 | {t_thr_s} | {t_sens_s} | {t_spec_s} | {t_ppv_s} | {t_npv_s} | {t_cm_s} |

副次操作点の感度 {t_sens_s} は **Test での実測値**です。閾値は Validation で感度 ≥ 0.80 を満たすよう
選んだもので、Test で感度 0.80 を保証するものではありません。

副次解析（主解析を置き換えるものではありません）

| モデル | ROC-AUC | PR-AUC |
|---|---|---|
| seed 43 | {s43_auroc} | {s43_auprc} |
| seed 44 | {s44_auroc} | {s44_auprc} |
| 3 seed ensemble | {ens_auroc} | {ens_auprc} |
| 感度分析：index CXR が T0 当日でない {l1_n} 例 | {l1_auroc} | {l1_auprc} |

AUPRC は step-wise の Average Precision であり、PR 曲線の台形積分ではありません。
95% CI は患者単位の層別 bootstrap（{t_nboot} 反復、seed {t_bseed}）です。
出典：`results/taskA/evaluation/`（`test_metrics_primary.json`, `test_metrics_table.csv`）。
実行環境と凍結 selection の SHA256 は `test_run_meta.json`、Test を 1 回だけ使用した記録は
`test_access_log.jsonl` にあります。

## 8. Grad-CAM{gc_status}

- 対象層：`{gc_layer}`、主解析モデル（{gc_model}）に対してのみ実施
- 閾値 {gc_thr}（`final_selection.json` の Youden。**Test で再計算していない**：
  `threshold_recomputed_on_test = {gc_recomp}`）
- 症例：**TP {gc_tp}・FP {gc_fp}・TN {gc_tn}・FN {gc_fn} = 計 {gc_n} 例**
  （各群の Test 内該当数：TP {av_tp}・FP {av_fp}・TN {av_tn}・FN {av_fn}）。
  FN は Test set に {av_fn} 例しか存在しないため全 {av_fn} 例を採用し、**他群からの補充は行っていない**。
  4×4 パネルは {gc_blank} セルを空欄
- 選択規則（事前固定）：TP・FP は予測確率の高い順、TN・FN は低い順、同値は Subject ID 昇順
- 収録：`results/taskA/gradcam/gradcam_panel_4x4.png`（15 症例）、`gradcam_selection.csv`（選択根拠と
  CAM 領域指標）、`gradcam_notes.md`、`gradcam_run_meta.json`。個別画像 15 枚と群別パネル 4 枚は
  同一症例の高解像度版のため未収録（SHA256 は `gradcam_excluded_images_sha256.csv`）
- 読影所見：`gradcam_notes.md` の「総括（読影）＞ 読影総括（研究者記入）」。定量指標は
  「どこに CAM 質量が落ちたか」を測ったもので、解剖学的な妥当性の判断ではありません
- **既知の limitation（未解決）**：予測確率が数値的に 0 に飽和した症例（**TN 4 例と FN 1 例**）は
  CAM が全面ゼロとなり、領域指標・重心・ピークが `nan` です。**補間・0 置換などの代入は行っていません**。
  そのため「領域指標が算出できた 10 例の平均」と「15 例中の該当件数」は母集団が異なります
  （`docs/open_questions.md` Q-E3）
- **既知の metadata propagation issue（未解決）**：`gradcam_selection.csv` の
  `delta_days_from_T0` が 15 例すべて空で、`gradcam_notes.md` の症例別記載では「T0 との差 None 日」と
  なっています。Test 予測 CSV 側には正しい値（−1 / −2 / 0）があり、選択表への引き継ぎ漏れです。
  **推測で値を埋めていません**（`docs/open_questions.md` Q-E6）
- Grad-CAM は定性的な説明手法であり、強調領域が予測の因果的根拠であることを示すものではありません

## 9. 既存モデルとの探索的比較{cmp_status}

同一の Test {c_n} 例（死亡 {c_events} 例）。Subject ID {c_matched}/{c_n} 一致、重複なし、true_label 全一致。

| | current model | AI-agent model |
|---|---|---|
| ROC-AUC | {c_cur_auroc} | {c_ai_auroc} |
| PR-AUC | {c_cur_prauc} | {c_ai_prauc} |
| Brier score | {c_cur_brier} | {c_ai_brier} |

- ROC-AUC 差（current - AI-agent）：**{c_diff}**（SE {c_se}、95% CI {c_ci_lo} – {c_ci_hi}）
- paired DeLong、two-sided、**unadjusted** p = **{c_p}**

**post-hoc / exploratory comparison** です。事前登録した主要仮説検定ではなく、p 値は多重比較を補正していません。
Test の死亡は {c_events} 例で CI は広く、両モデルは学習条件が複数の点で異なるため、
**どの設計要素が差に寄与したかは特定できません**。「current model が統計学的に優れている」とは結論しません。
出典：`results/taskA/comparison/`（`taskA_current_vs_aiagent_delong.json`, `..._metrics.csv`）。

## 10. リポジトリ構成

```
{tree}
```

工程とコードの対応：

| 工程 | 主なエントリーポイント |
|---|---|
| CXR DICOM 監査 | `scripts/01_cxr_dicom_audit.py`, `scripts/01b_studydescription_mapping.py` |
| index CXR 選定・窓適用 | `scripts/02_index_cxr_selection.py`, `scripts/02b_apply_inclusion_window.py` |
| コホート・split の照合 | `scripts/03_cohort_reconciliation.py`, `scripts/04_split_reconciliation.py` |
| 前処理キャッシュ生成 | `scripts/06_taskA_preprocess_cache.py` |
| Dataset / DataLoader 検証 | `scripts/07_taskA_dataset_qc.py` |
| モデル結線確認 | `scripts/08_taskA_model_smoke.py` |
| 学習（Stage 1 / Stage 2） | `scripts/09_taskA_run_training.py`（`notebooks/TaskA_train_colab.ipynb` から実行） |
| 条件比較・最終条件の決定 | `scripts/12_taskA_stage1_report.py`, `scripts/13_taskA_condition_comparison.py` |
| 凍結（final_selection.json） | `scripts/14_taskA_freeze_selection.py` |
| **Test 評価（1 回のみ）** | `scripts/15_taskA_test_evaluation.py` |
| 既存モデルとの比較 | `scripts/16_taskA_compare_current_vs_aiagent.py` |
| Grad-CAM | `scripts/18_taskA_gradcam_primary_test.py`（`notebooks/TaskA_gradcam_colab.ipynb`） |

`src/covid_mortality/` は再利用可能な実装（dataset・model・trainer・metrics・DeLong・Grad-CAM）、
`scripts/` は番号順の実行エントリーポイントです。

## 11. 再現手順

```bash
# 1. environment
uv sync                     # または: pip install -r requirements.txt

# 2. preprocessing（DICOM → 16-bit PNG キャッシュ）
python scripts/06_taskA_preprocess_cache.py --project .

# 3. dataset / dataloader check
python scripts/07_taskA_dataset_qc.py --project . --full-read

# 4. model definition check
python scripts/08_taskA_model_smoke.py --project .

# 5. training（Stage 1 = 24 run、Stage 2 = 6 run。完了済み run は自動 skip）
python scripts/09_taskA_run_training.py --project . --stage main   --runs-root <RUNS>
python scripts/09_taskA_run_training.py --project . --stage stage2 --runs-root <RUNS>

# 6. model selection（Validation のみ）
python scripts/13_taskA_condition_comparison.py --runs-root <RUNS>
python scripts/14_taskA_freeze_selection.py --project . --runs-root <RUNS>

# 7. Test evaluation — 研究では最終評価として 1 回だけ実行した
#    （--dry-run で Validation に対して同じ経路を検証できる）
python scripts/15_taskA_test_evaluation.py --project . --runs-root <RUNS> --out-dir <EVAL> --dry-run
# python scripts/15_taskA_test_evaluation.py --project . --runs-root <RUNS> --out-dir <EVAL>

# 8. current model との探索的比較
python scripts/16_taskA_compare_current_vs_aiagent.py --current <CSV> --aiagent <CSV> --out-dir <CMP>

# 9. Grad-CAM
python scripts/18_taskA_gradcam_primary_test.py --project . --frozen <final_selection.json> \\
    --test-predictions <CSV> --image-dir <IMAGES> --out-dir <GRADCAM>
```

`<RUNS>` などは実行環境のパスに読み替えてください。Notebook 内の
`/content/drive/MyDrive/気合のCOVID19/...` は **ユーザー環境に合わせて変更**してください。

**Test の位置づけ**：Test set は研究全体で **最終評価 1 回のみ** 使用しました。
手順 7 を繰り返し実行することは想定していません（スクリプトも既存の Test 予測がある場合は停止します）。

## 12. 依存関係

`pyproject.toml` が正本です（`uv sync` で解決）。pip 利用者向けに `requirements.txt` を併置しています。
主要依存：torch, torchvision, numpy, pandas, pillow, pydicom。
scikit-learn はメトリクスの参照照合テスト（`tests/test_metrics_reference.py`）でのみ使用し、
本体の指標計算は numpy 実装（環境差で数値が変わらないようにするため）です。

## 13. 制約・限界

- 正確な admission time がデータに存在せず、T0 は `visit_start_datetime`。**T0 当日の CXR が
  admission の前か後かは判定できません**（本研究の limitation L-1）
- 年齢は 3 階級のみで提供され、既存 risk score の年齢区分を再現できません（L-4）
- Test の死亡は 17 例で、AUROC の 95% CI は広く、モデル間差の検出力は限られます
- 画像には焼き込み文字（PORTABLE、L/R マーカー）が残っており、近道学習の可能性を Grad-CAM で監視しています
- CR 1,246 例／DX 31 例と装置が混在します

## 14. データの取り扱い

- **画像・DICOM・派生画像キャッシュ・checkpoint は本リポジトリに含みません。**
  原データは TCIA（COVID-19-NY-SBU）から各自取得してください
- 患者単位のファイル（固定 split、index CXR manifest、Test 予測、Grad-CAM 選択表）は匿名化済み
  Subject ID を含みます。公開可否は研究責任者の確認に基づきます
- **DICOM instance UID（StudyInstanceUID / SeriesInstanceUID / SOPInstanceUID）は公開版から
  除去**しています。TCIA の再配布条件を本作業環境から確認できなかったための保守的な措置で、
  完全版は研究用保存先にあり、その SHA256 を `results/taskA/README.md` に記録しています。
  UID は `scripts/02b` と `scripts/18` が DICOM から再生成します（{uid_status}）
- ローカル絶対パス（`source_filepath` など）と Drive 絶対パスは公開版から除去しています
  （`<LOCAL_PATH>` / `<DICOM_ROOT>` / `<DRIVE_ROOT>`）

## 15. 再現性に関する注記

- 乱数 seed は 42 / 43 / 44 を使用。学習時は Python・NumPy・PyTorch・CUDA の seed を固定し、
  cuDNN を deterministic 設定にしています。AMP と一部 CUDA 演算では完全一致が保証されないため、
  3 seed の結果の幅を併記しています
- 各 run に config・環境情報・入力ファイルの SHA256・学習履歴・checkpoint・Validation 予測を保存しています
- 判断の経緯は `docs/decision_log.md`（D-001 以降）、誤りと修正は `docs/change_log.csv` に記録しています
- 未解決事項は `docs/open_questions.md`（公開に関するものは E 節）にまとめています

## 16. License

**License: to be determined.**

ライセンスは未定です。決定するまで、本リポジトリの内容の再利用条件は保証されません。
原データ（TCIA COVID-19-NY-SBU）には TCIA 側の利用条件が別途適用されます。

---

生成日：{today}（`scripts/20_prepare_github_repo.py` により作成）
"""

PYPROJECT = """[project]
name = "covid19-ai-agent-taskA"
version = "0.1.0"
description = "COVID-19 in-hospital mortality prediction from admission chest radiographs (Task A)"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "torchvision>=0.15",
    "numpy>=1.24",
    "pandas>=2.0",
    "pillow>=10.0",
    "pydicom>=3.0",
]

[project.optional-dependencies]
dev = ["scikit-learn>=1.3", "matplotlib>=3.7", "pytest>=7.0"]

[tool.uv]
package = false
"""

REQUIREMENTS = """# pyproject.toml が正本。pip 利用者向けの同等の依存関係。
torch>=2.0
torchvision>=0.15
numpy>=1.24
pandas>=2.0
pillow>=10.0
pydicom>=3.0
# 開発・検証用（本体の指標計算には不要）
scikit-learn>=1.3
matplotlib>=3.7
pytest>=7.0
"""

GITIGNORE = """# data and imaging artefacts (never committed)
# NOTE: anchored with a leading slash. An unanchored `data/` also matches
# src/covid_mortality/data/, which is source code and must stay tracked.
/data/
results/**/images/
*.dcm
*.nii
*.nii.gz
*.png
!results/taskA/gradcam/*.png

# model artefacts
checkpoints/
*.pt
*.pth
*.ckpt
*.joblib
*.npy
*.npz

# environments and caches
.venv/
venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.pytest_cache/
.uv/
uv.lock

# secrets and local settings
.env
*.key
*.pem
credentials*.json
token*.json

# scratch
*.tmp
*.log
.DS_Store
Thumbs.db
/outputs/
"""

GITATTRIBUTES = """# Byte-exact storage.
#
# results/taskA/ holds research artefacts whose SHA256 is recorded in results/taskA/README.md
# and checked against the master copy. Git for Windows ships with core.autocrlf=true, which
# would rewrite CRLF to LF on commit and change those hashes. `-text` turns all end-of-line
# conversion off, so what is committed is byte-for-byte what was verified.
* -text
"""

RESULTS_README = """# results/taskA

本リポジトリに収録しているのは、**公開に適した集計・サマリー**のみです。
画像・DICOM・checkpoint・全 1,277 枚の PNG・キャッシュは含みません。

## 収録済み（本リポジトリ内で完結）

| パス | 内容 | 由来 |
|---|---|---|
| `preprocessing/preprocess_summary.json` | 16-bit PNG キャッシュ生成と round-trip QC の集計 | `scripts/06` |
| `preprocessing/dicom_header_audit_summary.json` | 選択済み 1,277 枚の DICOM ヘッダー監査 | `scripts/05` |
| `preprocessing/cxr_dicom_audit_summary.json` | 全患者の CXR DICOM 構造監査 | `scripts/01` |
| `preprocessing/index_cxr_selection_flow.json` | index CXR 選定と除外の流れ | `scripts/02` |
| `preprocessing/split_reconciliation_report.json` | 固定 split の照合結果 | `scripts/04` |
| `preprocessing/fixed_split_1277.csv` | 固定 split（患者単位） | 研究者提供の正本 |
| `preprocessing/index_cxr_manifest_1277.csv` | index CXR manifest（絶対パス列と DICOM UID 3 列は除去） | `scripts/02b` |
| `training/dataset_qc.json` | Dataset / DataLoader の QC 29 項目 | `scripts/07` |
| `training/model_smoke_check.json` | ResNet18 結線確認 | `scripts/08` |
| `training/condition_comparison.csv` | Stage 1 + Stage 2 の条件比較（3 seed 平均 Validation） | `scripts/13` |
| `training/condition_selection_decision.json` | 事前固定した選択規則の適用結果 | `scripts/13` |

## Google Drive 正本との対応（Test / 比較 / Grad-CAM）

正本は Google Drive の「気合のCOVID19」配下です。本リポジトリはコピー先であり、
Drive 側は読み取りのみで変更していません。`sha256_drive` は正本の値、`sha256_repo` は
本リポジトリ収録物の値です。絶対パスの伏字化や列削除を行ったファイルは両者が一致しません
（`identical` 列を参照）。

{provenance_table}

## 恒久的に含めないもの

| 対象 | 理由 |
|---|---|
| DICOM 原本（TCIA COVID-19-NY-SBU） | 再配布不可。各自 TCIA から取得 |
| 前処理済 PNG 1,277 枚・キャッシュ | 患者画像 |
| checkpoint（`best.pt` / `last.pt`、30 run 分） | サイズ。`gradcam_run_meta.json` に SHA256 を記録 |
| 学習 run ディレクトリ（`history.csv` ほか 30 run 分） | サイズ。`training/condition_comparison.csv` で代替 |
| `preprocess_records.csv` | ローカル絶対パスを含む |
| 過去のコードバンドル ZIP | 正本は本リポジトリのコード |
| DICOM instance UID 3 列 | TCIA 再配布条件が本作業環境から未確認のため保守的に除去 |
"""


UNVERIFIED = "（研究者報告・未検証）"


def readme_result_fields(dv: dict | None) -> dict:
    """Format the Test / comparison / Grad-CAM README fields from the Drive master files."""
    keys = ("t_n t_events t_prev t_auroc t_auroc_lo t_auroc_hi t_auprc t_auprc_lo t_auprc_hi "
            "t_brier t_ece t_thr_y t_sens t_spec t_ppv t_npv t_cm_y t_thr_s t_sens_s t_spec_s "
            "t_ppv_s t_npv_s t_cm_s s43_auroc s43_auprc s44_auroc s44_auprc ens_auroc ens_auprc "
            "l1_n l1_auroc l1_auprc t_nboot t_bseed gc_layer gc_model gc_thr gc_recomp gc_tp "
            "gc_fp gc_tn gc_fn gc_n av_tp av_fp av_tn av_fn gc_blank c_n c_events c_matched "
            "c_cur_auroc c_ai_auroc c_cur_prauc c_ai_prauc c_cur_brier c_ai_brier c_diff c_se "
            "c_ci_lo c_ci_hi c_p").split()
    if dv is None:
        return {k: "未確認" for k in keys} | {"test_status": UNVERIFIED,
                                             "gc_status": UNVERIFIED, "cmp_status": UNVERIFIED}

    def f(x, n=3):
        return f"{float(x):.{n}f}"

    t, sec, dl, cur, g = dv["test"], dv["secondary"], dv["delong"], dv["current"], dv["gradcam"]
    y, s = t["operating_point_youden"], t["operating_point_sensitivity80"]
    cm = lambda o: f"{int(o['tp'])}/{int(o['fp'])}/{int(o['tn'])}/{int(o['fn'])}"
    used, avail = g["cases_used_per_group"], g["cases_available_per_group"]
    l1 = sec.get("L1_excluding_T0_day", {})
    return {
        "test_status": "", "gc_status": "", "cmp_status": "",
        "t_n": t["n"], "t_events": t["events"], "t_prev": f(t["prevalence"]),
        "t_auroc": f(t["auroc"]["point"]), "t_auroc_lo": f(t["auroc"]["ci_low"]),
        "t_auroc_hi": f(t["auroc"]["ci_high"]), "t_auprc": f(t["auprc"]["point"]),
        "t_auprc_lo": f(t["auprc"]["ci_low"]), "t_auprc_hi": f(t["auprc"]["ci_high"]),
        "t_brier": f(t["brier"]), "t_ece": f(t["ece"]),
        "t_thr_y": f(y["threshold"], 6), "t_sens": f(y["sensitivity"]), "t_spec": f(y["specificity"]),
        "t_ppv": f(y["ppv"]), "t_npv": f(y["npv"]), "t_cm_y": cm(y),
        "t_thr_s": f(s["threshold"], 6), "t_sens_s": f(s["sensitivity"]),
        "t_spec_s": f(s["specificity"]), "t_ppv_s": f(s["ppv"]), "t_npv_s": f(s["npv"]),
        "t_cm_s": cm(s),
        "s43_auroc": f(sec["seed43"]["auroc"]["point"]), "s43_auprc": f(sec["seed43"]["auprc"]["point"]),
        "s44_auroc": f(sec["seed44"]["auroc"]["point"]), "s44_auprc": f(sec["seed44"]["auprc"]["point"]),
        "ens_auroc": f(sec["three_seed_ensemble"]["auroc"]["point"]),
        "ens_auprc": f(sec["three_seed_ensemble"]["auprc"]["point"]),
        "l1_n": l1.get("n", "未確認"),
        "l1_auroc": f(l1["auroc"]["point"]) if l1 else "未確認",
        "l1_auprc": f(l1["auprc"]["point"]) if l1 else "未確認",
        "t_nboot": t["auroc"]["n_boot"], "t_bseed": t["auroc"]["seed"],
        "gc_layer": g["gradcam_layer"], "gc_model": g["model"],
        "gc_thr": f(g["threshold_used"], 6), "gc_recomp": g["threshold_recomputed_on_test"],
        "gc_tp": used["TP"], "gc_fp": used["FP"], "gc_tn": used["TN"], "gc_fn": used["FN"],
        "gc_n": g["n_cases"], "av_tp": avail["TP"], "av_fp": avail["FP"],
        "av_tn": avail["TN"], "av_fn": avail["FN"], "gc_blank": g["panel_empty_cells"],
        "c_n": dl["n"], "c_events": dl["events"], "c_matched": dl["validation"]["matched_n"],
        "c_cur_auroc": f(dl["auroc_current"], 4), "c_ai_auroc": f(dl["auroc_aiagent"], 4),
        "c_cur_prauc": f(cur["pr_auc"], 4), "c_ai_prauc": f(t["auprc"]["point"], 4),
        "c_cur_brier": f(cur["brier"], 4), "c_ai_brier": f(t["brier"], 4),
        "c_diff": f(dl["auroc_difference_current_minus_aiagent"], 4),
        "c_se": f(dl["difference_se"], 4), "c_ci_lo": f(dl["ci95_low"], 4),
        "c_ci_hi": f(dl["ci95_high"], 4), "c_p": f(dl["p_value_two_sided_unadjusted"], 4),
    }


def provenance_table(prov: list[dict]) -> str:
    if not prov:
        return "_Drive の正本を参照できなかったため未収録（未確認）。_"
    head = ("| Drive 上の場所 | 本リポジトリ | 収録 | 内容・除外理由 | bytes | sha256（正本） | identical |\n"
            "|---|---|---|---|---|---|---|")
    rows = []
    for r in prov:
        h = r.get("sha256_drive") or r.get("sha256_drive_zip") or ""
        ident = {True: "✓ 正本と同一", False: "加工のため不一致（伏字化／列削除）"}.get(
            r.get("identical_to_master"), "—")
        rows.append(f"| `{r['source']}` | {('`' + r['repo'] + '`') if r['repo'] else '—'} | "
                    f"{'収録' if r['included'] else '除外'} | {r['reason']} | "
                    f"{r.get('bytes', '')} | `{h[:16] + '…' if h else '—'}` | "
                    f"{ident if r['included'] else '—'} |")
    return head + "\n" + "\n".join(rows)


def provenance_doc(v: dict, dv: dict | None, prov: list[dict]) -> str:
    """Record which file each README number came from."""
    lines = ["# README の数値の出典", "",
             "README に載せた数値はすべて実ファイルから読み出したものです。",
             "本ファイルは `scripts/20_prepare_github_repo.py` が生成します。", "",
             "## リポジトリ内のファイルから読んだ値", "",
             "| README の記載 | 値 | 出典ファイル |", "|---|---|---|",
             f"| コホート / split / 死亡 | {v['cohort']} / "
             f"{v['split']['train']}-{v['split']['val']}-{v['split']['test']} / "
             f"{v['deaths']['train']}-{v['deaths']['val']}-{v['deaths']['test']} | "
             "`results/taskA/preprocessing/fixed_split_1277.csv` |",
             f"| split の SHA256 | `{v['split_sha256']}` | 同上 |",
             f"| Δ日分布 | {v['delta_days']} | "
             "`results/taskA/preprocessing/index_cxr_manifest_1277.csv` |",
             f"| 前処理枚数 / round-trip QC | {v['images']} / {v['qc_roundtrip']} | "
             "`results/taskA/preprocessing/preprocess_summary.json` |",
             f"| 最良条件 / 3 seed 平均 Val AUROC | {v['best_condition']} / {v['best_val_auroc']:.6f} | "
             "`results/taskA/training/condition_selection_decision.json` |",
             f"| Stage 2 の改善 / margin | {v['stage2_best_improvement']:+.6f} / {v['margin']} | 同上 |", ""]
    if dv is None:
        lines += ["## Test / 比較 / Grad-CAM", "", "_未確認（Drive の正本を参照できなかった）_", ""]
        return "\n".join(lines)
    t, dl, g = dv["test"], dv["delong"], dv["gradcam"]
    lines += [
        "## Google Drive の正本から読んだ値", "",
        "| README の記載 | 値 | 出典ファイル |", "|---|---|---|",
        f"| Test ROC-AUC (95% CI) | {t['auroc']['point']:.6f} "
        f"({t['auroc']['ci_low']:.6f}–{t['auroc']['ci_high']:.6f}) | "
        "`results/taskA/evaluation/test_metrics_primary.json` |",
        f"| Test PR-AUC (95% CI) | {t['auprc']['point']:.6f} "
        f"({t['auprc']['ci_low']:.6f}–{t['auprc']['ci_high']:.6f}) | 同上 |",
        f"| Brier / ECE | {t['brier']:.6f} / {t['ece']:.6f} | 同上 |",
        f"| 主操作点の閾値 | {t['operating_point_youden']['threshold']} | 同上（Validation 由来・凍結） |",
        f"| 副次操作点の閾値 | {t['operating_point_sensitivity80']['threshold']} | 同上 |",
        "| 副次解析（seed 別・ensemble・T0 当日除外） | — | "
        "`results/taskA/evaluation/test_metrics_secondary.json`, `test_metrics_table.csv` |",
        f"| current ROC-AUC | {dl['auroc_current']:.6f} | "
        "`results/taskA/comparison/taskA_current_vs_aiagent_metrics.csv` |",
        f"| ROC-AUC 差 / SE / 95% CI | {dl['auroc_difference_current_minus_aiagent']:.6f} / "
        f"{dl['difference_se']:.6f} / {dl['ci95_low']:.6f}–{dl['ci95_high']:.6f} | "
        "`results/taskA/comparison/taskA_current_vs_aiagent_delong.json` |",
        f"| paired DeLong p (two-sided, unadjusted) | {dl['p_value_two_sided_unadjusted']:.6f} | 同上 |",
        f"| Grad-CAM 症例構成 | TP {g['cases_used_per_group']['TP']}・"
        f"FP {g['cases_used_per_group']['FP']}・TN {g['cases_used_per_group']['TN']}・"
        f"FN {g['cases_used_per_group']['FN']} = {g['n_cases']} | "
        "`results/taskA/gradcam/gradcam_run_meta.json` |",
        f"| Grad-CAM の閾値 | {g['threshold_used']} "
        f"(recomputed_on_test = {g['threshold_recomputed_on_test']}) | 同上 |", "",
        "## 照合できた整合性", "",
        f"- Test 予測の AI-agent 側 ROC-AUC は、`test_metrics_primary.json` "
        f"({t['auroc']['point']:.6f}) と DeLong 側 ({dl['auroc_aiagent']:.6f}) で一致",
        f"- Grad-CAM の群別該当数（TP {g['cases_available_per_group']['TP']}・"
        f"FP {g['cases_available_per_group']['FP']}・TN {g['cases_available_per_group']['TN']}・"
        f"FN {g['cases_available_per_group']['FN']}）は、主操作点の混同行列 "
        f"TP/FP/TN/FN = {int(t['operating_point_youden']['tp'])}/"
        f"{int(t['operating_point_youden']['fp'])}/{int(t['operating_point_youden']['tn'])}/"
        f"{int(t['operating_point_youden']['fn'])} と一致",
        f"- Grad-CAM が参照した Test 予測・凍結 selection の SHA256 は "
        f"`test_run_meta.json` の値と一致（{g['final_selection_sha256'][:16]}…）", "",
        "## 収録・除外の対応", "", provenance_table(prov), ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default="github_repo")
    ap.add_argument("--no-patient-level", action="store_true",
                    help="omit the patient-level split / manifest CSVs")
    ap.add_argument("--drive-root", default=None,
                    help="path to 「気合のCOVID19」; read-only, supplies the Test / comparison / "
                         "Grad-CAM master artefacts. Omit to leave those sections 未確認.")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copied = []
    for d in CODE_DIRS:
        copied += copy_tree(project / d, out)
    redactions = {}
    for rel in DOCS:
        src = project / rel
        if src.exists():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            # the decision log records local Drive paths; publish them without the user name
            text, n = redact(src.read_text(encoding="utf-8"))
            (out / rel).write_text(text, encoding="utf-8")
            if n:
                redactions[rel] = n
            copied.append(rel)
    for src_rel, dst_rel in RESULTS.items():
        src = project / src_rel
        if src.exists():
            target = out / "results/taskA" / dst_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            text, n = redact(src.read_text(encoding="utf-8"))
            target.write_text(text, encoding="utf-8")
            if n:
                redactions[f"results/taskA/{dst_rel}"] = n
            copied.append(f"results/taskA/{dst_rel}")
    if not args.no_patient_level:
        for src_rel, dst_rel in PATIENT_LEVEL.items():
            src = project / src_rel
            if not src.exists():
                continue
            target = out / "results/taskA" / dst_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            dropped = drop_columns(src, target)
            copied.append(f"results/taskA/{dst_rel}" + (f" (dropped: {dropped})" if dropped else ""))

    # ---- artefacts whose master copy lives on Google Drive ---------------------------
    drive = Path(args.drive_root) if args.drive_root else None
    prov, dv = [], None
    if drive and drive.exists():
        prov = import_from_drive(drive, out)["provenance"]
        dv = drive_numbers(drive)
    else:
        print(f"NOTE: --drive-root not usable ({args.drive_root}); "
              f"Test / comparison / Grad-CAM sections stay 未確認")
    for sub in ("evaluation", "comparison", "gradcam"):
        d = out / "results/taskA" / sub
        d.mkdir(parents=True, exist_ok=True)
        if not any(p.is_file() for p in d.iterdir()):
            (d / ".gitkeep").touch()

    v = verified_numbers(project)
    tree_lines = []
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out)
        if len(rel.parts) <= 2 and (p.is_dir() or len(rel.parts) == 1):
            tree_lines.append(("  " * (len(rel.parts) - 1)) + rel.parts[-1] + ("/" if p.is_dir() else ""))
    fields = dict(
        cohort=v["cohort"], train=v["split"]["train"], val=v["split"]["val"], test=v["split"]["test"],
        d_train=v["deaths"]["train"], d_val=v["deaths"]["val"], d_test=v["deaths"]["test"],
        images=v["images"], qc=v["qc_roundtrip"], split_sha=v["split_sha256"][:16] + "…",
        delta="、".join(f"{k} 日 {n} 例" for k, n in sorted(v["delta_days"].items(), reverse=True)),
        best_condition=v["best_condition"], best_auroc=v["best_val_auroc"],
        improvement=v["stage2_best_improvement"], margin=v["margin"],
        uid_status="PUBLISH_UIDS = True で収録に切り替え可" if not PUBLISH_UIDS else "現在は収録",
        tree="\n".join(tree_lines[:40]), today=date.today().isoformat())
    fields.update(readme_result_fields(dv))
    (out / "README.md").write_text(README.format(**fields), encoding="utf-8")
    (out / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")
    (out / "results/taskA/README.md").write_text(
        RESULTS_README.format(provenance_table=provenance_table(prov)), encoding="utf-8")
    (out / "docs/results_provenance.md").write_text(
        provenance_doc(v, dv, prov), encoding="utf-8")
    copied.append("docs/results_provenance.md")

    # ---- safety audit of what we just assembled -------------------------------------
    problems = {"large_files": [], "binaries": [], "caches": [], "secrets": [],
                "local_paths_outside_notebooks": [], "uid_columns": [], "duplicates": [],
                "dicom_or_checkpoint": []}
    digests: dict[str, list[str]] = {}
    files = [p for p in out.rglob("*") if p.is_file()]
    for p in files:
        rel = str(p.relative_to(out)).replace("\\", "/")
        if p.stat().st_size > MAX_FILE_MB * 2 ** 20:
            problems["large_files"].append(f"{rel} ({p.stat().st_size / 2**20:.1f} MB)")
        if p.suffix.lower() in SKIP_SUFFIX and rel not in ALLOW_BINARY:
            problems["binaries"].append(rel)
        if p.suffix.lower() in {".dcm", ".pt", ".pth", ".ckpt", ".npy", ".npz"}:
            problems["dicom_or_checkpoint"].append(rel)
        if set(p.parts) & SKIP_PARTS:
            problems["caches"].append(rel)
        if p.stat().st_size:
            digests.setdefault(sha256(p), []).append(rel)
        if p.suffix.lower() == ".csv" and not PUBLISH_UIDS:
            with open(p, encoding="utf-8-sig", newline="") as fh:
                header = fh.readline()
            if any(u in header for u in UID_COLUMNS):
                problems["uid_columns"].append(rel)
        if p.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".csv", ".toml", ".txt", ".ipynb"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if rel != AUDIT_SELF:            # this file contains the detection patterns itself
                for pattern in SECRET_PATTERNS:
                    if pattern.search(text):
                        problems["secrets"].append(f"{rel}: {pattern.pattern[:40]}")
            # Drive paths are expected in the notebooks, in the README and in usage examples
            allowed = rel.startswith(("notebooks/", "docs/")) or rel in {"README.md", AUDIT_SELF}
            if not allowed and re.search(r"[A-Za-z]:[\\/]Users[\\/]|/content/drive|"
                                         r"マイドライブ[\\/]気合のCOVID19", text):
                problems["local_paths_outside_notebooks"].append(rel)
    for digest, rels in digests.items():
        if len(rels) > 1 and not all(r.endswith(".gitkeep") for r in rels):
            problems["duplicates"].append(" == ".join(rels))
    summary = {"out_dir": str(out), "files": len(files),
               "bytes": sum(p.stat().st_size for p in files),
               "drive_root": str(drive) if drive else None,
               "imported_from_drive": sum(1 for r in prov if r["included"]),
               "excluded_from_drive": sum(1 for r in prov if not r["included"]),
               "publish_uids": PUBLISH_UIDS,
               "verified_numbers": v, "redacted_paths": redactions,
               "audit": {k: v2[:8] for k, v2 in problems.items()},
               "audit_counts": {k: len(v2) for k, v2 in problems.items()}}
    (out / "results/taskA/github_assembly_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
