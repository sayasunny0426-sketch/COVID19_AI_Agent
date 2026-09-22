"""Task B: copy the development-phase artefacts into the GitHub repo and audit them.

Adds Task B to the existing private repository without touching anything Task A produced.
Refuses to run if a Test prediction or Test metric file has appeared.

Usage:
    python scripts/31_taskB_prepare_commit.py --project . --repo github_repo
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CODE = [
    "src/covid_mortality/features/taskB_clinical.py",
    "src/covid_mortality/features/taskB_preprocess.py",
    "src/covid_mortality/training/taskB_models.py",
    "src/covid_mortality/evaluation/taskB_schema.py",
    "scripts/21_taskB_clinical_audit.py",
    "scripts/22_taskB_preprocess_spec.py",
    "scripts/23_taskB_missingness_mechanism.py",
    "scripts/24_taskB_missing_indicator_decision.py",
    "scripts/25_taskB_variable_selection.py",
    "scripts/26_taskB_train_models.py",
    "scripts/27_taskB_select_and_freeze.py",
    "scripts/28_taskB_pre_test_qc.py",
    "scripts/29_taskB_indicator_sensitivity.py",
    "scripts/30_taskB_test_evaluation.py",
    "scripts/31_taskB_prepare_commit.py",
    "scripts/32_taskB_post_test_qc.py",
]
DOCS = ["docs/decision_log.md", "docs/change_log.csv", "docs/open_questions.md"]
RESULT_DIRS = ["preprocessing", "variable_selection", "modeling", "qc",
               # kept in the repository but clearly separated: an analysis that was NOT
               # performed, so its material must never be read as a result (D-081)
               "secondary_exploratory_unused",
               # populated only after the approved Test evaluation
               "evaluation", "feature_importance"]
TOP_FILES = ["results/taskB/README.md"]
# files that moved out of the primary tree and must not linger in the repository
STALE = ["results/taskB/preprocessing/inpatient_sensitivity_flow.csv",
         "results/taskB/preprocessing/visit_type_composition.csv"]
# model binaries stay out of the repository; their hashes are in the frozen JSON
SKIP_SUFFIX = {".joblib", ".pth", ".pt", ".pkl"}
MAX_MB = 5
SECRET_PATTERNS = [re.compile(p, re.IGNORECASE) for p in
                   (r"(api[_-]?key|secret|passwd|password|access[_-]?token)\s*[:=]\s*[\"'][^\"']+[\"']",
                    r"gh[pousr]_[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}",
                    r"-----BEGIN [A-Z ]*PRIVATE KEY-----")]
PATH_PATTERNS = [
    (re.compile(r"[A-Za-z]:[\\/]{1,2}Users[\\/]{1,2}[^\s\"'、。，`)\]}<>|]*"), "<LOCAL_PATH>"),
    (re.compile(r"[A-Za-z]:[\\/]{1,2}マイドライブ[\\/]{1,2}気合のCOVID19"), "<DRIVE_ROOT>"),
    (re.compile(r"<DRIVE_ROOT>"), "<DRIVE_ROOT>"),
]
# raw clinical source data must never be committed
FORBIDDEN_NAMES = {"患者データファイル.csv", "metadata.csv", "患者データまとめ、定義.csv"}


def redact(text: str) -> tuple[str, int]:
    n = 0
    original = text
    for pat, rep in PATH_PATTERNS:
        text, k = pat.subn(rep, text)
        n += k
    if text.count("\n") != original.count("\n"):
        raise SystemExit("STOP: redaction crossed a newline")
    return text, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="github_repo")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    repo = project / args.repo
    if not repo.exists():
        print(f"STOP: {repo} does not exist")
        return 1

    # Before the Test evaluation is approved this refuses to run; afterwards the Test outputs
    # are expected, so the guard is satisfied by the post-Test QC report existing.
    ev = project / "results/taskB/evaluation"
    test_out = list(ev.glob("test_*")) if ev.exists() else []
    approved = (project / "results/taskB/qc/post_test_qc_report.json").exists()
    if test_out and not approved:
        print(f"STOP: Test output exists ({[p.name for p in test_out]}) but the post-Test QC "
              f"has not been run; refusing to commit an unverified Test result")
        return 1
    if test_out:
        qc = json.loads((project / "results/taskB/qc/post_test_qc_report.json")
                        .read_text(encoding="utf-8"))
        if qc["n_failed"]:
            print(f"STOP: post-Test QC has {qc['n_failed']} failing check(s)")
            return 1
        print(f"post-Test QC passed ({qc['n_checks']} checks); Test outputs will be committed")

    removed = []
    for rel in STALE:
        p = repo / rel
        if p.exists():
            p.unlink()
            removed.append(rel)
    if removed:
        print(f"removed stale copies that moved out of the primary tree: {removed}")

    copied, redactions = [], {}
    for rel in CODE + DOCS + TOP_FILES:
        src = project / rel
        if not src.exists():
            print(f"  (missing, skipped) {rel}")
            continue
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text, n = redact(src.read_text(encoding="utf-8-sig"))
        dst.write_text(text, encoding="utf-8")
        if n:
            redactions[rel] = n
        copied.append(rel)

    for sub in RESULT_DIRS:
        s = project / "results/taskB" / sub
        if not s.exists():
            continue
        for p in sorted(s.rglob("*")):
            if not p.is_file() or p.suffix.lower() in SKIP_SUFFIX:
                continue
            if p.stat().st_size > MAX_MB * 2 ** 20:
                print(f"  (too large, skipped) {p.name}")
                continue
            rel = f"results/taskB/{sub}/{p.relative_to(s).as_posix()}"
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix.lower() == ".png":
                shutil.copy2(p, dst)
            else:
                text, n = redact(p.read_text(encoding="utf-8-sig"))
                dst.write_text(text, encoding="utf-8")
                if n:
                    redactions[rel] = n
            copied.append(rel)

    # ---- safety audit over the whole repository -----------------------------------
    problems = {"raw_clinical_source": [], "dicom": [], "secrets": [], "checkpoints": [],
                "large_files": [], "test_outputs": [], "local_paths": []}
    for p in repo.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = str(p.relative_to(repo)).replace("\\", "/")
        if p.name in FORBIDDEN_NAMES:
            problems["raw_clinical_source"].append(rel)
        if p.suffix.lower() in {".dcm", ".nii"}:
            problems["dicom"].append(rel)
        if p.suffix.lower() in SKIP_SUFFIX | {".ckpt", ".npy", ".npz"}:
            problems["checkpoints"].append(rel)
        if p.stat().st_size > MAX_MB * 2 ** 20:
            problems["large_files"].append(f"{rel} ({p.stat().st_size / 2**20:.1f} MB)")
        # Test outputs are a defect before approval and expected after it
        if rel.startswith("results/taskB/evaluation/test_") and not approved:
            problems["test_outputs"].append(rel)
        if p.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".csv", ".toml", ".txt", ".ipynb"}:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if rel != "scripts/31_taskB_prepare_commit.py":
                for pat in SECRET_PATTERNS:
                    if pat.search(t):
                        problems["secrets"].append(rel)
            allowed = rel.startswith(("notebooks/", "docs/")) or rel in {
                "README.md", "scripts/09_taskA_run_training.py",
                "scripts/20_prepare_github_repo.py", "scripts/31_taskB_prepare_commit.py"}
            if not allowed and re.search(r"[A-Za-z]:[\\/]Users[\\/]|/content/drive", t):
                problems["local_paths"].append(rel)

    print(f"\ncopied {len(copied)} files into {repo}")
    print(f"redacted paths in {len(redactions)} files")
    print("\n=== safety audit ===")
    for k, v in problems.items():
        print(f"  [{'OK ' if not v else 'NG '}] {k}: {len(v)}{'' if not v else ' ' + str(v[:4])}")

    report = {"generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
              "script": Path(__file__).name, "files_copied": len(copied),
              "copied": copied, "redactions": redactions, "stale_removed": removed,
              "audit_counts": {k: len(v) for k, v in problems.items()},
              "audit": {k: v[:8] for k, v in problems.items()},
              "test_outputs_present": bool(test_out),
              "post_test_qc_passed": approved}
    (repo / "results/taskB/commit_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any(problems.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
