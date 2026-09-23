"""Task C: copy the Task C artefacts into github_repo/, verify, and audit before committing.

The working directory and the git repository are separate trees, so "the file exists" and "the
file is in the repository" are different statements. This script copies, then proves the copy
landed and matches the source, then audits what would be committed.

Task A and Task B artefacts are never written to.

Usage:
    python scripts/39_taskC_prepare_commit.py --project . --repo github_repo
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

CODE = [
    "src/covid_mortality/fusion/__init__.py",
    "src/covid_mortality/fusion/taskC_fusion.py",
    "scripts/35_taskC_preflight_audit.py",
    "scripts/36_taskC_fusion_search.py",
    "scripts/37_taskC_freeze.py",
    "scripts/38_taskC_pre_test_qc.py",
    "scripts/39_taskC_prepare_commit.py",
    "scripts/40_taskC_test_evaluation.py",
    "scripts/41_taskC_delong.py",
    "scripts/42_taskC_post_test_qc.py",
    "scripts/43_taskC_human_vs_agent_comparison.py",
]
DOCS = ["docs/decision_log.md", "docs/change_log.csv", "docs/open_questions.md",
        "docs/taskC_human_vs_ai_agent_comparison.md"]
RESULT_DIRS = ["preflight", "inputs", "fusion", "validation", "modeling", "qc", "evaluation",
               "comparison"]
COMPARISON_FILES = ["results/comparison/README.md", "results/comparison/delong_plan.json",
                    "results/comparison/test_predictions_all_models.csv",
                    "results/comparison/delong_results.csv",
                    "results/comparison/delong_results.json"]
SKIP_SUFFIX = {".joblib", ".pth", ".pt", ".pkl", ".ckpt"}
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
FORBIDDEN_NAMES = {"患者データファイル.csv", "metadata.csv", "患者データまとめ、定義.csv"}
SELF = "scripts/39_taskC_prepare_commit.py"


def redact(text: str) -> tuple[str, int]:
    original, n = text, 0
    for pat, rep in PATH_PATTERNS:
        text, k = pat.subn(rep, text)
        n += k
    if text.count("\n") != original.count("\n"):
        raise SystemExit("STOP: redaction crossed a newline")
    return text, n


def copy_text(src: Path, dst: Path) -> int:
    raw = src.read_bytes()
    text, n = redact(raw.decode("utf-8-sig"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw if n == 0 else text.encode("utf-8"))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="github_repo")
    ap.add_argument("--stage", default="pre_test", choices=["pre_test", "final"])
    args = ap.parse_args()
    project = Path(args.project).resolve()
    repo = project / args.repo
    if not repo.exists():
        print(f"STOP: {repo} does not exist")
        return 1

    ev = project / "results/taskC/evaluation"
    test_out = sorted(p.name for p in ev.glob("test_*")) if ev.exists() else []
    if args.stage == "pre_test" and test_out:
        print(f"STOP: Task C Test output already exists ({test_out}); "
              f"this checkpoint is the pre-Test freeze")
        return 1
    if args.stage == "final" and not test_out:
        print("STOP: --stage final but no Task C Test output exists")
        return 1

    copied, redactions = [], {}
    for rel in CODE + DOCS:
        src = project / rel
        if not src.exists():
            continue
        n = copy_text(src, repo / rel)
        if n:
            redactions[rel] = n
        copied.append(rel)
    for sub in RESULT_DIRS:
        s = project / "results/taskC" / sub
        if not s.exists():
            continue
        for p in sorted(s.rglob("*")):
            if not p.is_file() or p.suffix.lower() in SKIP_SUFFIX:
                continue
            if p.stat().st_size > MAX_MB * 2 ** 20:
                print(f"  (too large, skipped) {p.name}")
                continue
            rel = f"results/taskC/{sub}/{p.relative_to(s).as_posix()}"
            if p.suffix.lower() == ".png":
                (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, repo / rel)
            else:
                n = copy_text(p, repo / rel)
                if n:
                    redactions[rel] = n
            copied.append(rel)
    for rel in COMPARISON_FILES:
        src = project / rel
        if src.exists():
            n = copy_text(src, repo / rel)
            if n:
                redactions[rel] = n
            copied.append(rel)

    print(f"copied {len(copied)} files into {repo}")
    print(f"redacted paths in {len(redactions)} files")

    # ---- prove the copy landed and matches -------------------------------------------
    absent, stale = [], []
    for rel in copied:
        s, d = project / rel, repo / rel
        if not d.exists():
            absent.append(rel)
        elif s.suffix.lower() != ".png":
            src_txt, _ = redact(s.read_bytes().decode("utf-8-sig"))
            if src_txt != d.read_bytes().decode("utf-8-sig"):
                stale.append(rel)
    if absent or stale:
        print(f"STOP: {len(absent)} missing, {len(stale)} stale")
        for r in (absent + stale)[:10]:
            print(f"    {r}")
        return 1
    print(f"verified: all {len(copied)} copied files are present and match the source")

    # ---- Task A / Task B artefacts must be untouched -----------------------------------
    print("(Task A and Task B artefacts are not written by this script)")

    # ---- safety audit over the whole repository ----------------------------------------
    problems = {"raw_clinical_source": [], "dicom": [], "phi_images": [], "secrets": [],
                "checkpoints": [], "large_files": [], "local_paths": []}
    for p in repo.rglob("*"):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = str(p.relative_to(repo)).replace("\\", "/")
        if p.name in FORBIDDEN_NAMES:
            problems["raw_clinical_source"].append(rel)
        if p.suffix.lower() in {".dcm", ".nii"}:
            problems["dicom"].append(rel)
        if p.suffix.lower() in SKIP_SUFFIX | {".npy", ".npz"}:
            problems["checkpoints"].append(rel)
        # Patient imaging must not appear except where it was explicitly approved. Aggregate
        # figures (distributions, curves) are not patient images and are allowed.
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} and not (
                rel == "results/taskA/gradcam/gradcam_panel_4x4.png"     # approved, D-066
                or rel.startswith("results/taskB/preprocessing/figures/")  # distribution plots
                or rel.startswith("results/taskC/")):                     # curves written here
            problems["phi_images"].append(rel)
        if p.stat().st_size > MAX_MB * 2 ** 20:
            problems["large_files"].append(f"{rel} ({p.stat().st_size / 2**20:.1f} MB)")
        if p.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".csv", ".toml", ".txt", ".ipynb"}:
            t = p.read_text(encoding="utf-8", errors="ignore")
            if rel != SELF:
                for pat in SECRET_PATTERNS:
                    if pat.search(t):
                        problems["secrets"].append(rel)
            allowed = rel.startswith(("notebooks/", "docs/")) or rel in {
                "README.md", "scripts/09_taskA_run_training.py",
                "scripts/20_prepare_github_repo.py", "scripts/31_taskB_prepare_commit.py", SELF}
            if not allowed and re.search(r"[A-Za-z]:[\\/]Users[\\/]|/content/drive", t):
                problems["local_paths"].append(rel)

    print("\n=== safety audit ===")
    for k, v in problems.items():
        print(f"  [{'OK ' if not v else 'NG '}] {k}: {len(v)}{'' if not v else ' ' + str(v[:4])}")

    (repo / "results/taskC").mkdir(parents=True, exist_ok=True)
    (repo / "results/taskC/commit_audit.json").write_text(json.dumps({
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name, "stage": args.stage,
        "files_copied": len(copied), "copied": copied, "redactions": redactions,
        "test_outputs_present": bool(test_out),
        "audit_counts": {k: len(v) for k, v in problems.items()},
        "audit": {k: v[:8] for k, v in problems.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if any(problems.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
