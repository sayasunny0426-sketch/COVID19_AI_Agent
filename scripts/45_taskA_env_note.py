"""Record what is actually known about the Task A run environment, as a NEW file.

Task A's `results/taskA/evaluation/test_run_meta.json` is a frozen artefact and is not
modified. It omits package versions, the random seed and the generating script name. This
script collects those facts from artefacts that do record them and writes a separate note:

    results/taskA/evaluation/test_run_env_note.json

Nothing is inferred. Every field carries a `status`:

    confirmed              read from a named artefact; `source` says which
    documented             not stored as a machine-readable field, but stated in a document
                           inside this repository; `source` names the document
    not recorded           nobody wrote it down; the value is null

The frozen `final_selection.json` lives on Google Drive rather than in this repository. It is
read READ-ONLY and only after its SHA256 matches the value recorded inside the repository, so
the numbers taken from it are verifiably the frozen ones.

Usage:
    python scripts/45_taskA_env_note.py --project . --repo github_repo \
        --drive-root "<DRIVE_ROOT>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

FINAL_SELECTION_REL = "01_TaskA_CXR/04_Training/AIagent_taskA_runs/final_selection.json"
OUT_REL = "results/taskA/evaluation/test_run_env_note.json"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def confirmed(value, source):
    return {"value": value, "status": "confirmed", "source": source}


def documented(value, source):
    return {"value": value, "status": "documented", "source": source}


def missing(reason):
    return {"value": None, "status": "not recorded", "source": None, "note": reason}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="github_repo")
    ap.add_argument("--drive-root", default=r"<DRIVE_ROOT>")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    repo = project / args.repo
    out = repo / OUT_REL

    if out.exists():
        print(f"NOTE: {OUT_REL} already exists and will be rewritten "
              f"(it is a generated note, not a frozen result)")

    meta_rel = "results/taskA/evaluation/test_run_meta.json"
    gc_rel = "results/taskA/gradcam/gradcam_run_meta.json"
    prim_rel = "results/taskA/evaluation/test_metrics_primary.json"
    dec_rel = "results/taskA/training/condition_selection_decision.json"

    meta = json.loads((repo / meta_rel).read_text(encoding="utf-8-sig"))
    gc = json.loads((repo / gc_rel).read_text(encoding="utf-8-sig"))
    prim = json.loads((repo / prim_rel).read_text(encoding="utf-8-sig"))
    dec = json.loads((repo / dec_rel).read_text(encoding="utf-8-sig"))

    # ---- the frozen selection file, verified by the hash the repository already records ----
    fs_path = Path(args.drive_root) / FINAL_SELECTION_REL
    expected = meta["frozen_sha256"]
    fs, fs_status = None, None
    if not fs_path.exists():
        fs_status = {"available": False, "reason": "final_selection.json not reachable at the "
                                                   "recorded Drive location",
                     "expected_sha256": expected}
        print("WARNING: final_selection.json not found; its fields will be 'not recorded'")
    else:
        got = sha256(fs_path)
        if got != expected:
            print(f"STOP: final_selection.json SHA256 {got[:16]}… does not match the value "
                  f"recorded in {meta_rel} ({expected[:16]}…). Refusing to read it.")
            return 1
        fs = json.loads(fs_path.read_text(encoding="utf-8-sig"))
        fs_status = {"available": True, "sha256": got, "sha256_matches_repository_record": True,
                     "recorded_in": meta_rel,
                     "location": "Google Drive (not published in this repository)"}
        print(f"final_selection.json verified: SHA256 matches {meta_rel}")

    seed = str(fs["primary_analysis"]["model"].split("seed")[-1]) if fs else None
    run = fs["runs"][seed] if fs and seed in fs.get("runs", {}) else None

    note = {
        "purpose": "Environment and provenance facts that results/taskA/evaluation/"
                   "test_run_meta.json does not record. This file adds information; it does "
                   "not restate, override or correct any frozen Task A artefact.",
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generated_by": "scripts/45_taskA_env_note.py",
        "rule": "Nothing here is inferred. Each field carries a status of `confirmed`, "
                "`documented` or `not recorded`, and a `source`. Fields nobody wrote down are "
                "null with status `not recorded`.",
        "status_legend": {
            "confirmed": "read from the named artefact",
            "documented": "not stored as a machine-readable field, but stated in the named "
                          "document inside this repository",
            "not recorded": "not written down anywhere that could be checked; value is null",
        },
        "frozen_selection_file": fs_status,

        # ---------------------------------------------------------------- package versions
        "package_versions": {
            "python": confirmed(meta["python"], f"{meta_rel}.python") if "python" in meta
                      else missing("absent from test_run_meta.json"),
            "torch": confirmed(meta["torch"], f"{meta_rel}.torch") if "torch" in meta
                     else missing("absent from test_run_meta.json"),
            "torchvision": missing("no Task A artefact records the torchvision version"),
            "numpy": missing("no Task A artefact records the numpy version"),
            "pandas": missing("no Task A artefact records the pandas version"),
            "scikit-learn": missing("no Task A artefact records the scikit-learn version"),
            "pillow": missing("no Task A artefact records the pillow version"),
            "pydicom": missing("no Task A artefact records the pydicom version"),
            "cuda_toolkit": missing(
                "only the CUDA suffix of the torch build string is recorded "
                f"({meta.get('torch')}); the toolkit version itself was not recorded"),
        },
        "platform": {
            "os": confirmed(meta["platform"], f"{meta_rel}.platform"),
            "device": confirmed(meta["device"], f"{meta_rel}.device"),
            "gpu": confirmed(meta["gpu"], f"{meta_rel}.gpu"),
            "python_at_freeze_time": (
                confirmed(fs["environment"]["python"], "final_selection.json.environment.python")
                if fs else missing("final_selection.json not read")),
            "platform_at_freeze_time": (
                confirmed(fs["environment"]["platform"],
                          "final_selection.json.environment.platform")
                if fs else missing("final_selection.json not read")),
        },

        # ------------------------------------------------------------------------- seeds
        "random_seeds": {
            "model_seed_primary": (
                confirmed(run["seed"], "final_selection.json.runs.42.seed") if run
                else confirmed(gc["seed"], f"{gc_rel}.seed")),
            "model_seeds_all_runs": (
                confirmed(sorted(int(k) for k in fs["runs"]), "final_selection.json.runs")
                if fs else missing("final_selection.json not read")),
            "bootstrap_seed": confirmed(prim["auroc"]["seed"],
                                        f"{prim_rel}.auroc.seed"),
            "bootstrap_n_resamples": confirmed(prim["auroc"]["n_boot"],
                                               f"{prim_rel}.auroc.n_boot"),
            "bootstrap_stratified": confirmed(prim["auroc"]["stratified"],
                                              f"{prim_rel}.auroc.stratified"),
            "dataloader_shuffle_seed": missing(
                "not recorded separately from the model seed"),
        },

        # --------------------------------------------------------------- frozen model spec
        "frozen_model": {
            "condition": confirmed(dec["decision"]["selected_condition"],
                                   f"{dec_rel}.decision.selected_condition"),
            "model": confirmed(gc["model"], f"{gc_rel}.model"),
            "best_epoch": (confirmed(run["best_epoch"], "final_selection.json.runs.42.best_epoch")
                           if run else missing("final_selection.json not read")),
            "learning_rate": (confirmed(run["lr"], "final_selection.json.runs.42.lr") if run
                              else missing("final_selection.json not read")),
            "augmentation": (confirmed(run["augment"], "final_selection.json.runs.42.augment")
                             if run else missing("final_selection.json not read")),
            "pos_weight": (confirmed(run["pos_weight"], "final_selection.json.runs.42.pos_weight")
                           if run else missing("final_selection.json not read")),
            "unfreeze_schedule": (
                confirmed(run["unfreeze_schedule"],
                          "final_selection.json.runs.42.unfreeze_schedule") if run
                else missing("final_selection.json not read")),
            "validation_auroc_at_best_epoch": (
                confirmed(run["val_auroc_at_best"],
                          "final_selection.json.runs.42.val_auroc_at_best") if run
                else missing("final_selection.json not read")),
            "checkpoint_sha256": confirmed(gc["checkpoint_sha256"], f"{gc_rel}.checkpoint_sha256"),
            "frozen_selection_sha256": confirmed(meta["frozen_sha256"],
                                                 f"{meta_rel}.frozen_sha256"),
        },

        # ------------------------------------------------------------------- input hashes
        "input_hashes": {
            "split_manifest_sha256": (
                confirmed(fs["inputs"]["split_manifest_sha256"],
                          "final_selection.json.inputs.split_manifest_sha256") if fs
                else confirmed(gc["split_manifest_sha256"], f"{gc_rel}.split_manifest_sha256")),
            "index_cxr_manifest_sha256": (
                confirmed(fs["inputs"]["index_cxr_manifest_sha256"],
                          "final_selection.json.inputs.index_cxr_manifest_sha256") if fs
                else confirmed(gc["index_manifest_sha256"], f"{gc_rel}.index_manifest_sha256")),
            "run_config_sha256": (
                confirmed(run["config_sha256"], "final_selection.json.runs.42.config_sha256")
                if run else missing("final_selection.json not read")),
        },

        # --------------------------------------------------------------- generating scripts
        "generating_scripts": {
            "preprocessing_cache": documented(
                "scripts/06_taskA_preprocess_cache.py",
                "README.md section 12 (the Task A README), reproduction-steps table"),
            "training": documented(
                "scripts/09_taskA_run_training.py",
                "README.md section 12 (the Task A README), reproduction-steps table"),
            "freeze": documented(
                "scripts/14_taskA_freeze_selection.py",
                "README.md section 12 (the Task A README), reproduction-steps table"),
            "test_evaluation": documented(
                "scripts/15_taskA_test_evaluation.py",
                "README.md section 12 (the Task A README), reproduction-steps table; the "
                f"`outputs` list in {meta_rel} matches the files this script writes"),
            "gradcam": confirmed(gc["script"], f"{gc_rel}.script"),
            "gradcam_script_sha256": confirmed(gc["script_sha256"], f"{gc_rel}.script_sha256"),
            "test_evaluation_script_sha256": missing(
                "scripts/15_taskA_test_evaluation.py was not hashed at run time; only the "
                "Grad-CAM script recorded its own hash"),
        },

        "caveats": [
            "Task A ran on Google Colab, Task B and Task C ran locally. The package versions "
            "in results/task{B,C}/evaluation/test_run_meta.json describe the local "
            "environment and must not be read as Task A's.",
            "The Task A model checkpoint is not published. Training and Grad-CAM cannot be "
            "re-run from this repository alone; every downstream result can be, from the "
            "saved Test predictions.",
            "This note adds information only. Where it and a frozen artefact both state a "
            "value, the frozen artefact is authoritative.",
        ],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(note, ensure_ascii=False, indent=2) + "\n"
    out.write_text(payload, encoding="utf-8")
    src = project / OUT_REL
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(payload, encoding="utf-8")

    # ---- report what was and was not confirmed ------------------------------------------
    def tally(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict) and "status" in v:
                yield f"{pre}{k}", v["status"], v["value"]
            elif isinstance(v, dict):
                yield from tally(v, f"{pre}{k}.")

    rows = list(tally(note))
    for name, status, value in rows:
        mark = {"confirmed": "OK ", "documented": "DOC", "not recorded": "-- "}[status]
        shown = "null" if value is None else str(value)[:54]
        print(f"  [{mark}] {name:52s} {shown}")
    counts = {}
    for _, s, _ in rows:
        counts[s] = counts.get(s, 0) + 1
    print(f"\n{OUT_REL}: {len(payload.encode('utf-8')):,} B  "
          + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("No existing Task A artefact was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
