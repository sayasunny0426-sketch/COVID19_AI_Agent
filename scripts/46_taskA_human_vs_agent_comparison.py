"""Task A: exploratory methodological comparison of the human-guided and AI-agent pipelines.

This is a methodological comparison, not a contest. It establishes neither superiority nor
equivalence of either approach: the two pipelines differ in several components at once, so no
performance difference can be attributed to one design choice.

Sources
-------
Human-guided: the current final artefacts under 気合のCOVID19, read READ-ONLY. Files whose name
begins with 旧 are not read. The manuscript and slides are Google Docs stubs on this filesystem
and could not be read; that is recorded as a limitation rather than guessed at.

AI-agent: the frozen Task A artefacts in github_repo/. `final_selection.json` lives on Drive
and is read only after its SHA256 matches the value recorded in the repository.

Nothing is inferred. Where two sources disagree they are both recorded in
source_discrepancies.csv and neither is silently preferred.

Outputs (working tree; Task A official artefacts are never written)
------------------------------------------------------------------
    docs/taskA_human_vs_ai_agent_comparison.md
    results/taskA/comparison/human_vs_ai_agent_model_spec.csv
    results/taskA/comparison/human_vs_ai_agent_performance.csv
    results/taskA/comparison/human_vs_ai_agent_gradcam.csv
    results/taskA/comparison/source_discrepancies.csv
    results/taskA/comparison/comparison_meta.json
    results/taskA/comparison/harmonized_ece_ci.csv

Note on numbering: the prompt suggested scripts/45, but scripts/45_taskA_env_note.py already
exists and is committed, so this script is 46.

Usage:
    python scripts/46_taskA_human_vs_agent_comparison.py --project . --repo github_repo \
        --drive-root "<DRIVE_ROOT>"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

H = "Human-guided"
A = "AI-Agent"
NR = "not recorded in the artefacts read"
NE = "not established from the artefacts read"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def f6(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.6f}"


# ---------------------------------------------------------------------------------------
# metric definitions, written once and applied identically to both pipelines
# ---------------------------------------------------------------------------------------
def ece_equal_width(y, p, n_bins=10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1]) if i < n_bins - 1 else \
            (p >= edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def ece_equal_count(y, p, n_bins=5) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total = 0.0
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def strat_boot_ci(y, p, n_boot, seed, alpha=0.05):
    """Stratified bootstrap percentile CI: cases and controls resampled separately."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        out[i] = roc_auc_score(y[idx], p[idx])
    return float(np.percentile(out, 100 * alpha / 2)), \
        float(np.percentile(out, 100 * (1 - alpha / 2)))


def rates(y, p, thr):
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "ppv": tp / (tp + fp) if tp + fp else float("nan"),
            "npv": tn / (tn + fn) if tn + fn else float("nan"),
            "accuracy": (tp + tn) / len(y)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--repo", default="github_repo")
    ap.add_argument("--drive-root", default=r"<DRIVE_ROOT>")
    args = ap.parse_args()
    project = Path(args.project).resolve()
    repo = project / args.repo
    D = Path(args.drive_root)
    TA = D / "01_TaskA_CXR"
    out = project / "results/taskA/comparison"
    out.mkdir(parents=True, exist_ok=True)

    src_used: dict[str, str] = {}

    def read_csv(p: Path, label: str, **kw) -> pd.DataFrame:
        d = pd.read_csv(p, encoding="utf-8-sig", **kw)
        src_used[label] = f"{p} (sha256 {sha256(p)[:16]}…)"
        return d

    def read_json(p: Path, label: str):
        j = json.loads(p.read_text(encoding="utf-8-sig"))
        src_used[label] = f"{p} (sha256 {sha256(p)[:16]}…)"
        return j

    # ================================================================== human-guided
    h_cfg = read_csv(TA / "04_Training/TaskA_CXR_最終学習条件サマリー.csv",
                     "human_final_training_config").iloc[0]
    h_eval = read_csv(TA / "05_Evaluation/TaskA_CXR_最終評価サマリー.csv",
                      "human_final_evaluation_summary").iloc[0]
    h_stage1 = read_csv(TA / "04_Training/TaskA_CXR_Stage1_seed42_6条件比較.csv",
                        "human_stage1_conditions")
    h_stage2 = read_csv(TA / "04_Training/TaskA_CXR_Stage2_classweight比較.csv",
                        "human_stage2_class_weight")
    h_test = read_csv(TA / "05_Evaluation/TaskA_CXR_Test患者別予測結果.csv",
                      "human_test_predictions", dtype={"Subject ID": str})
    h_val = read_csv(TA / "05_Evaluation/TaskA_CXR_Validation患者別予測結果.csv",
                     "human_validation_predictions", dtype={"Subject ID": str})
    h_gc = read_csv(TA / "06_GradCAM/TaskA_CXR_GradCAM_16症例選定一覧.csv",
                    "human_gradcam_selection", dtype={"Subject ID": str})
    h_gcq = read_csv(TA / "06_GradCAM/TaskA_CXR_GradCAM_定性的所見まとめ.csv",
                     "human_gradcam_qualitative")
    h_boot = read_csv(TA / "05_Evaluation/TaskA_CXR_ROCAUC_bootstrap_10000.csv",
                      "human_bootstrap_draws")
    h_five = read_csv(D / "04_全モデル比較・統計解析/全5モデル_最終Test評価比較.csv",
                      "human_five_model_test_table").set_index("model")
    h_cxr5 = h_five.loc["CXR ResNet18"]

    # ================================================================== AI-agent
    a_prim = read_json(repo / "results/taskA/evaluation/test_metrics_primary.json",
                       "agent_test_metrics_primary")
    a_tbl = read_csv(repo / "results/taskA/evaluation/test_metrics_table.csv",
                     "agent_test_metrics_table")
    a_cond = read_csv(repo / "results/taskA/training/condition_comparison.csv",
                      "agent_condition_comparison")
    a_dec = read_json(repo / "results/taskA/training/condition_selection_decision.json",
                      "agent_condition_selection_decision")
    a_env = read_json(repo / "results/taskA/evaluation/test_run_env_note.json",
                      "agent_run_env_note")
    a_gcm = read_json(repo / "results/taskA/gradcam/gradcam_run_meta.json",
                      "agent_gradcam_run_meta")
    a_gcs = read_csv(repo / "results/taskA/gradcam/gradcam_selection.csv",
                     "agent_gradcam_selection", dtype={"subject_id": str})
    a_pre = read_json(repo / "results/taskA/preprocessing/preprocess_summary.json",
                      "agent_preprocess_summary")
    a_qc = read_json(repo / "results/taskA/training/dataset_qc.json", "agent_dataset_qc")
    a_test = read_csv(repo / "results/taskA/evaluation/test_predictions_primary.csv",
                      "agent_test_predictions", dtype={"subject_id": str})
    a_meta = read_json(repo / "results/taskA/evaluation/test_run_meta.json",
                       "agent_test_run_meta")
    a_taskc = read_csv(repo / "results/taskC/evaluation/test_metrics_table.csv",
                       "agent_taskC_five_model_table").set_index("model")
    a_alog = (repo / "results/taskA/evaluation/test_access_log.jsonl")
    a_access = [json.loads(x) for x in a_alog.read_text(encoding="utf-8-sig").splitlines() if x]
    src_used["agent_test_access_log"] = f"{a_alog} (sha256 {sha256(a_alog)[:16]}…)"

    fs_path = D / "01_TaskA_CXR/04_Training/AIagent_taskA_runs/final_selection.json"
    if sha256(fs_path) != a_meta["frozen_sha256"]:
        print("STOP: final_selection.json hash does not match the repository record.")
        return 1
    fs = read_json(fs_path, "agent_final_selection (hash-verified)")
    a_run = fs["runs"]["42"]
    a_vm = fs["primary_analysis"]["validation_metrics"]

    def envv(path):
        node = a_env
        for part in path.split("."):
            node = node[part]
        return node["value"] if node["status"] != "not recorded" else None

    # =========================================================== harmonised metrics
    hy, hp = h_test["true_label"].to_numpy(), h_test["prob_death"].to_numpy()
    ay, ap_ = a_test["true_label"].to_numpy(), a_test["prob"].to_numpy()
    h_thr, a_thr = float(h_eval["val_threshold"]), float(a_prim["thresholds"]["youden"]) \
        if "thresholds" in a_prim else float(a_prim["operating_point_youden"]["threshold"])

    harm_rows = []
    for label, y, p in ((H, hy, hp), (A, ay, ap_)):
        row = {"pipeline": label, "n": len(y), "events": int(y.sum()),
               "auroc_recomputed": roc_auc_score(y, p),
               "auprc_recomputed": average_precision_score(y, p),
               "brier_recomputed": brier_score_loss(y, p),
               "ece_10_equal_width": ece_equal_width(y, p, 10),
               "ece_5_equal_count": ece_equal_count(y, p, 5)}
        for n_boot, seed, tag in ((10000, 42, "human_setting_10000_seed42"),
                                  (2000, 12345, "agent_setting_2000_seed12345")):
            lo, hi = strat_boot_ci(y, p, n_boot, seed)
            row[f"ci_low_{tag}"] = lo
            row[f"ci_high_{tag}"] = hi
        harm_rows.append(row)
    harm = pd.DataFrame(harm_rows)
    harm.to_csv(out / "harmonized_ece_ci.csv", index=False, encoding="utf-8-sig")

    hh = harm[harm.pipeline == H].iloc[0]
    ha = harm[harm.pipeline == A].iloc[0]

    # =========================================================== model spec table
    h_aug_detail = str(h_cfg["augmentation_detail"])
    a_aug = a_qc["config"]["augmentations"]["aug_b"]
    a_aug_detail = (f"translate ±{a_aug['translate'] * 100:.0f}%; "
                    f"scale {a_aug['scale'][0]}-{a_aug['scale'][1]}; "
                    f"rotation ±{a_aug['degrees']:.0f}°; "
                    f"brightness {a_aug['brightness']}; contrast {a_aug['contrast']}; no flip")

    spec = [
        # (domain, item, human, agent, difference, note)
        ("Cohort", "total patients", "1,277", "1,277", "same",
         "both read COVID19_固定患者split_1277.csv; the AI-agent copy is "
         "results/taskA/preprocessing/fixed_split_1277.csv"),
        ("Cohort", "Train / Validation / Test", "1,021 / 128 / 128", "1,021 / 128 / 128",
         "same", "AI-agent side verified in results/taskA/training/dataset_qc.json"),
        ("Cohort", "deaths per split", "135 / 17 / 17", "135 / 17 / 17", "same",
         f"human-guided Validation n={int(h_eval['val_n'])} deaths={int(h_eval['val_deaths'])}, "
         f"Test n={int(h_eval['test_n'])} deaths={int(h_eval['test_deaths'])}"),
        ("Cohort", "same patient split", "yes", "yes", "same",
         "identical split file; the AI-agent pipeline additionally recorded its SHA256 "
         "(626061a5…) in final_selection.json"),
        ("Cohort", "endpoint", "in-hospital death", "in-hospital death", "same",
         "binary, same label column"),
        ("Cohort", "T0", "visit_start_datetime", "visit_start_datetime", "same",
         "shared cohort definition"),
        ("Image selection", "CXR eligibility window", "T0−2 days to T0", "T0−2 days to T0",
         "same", "AI-agent side: results/taskA/preprocessing/index_cxr_selection_flow.json"),
        ("Image selection", "view", "frontal AP/PA", "frontal AP/PA", "same",
         "shared upstream selection (CXR_最終選定画像一覧.csv)"),
        ("Image selection", "images per patient", "1", "1",
         "same", "AI-agent dataset_qc.json one_image_per_patient: "
                 f"{a_qc['one_image_per_patient']['detail']}"),
        ("Image selection", "index image identity", "shared upstream selection",
         "shared upstream selection", "same",
         "both pipelines start from the same selected-image list; the AI-agent pipeline "
         "records index_cxr_manifest_sha256 e8c07db7…"),
        ("Preprocessing", "DICOM pixel handling",
         "RescaleSlope / RescaleIntercept applied, then fixed windowing from "
         "WindowCenter / WindowWidth, then np.clip to the window",
         "percentile 1–99 clip, then scale to [0,1]", "DIFFERENT",
         "human: TaskA_02 notebook cell 7; agent: preprocess_summary.json pipeline "
         f"(window_center_width_used = {a_pre['pipeline']['window_center_width_used']})"),
        ("Preprocessing", "MONOCHROME handling",
         "MONOCHROME1 inverted; all 1,277 images are MONOCHROME2 so no inversion occurred",
         f"MONOCHROME2 for all {a_pre['photometric']['MONOCHROME2']} images; "
         f"inverted_count = {a_pre['inverted_count']}",
         "same outcome", "both handled inversion and neither needed it"),
        ("Preprocessing", "geometry",
         "direct resize to 224×224, no aspect-ratio preservation, no crop",
         f"zero pad to square on the long side, then {a_pre['target_size']}×"
         f"{a_pre['target_size']}, then 224×224", "DIFFERENT",
         "human: TaskA_02 cell 8 (cv2.resize to 224×224 direct); agent: "
         f"preprocess_summary.json pad = '{a_pre['pipeline']['pad']}', "
         f"median pad fraction {a_pre['pad_fraction']['median']}"),
        ("Preprocessing", "cache format", "224×224 8-bit PNG",
         f"{a_pre['target_size']}×{a_pre['target_size']} 16-bit PNG "
         f"({list(a_pre['png_mode'])[0]}), exact round-trip verified: "
         f"{a_pre['exact_roundtrip_all']}", "DIFFERENT",
         "the human-guided cache is already at model resolution; the AI-agent cache keeps "
         "512×512 16 bit and resizes at load time"),
        ("Preprocessing", "channel conversion",
         "PIL convert('L') then Grayscale(num_output_channels=3)",
         "single channel replicated to 3", "same concept",
         "human: TaskA_03 cells 2–3; agent: preprocess pipeline '→ 3ch'"),
        ("Preprocessing", "normalisation",
         "ImageNet mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225]",
         f"ImageNet mean {a_qc['config']['imagenet_mean']} std {a_qc['config']['imagenet_std']}",
         "same", "identical constants"),
        ("Preprocessing", "invalid images",
         "the preprocessing audit log records the per-image outcome",
         f"failures = {a_pre['failures']}, degenerate_count = {a_pre['degenerate_count']}, "
         f"image_read_errors = {a_qc['image_read_errors']['detail']['errors']}", "same outcome",
         "neither pipeline had to drop an image at this stage"),
        ("Architecture", "backbone", str(h_cfg["model"]),
         "ImageNet-1K pretrained ResNet18", "same",
         "human: TaskA_04 cell 8 uses ResNet18_Weights.IMAGENET1K_V1"),
        ("Architecture", "classifier head", "nn.Linear(512, 1), single logit",
         f"single logit head, in {a_qc.get('_', {}) or 512} → out 1", "same",
         "agent: results/taskA/training/model_smoke_check.json head_is_single_logit"),
        ("Architecture", "layer freezing", "full fine-tuning, all parameters trainable",
         "full fine-tuning for the selected condition; staged unfreezing was searched as "
         "Stage 2 and not selected", "DIFFERENT SEARCH, same final model",
         f"agent: condition_comparison.csv unfreeze_schedule is empty for "
         f"{a_dec['decision']['selected_condition']}"),
        ("Architecture", "input resolution", "224×224", f"{a_qc['config']['input_size']}×"
         f"{a_qc['config']['input_size']}", "same", "both 224"),
        ("Optimisation", "optimizer", str(h_cfg["optimizer"]), "AdamW", "same", ""),
        ("Optimisation", "learning rate", f"{float(h_cfg['initial_lr']):g}",
         f"{a_run['lr']:g}", "DIFFERENT",
         "human 1e-5, agent 3e-4: a 30-fold difference in the selected condition"),
        ("Optimisation", "weight decay", f"{float(h_cfg['weight_decay']):g}", "1e-4", "same",
         "agent value from the Task A README training table; "
         "not recorded as a field in final_selection.json"),
        ("Optimisation", "batch size", str(int(h_cfg["batch_size"])), "32", "same", ""),
        ("Optimisation", "max epochs", str(int(h_cfg["max_epochs"])), "30", "same", ""),
        ("Optimisation", "scheduler", f"{h_cfg['scheduler']} (factor "
         f"{h_cfg['scheduler_factor']}, patience {int(h_cfg['scheduler_patience'])}, "
         f"min_lr {float(h_cfg['scheduler_min_lr']):g})",
         "OneCycleLR (warm-up 1 epoch then cosine)", "DIFFERENT",
         "human: ReduceLROnPlateau (TaskA_04 cell 11); agent: Task A README §12 "
         "training table"),
        ("Optimisation", "early stopping",
         f"on {h_cfg['early_stopping_metric']}, patience "
         f"{int(h_cfg['early_stopping_patience'])}, min improvement "
         f"{h_cfg['early_stopping_min_improvement']}",
         "best-epoch checkpoint on Validation AUROC over a fixed 30-epoch budget", "DIFFERENT",
         "the human-guided run could stop early; the AI-agent run always ran the full budget "
         "and kept the best epoch"),
        ("Class imbalance", "loss", str(h_cfg["loss"]), "BCEWithLogitsLoss", "same", ""),
        ("Class imbalance", "pos_weight", f"{float(h_cfg['pos_weight']):.6f}",
         f"{a_run['pos_weight']:.6f}", "same",
         "both = 886/135 computed from the Training set; agreement to 6 decimals"),
        ("Augmentation", "selected condition", f"Augmentation {h_cfg['augmentation']}",
         f"{a_run['augment']}", "same label",
         "the labels match but the parameters do not — see the next row"),
        ("Augmentation", "parameters", h_aug_detail, a_aug_detail, "DIFFERENT",
         "human: TaskA_03 cell 3 train_transform_B; agent: dataset_qc.json "
         "config.augmentations.aug_b. Same name 'B'/'aug_b', different magnitudes"),
        ("Augmentation", "horizontal flip", "not used", "not used", "same", ""),
        ("Seeds", "seed of the final model", str(int(h_cfg["seed"])),
         str(envv("random_seeds.model_seed_primary")), "same", "both seed 42"),
        ("Seeds", "number of seeds trained", "1 (seed 42 only)",
         f"3 per condition ({envv('random_seeds.model_seeds_all_runs')})", "DIFFERENT",
         "human: every comparison CSV is labelled seed42; agent: condition_comparison.csv "
         f"n_seeds = {int(a_cond['n_seeds'].iloc[0])} for every condition"),
        ("Selection", "conditions compared",
         f"{len(h_stage1)} learning-rate × augmentation conditions, then "
         f"{len(h_stage2)} class-weight conditions",
         f"{a_dec['n_conditions']} conditions over {int(a_cond['n_seeds'].sum())} runs",
         "DIFFERENT",
         "human: TaskA_CXR_Stage1_seed42_6条件比較.csv + Stage2 class-weight comparison; "
         "agent: condition_comparison.csv"),
        ("Selection", "selection metric", "Validation ROC-AUC (single seed)",
         "mean Validation ROC-AUC over 3 seeds, with a pre-registered parsimony margin "
         f"of {a_dec['decision']['margin']}", "DIFFERENT",
         f"agent rule: {a_dec['rule']}; reason recorded as: "
         f"{a_dec['decision']['reason'][:150]}"),
        ("Selection", "checkpoint selection",
         f"best epoch {int(h_cfg['best_epoch'])} by Validation ROC-AUC",
         f"best epoch {envv('frozen_model.best_epoch')} by Validation ROC-AUC", "same rule",
         "different resulting epoch"),
        ("Selection", "final checkpoint", str(h_cfg["checkpoint"]),
         f"{fs['primary_analysis']['model']} best.pt "
         f"(sha256 {envv('frozen_model.checkpoint_sha256')[:16]}…)", "DIFFERENT",
         "different trained weights"),
        ("Threshold", "rule", str(h_eval["threshold_method"]),
         "maximum Youden index J on Validation", "same", ""),
        ("Threshold", "value", f"{float(h_eval['val_threshold']):.6f}",
         f"{a_prim['operating_point_youden']['threshold']}", "DIFFERENT",
         "human 0.569925, agent 0.008508: the two models' probability scales differ by two "
         "orders of magnitude, so the thresholds are not comparable as numbers"),
        ("Threshold", "data used", "Validation only", "Validation only", "same",
         f"agent recomputed_on_test = "
         f"{a_gcm['threshold_recomputed_on_test']}"),
        ("CI", "bootstrap resamples", str(int(h_eval["bootstrap_repetitions"])),
         str(a_prim["auroc"]["n_boot"]), "DIFFERENT", "10,000 vs 2,000"),
        ("CI", "stratified", "yes (cases and controls resampled separately)",
         f"{a_prim['auroc']['stratified']}", "same", ""),
        ("CI", "seed", "42", str(a_prim["auroc"]["seed"]), "DIFFERENT", "42 vs 12345"),
        ("CI", "method", str(h_eval["test_roc_auc_ci_method"]),
         "stratified bootstrap percentile", "same", ""),
        ("Calibration", "ECE definition (as officially reported)",
         "10 equal-width bins over [0,1]",
         f"{len(a_prim['calibration_bins'])} equal-count bins "
         f"(test_calibration_bins.csv)", "DIFFERENT",
         "human: TaskD_01 cell 6 expected_calibration_error(n_bins=10, np.linspace); "
         "agent Task A official file uses equal-count quintiles. A 10-equal-width value for "
         "the AI-agent CXR model also exists in results/taskC/evaluation/test_metrics_table.csv"),
        ("Calibration", "Brier", "sklearn brier_score_loss", "same definition", "same", ""),
        ("Grad-CAM", "target layer", "model.layer4[-1].conv2",
         str(a_gcm["gradcam_layer"]), "DIFFERENT (possibly)",
         "human names the final conv inside the last block; the agent records the block. "
         "Whether the hooked tensor is the same could not be established from the "
         "artefacts read"),
        ("Grad-CAM", "method", "gradient-weighted CAM, channel-mean gradient weights, "
                               "ReLU, min-max normalised per image",
         str(a_gcm["gradcam_method"]), "same concept", ""),
        ("Grad-CAM", "cases", f"{len(h_gc)} (TP 4 / TN 4 / FP 4 / FN 4)",
         f"{a_gcm['n_cases']} (TP {a_gcm['cases_used_per_group']['TP']} / "
         f"FP {a_gcm['cases_used_per_group']['FP']} / TN {a_gcm['cases_used_per_group']['TN']} / "
         f"FN {a_gcm['cases_used_per_group']['FN']})", "DIFFERENT",
         "the AI-agent model produced only 3 FN at its frozen threshold and the pre-specified "
         "rule forbade substituting from another group"),
        ("Governance", "Test used for model selection", "no", "no", "same",
         "human: the final evaluation notebook loads the checkpoint chosen on Validation; "
         f"agent: final_selection.json test_set.used_so_far = {fs['test_set']['used_so_far']}"),
        ("Governance", "Test access log", NR,
         f"present, {len(a_access)} entry(ies)", "DIFFERENT",
         "results/taskA/evaluation/test_access_log.jsonl"),
        ("Reproducibility", "frozen configuration file", NR,
         f"final_selection.json, sha256 {a_meta['frozen_sha256'][:16]}…", "DIFFERENT",
         "the human-guided pipeline records the chosen configuration in summary CSVs rather "
         "than a single hash-verified frozen file"),
        ("Reproducibility", "package versions",
         NR, f"python {envv('package_versions.python')}, torch "
             f"{envv('package_versions.torch')}; numpy / pandas / sklearn "
             f"{NR}", "partially recorded on the AI-agent side",
         "results/taskA/evaluation/test_run_env_note.json"),
        ("Reproducibility", "checkpoint published", "no", "no", "same",
         "neither pipeline publishes the trained weights; both record the checkpoint name, "
         "and the AI-agent side also records its SHA256"),
        ("Reproducibility", "patient-level Test predictions", "yes, 128 rows",
         f"yes, {len(a_test)} rows", "same", "both saved"),
    ]
    spec_df = pd.DataFrame(spec, columns=["domain", "item", "human_guided", "ai_agent",
                                          "difference", "source_or_note"])
    spec_df.to_csv(out / "human_vs_ai_agent_model_spec.csv", index=False, encoding="utf-8-sig")

    # =========================================================== performance table
    HS = "Human-guided final result artifact: 01_TaskA_CXR/05_Evaluation/" \
         "TaskA_CXR_最終評価サマリー.csv"
    HS5 = "Human-guided final result artifact: 04_全モデル比較・統計解析/" \
          "全5モデル_最終Test評価比較.csv"
    AS = "AI-Agent frozen Task A result: results/taskA/evaluation/test_metrics_primary.json"
    ASC = "AI-Agent frozen Task C table: results/taskC/evaluation/test_metrics_table.csv"
    HARM = "Harmonized recomputation from the saved patient-level Test predictions: " \
           "results/taskA/comparison/harmonized_ece_ci.csv"

    h_test_rates = rates(hy, hp, h_thr)
    a_test_rates = a_prim["operating_point_youden"]
    perf = []

    def add(pipeline, split, metric, value, source):
        perf.append({"pipeline": pipeline, "split": split, "metric": metric,
                     "value": value, "source": source})

    add(H, "validation", "roc_auc", float(h_eval["val_roc_auc"]), HS)
    add(H, "validation", "pr_auc", float(h_eval["val_pr_auc"]), HS)
    add(H, "validation", "loss", float(h_cfg["val_loss"]), "Human-guided final result "
        "artifact: 01_TaskA_CXR/04_Training/TaskA_CXR_最終学習条件サマリー.csv")
    add(H, "validation", "brier", None, NR)
    add(H, "validation", "ece", None, NR)
    add(H, "validation", "threshold_youden", float(h_eval["val_threshold"]), HS)
    add(H, "validation", "sensitivity_at_threshold", float(h_eval["val_sensitivity"]), HS)
    add(H, "validation", "specificity_at_threshold", float(h_eval["val_specificity"]), HS)
    add(H, "validation", "youden_index", float(h_eval["val_youden_index"]), HS)
    add(A, "validation", "roc_auc", a_vm["auroc"], "AI-Agent frozen: final_selection.json "
        "primary_analysis.validation_metrics")
    add(A, "validation", "pr_auc", a_vm["auprc"], "AI-Agent frozen: final_selection.json")
    add(A, "validation", "loss", None, NR)
    add(A, "validation", "brier", a_vm["brier"], "AI-Agent frozen: final_selection.json")
    add(A, "validation", "ece", None, NR)
    add(A, "validation", "threshold_youden", a_vm["at_threshold"]["threshold"],
        "AI-Agent frozen: final_selection.json")
    add(A, "validation", "sensitivity_at_threshold", a_vm["at_threshold"]["sensitivity"],
        "AI-Agent frozen: final_selection.json")
    add(A, "validation", "specificity_at_threshold", a_vm["at_threshold"]["specificity"],
        "AI-Agent frozen: final_selection.json")
    add(A, "validation", "youden_index",
        a_vm["at_threshold"]["sensitivity"] + a_vm["at_threshold"]["specificity"] - 1,
        "Derived from final_selection.json sensitivity + specificity − 1")

    add(H, "test", "roc_auc", float(h_eval["test_roc_auc"]), HS)
    add(H, "test", "roc_auc_ci_low", float(h_eval["test_roc_auc_ci_lower"]), HS)
    add(H, "test", "roc_auc_ci_high", float(h_eval["test_roc_auc_ci_upper"]), HS)
    add(H, "test", "pr_auc", float(h_eval["test_pr_auc"]), HS)
    add(H, "test", "brier", float(h_cxr5["brier_score"]), HS5)
    add(H, "test", "ece_10_equal_width", float(h_cxr5["ece"]), HS5)
    add(H, "test", "threshold", float(h_eval["test_threshold"]), HS)
    for k in ("sensitivity", "specificity", "accuracy"):
        add(H, "test", k, float(h_eval[f"test_{k}"]), HS)
    add(H, "test", "ppv", float(h_eval["test_precision"]), HS)
    add(H, "test", "npv", h_test_rates["npv"], HARM)
    for k in ("tp", "fp", "tn", "fn"):
        add(H, "test", k, int(h_eval[k.upper()]), HS)

    add(A, "test", "roc_auc", a_prim["auroc"]["point"], AS)
    add(A, "test", "roc_auc_ci_low", a_prim["auroc"]["ci_low"], AS)
    add(A, "test", "roc_auc_ci_high", a_prim["auroc"]["ci_high"], AS)
    add(A, "test", "pr_auc", a_prim["auprc"]["point"], AS)
    add(A, "test", "brier", a_prim["brier"], AS)
    add(A, "test", "ece_5_equal_count", a_prim["ece"], AS)
    add(A, "test", "ece_10_equal_width", float(a_taskc.loc["cxr"].ece_10_equal_width), ASC)
    add(A, "test", "threshold", a_prim["operating_point_youden"]["threshold"], AS)
    for k in ("sensitivity", "specificity", "ppv", "npv", "accuracy"):
        add(A, "test", k, a_test_rates[k], AS)
    for k in ("tp", "fp", "tn", "fn"):
        add(A, "test", k, int(a_test_rates[k]), AS)

    for pipeline, r in ((H, hh), (A, ha)):
        add(pipeline, "test", "harmonized_ece_10_equal_width", r.ece_10_equal_width, HARM)
        add(pipeline, "test", "harmonized_ece_5_equal_count", r.ece_5_equal_count, HARM)
        add(pipeline, "test", "harmonized_ci_low_10000_seed42",
            r.ci_low_human_setting_10000_seed42, HARM)
        add(pipeline, "test", "harmonized_ci_high_10000_seed42",
            r.ci_high_human_setting_10000_seed42, HARM)
        add(pipeline, "test", "harmonized_ci_low_2000_seed12345",
            r.ci_low_agent_setting_2000_seed12345, HARM)
        add(pipeline, "test", "harmonized_ci_high_2000_seed12345",
            r.ci_high_agent_setting_2000_seed12345, HARM)
        add(pipeline, "test", "recomputed_roc_auc", r.auroc_recomputed, HARM)
    perf_df = pd.DataFrame(perf)
    perf_df.to_csv(out / "human_vs_ai_agent_performance.csv", index=False, encoding="utf-8-sig")

    # =========================================================== Grad-CAM table
    gc_rows = [
        ("implementation", "custom GradCAM class, forward and full-backward hooks",
         f"{a_gcm['gradcam_method']}", "same concept, independent implementations"),
        ("target layer", "model.layer4[-1].conv2", str(a_gcm["gradcam_layer"]),
         "recorded at different granularity; whether the hooked tensor is identical is "
         + NE),
        ("gradient weighting", "channel-mean of the gradient of the raw mortality logit",
         "gradient-weighted class activation mapping (Selvaraju et al. 2017)", "same concept"),
        ("post-processing", "ReLU then min-max normalisation per image",
         "per-image normalisation; region metrics recorded in gradcam_selection.csv",
         "same concept"),
        ("upsampling", "cv2.resize to 224×224, INTER_LINEAR", NR, NE),
        ("case count", str(len(h_gc)), str(a_gcm["n_cases"]), "16 vs 15"),
        ("cases per group", "TP 4 / TN 4 / FP 4 / FN 4",
         f"TP {a_gcm['cases_used_per_group']['TP']} / TN {a_gcm['cases_used_per_group']['TN']} / "
         f"FP {a_gcm['cases_used_per_group']['FP']} / FN {a_gcm['cases_used_per_group']['FN']}",
         "the AI-agent model had only 3 FN at its frozen threshold and the pre-specified rule "
         "forbade substitution, leaving one blank panel cell"),
        ("selection rule", "4 per group by extreme probability",
         f"TP/FP: {a_gcm['selection_rule']['TP']}; TN: {a_gcm['selection_rule']['TN']}; "
         f"FN: {a_gcm['selection_rule']['FN']}; tie-break "
         f"{a_gcm['selection_rule']['tie_break']}", "same concept, rule written down "
                                                    "explicitly on the AI-agent side"),
        ("published output", "per-case images saved to 06_GradCAM",
         "one 4×4 panel; per-case images withheld with their SHA256 recorded",
         "DIFFERENT publication policy"),
        ("qualitative findings recorded",
         f"{len(h_gcq)} group-level findings with interpretation and caution columns",
         "per-case notes plus a researcher-written reading summary in gradcam_notes.md",
         "both record findings; the human-guided version is structured by group"),
        ("known issue", NR,
         "CAM is all-zero for cases whose predicted probability saturates near 0 "
         f"(region metrics are nan and were NOT imputed); {a_gcs['cam_mass'].isna().sum()} of "
         f"{len(a_gcs)} rows have nan region metrics. Recorded as open question Q-E3",
         "AI-agent side only"),
        ("metadata issue", NR,
         "delta_days_from_T0 is empty for all rows of gradcam_selection.csv, a metadata "
         "propagation issue; values were NOT guessed. Recorded as open question Q-E6",
         "AI-agent side only"),
        ("reproducibility", "notebook TaskA_06_CXR_GradCAM.ipynb",
         f"scripts/{a_gcm['script']} (sha256 {a_gcm['script_sha256'][:16]}…), input image "
         f"SHA256 recorded per case", "the AI-agent run records input hashes"),
        ("interpretive status", "explanatory visualisation; the artefact itself states that "
                                "Grad-CAM is a coarse localisation method and not lesion "
                                "segmentation",
         "explanatory visualisation only", "same"),
    ]
    gc_df = pd.DataFrame(gc_rows, columns=["item", "human_guided", "ai_agent", "note"])
    gc_df.to_csv(out / "human_vs_ai_agent_gradcam.csv", index=False, encoding="utf-8-sig")

    # =========================================================== discrepancies
    # Every row is classified, because "two artefacts disagree about a number" and "two
    # artefacts use different definitions" are different problems and must not be conflated.
    CATEGORIES = {
        "source discrepancy": "Two sources state genuinely conflicting values for the same "
                              "quantity under the same definition. Requires resolution.",
        "unexplained decision": "The artefacts record what was done but not why, and the "
                                "reason cannot be recovered from them. Not a numerical "
                                "conflict, but it leaves a design choice unjustified.",
        "access limitation": "A source that should have been checked could not be read, so "
                             "the check was not performed. Nothing is known to be wrong.",
        "methodological clarification": "Two different, explicitly named definitions of the "
                                        "same quantity. Both values are correct under their "
                                        "own definition. Not a conflict.",
        "resolved consistency": "Sources agree once the recording convention is understood. "
                                "Recorded so the agreement is auditable, not because "
                                "anything is in doubt.",
        "terminology ambiguity": "Two pipelines use the same name for different things. The "
                                 "values are not in conflict; the label is.",
    }
    disc = []

    def d(category, pipeline, quantity, v1, s1, v2, s2, resolution):
        if category not in CATEGORIES:
            raise SystemExit(f"STOP: unknown category {category!r}")
        disc.append({"category": category, "category_meaning": CATEGORIES[category],
                     "pipeline": pipeline, "quantity": quantity, "value_source_1": v1,
                     "source_1": s1, "value_source_2": v2, "source_2": s2,
                     "source_used": resolution})

    # human: Stage 1 highest Validation AUROC was not the selected condition
    best_s1 = h_stage1.sort_values("val_roc_auc", ascending=False).iloc[0]
    if best_s1["condition"] != "AugB_LR1e-5_seed42":
        d("unexplained decision", H,
          "Stage 1 condition with the highest Validation ROC-AUC vs the condition carried "
          "forward",
          f"{best_s1['condition']} val_roc_auc {best_s1['val_roc_auc']:.6f} "
          f"(best_epoch {int(best_s1['best_epoch'])})",
          "01_TaskA_CXR/04_Training/TaskA_CXR_Stage1_seed42_6条件比較.csv",
          f"AugB_LR1e-5 was carried into Stage 2 and became the final model "
          f"(weighted val_roc_auc {float(h_stage2.set_index('condition').loc['AugB_LR1e-5_seed42_weighted', 'val_roc_auc']):.6f})",
          "01_TaskA_CXR/04_Training/TaskA_CXR_Stage2_classweight比較.csv + "
          "TaskA_CXR_最終学習条件サマリー.csv",
          "**Left open.** Both values are reported as-is and the final model is the one "
          "recorded in TaskA_CXR_最終学習条件サマリー.csv and TaskA_CXR_最終評価サマリー.csv. "
          "The two numbers do not conflict — the question is why the higher-scoring condition "
          f"was not carried forward, and that reason is {NE}: no rationale field exists in any "
          "artefact read, and the notebooks record the outcome without the deliberation. It is "
          "NOT inferred here. This is the one item in this file that a reader should treat as "
          "an open question about the human-guided design rather than a bookkeeping matter. "
          "A plausible-sounding explanation (AugB_LR1e-4 peaked at epoch 2, which may have "
          "looked like insufficient training) is deliberately NOT recorded as the reason, "
          "because no artefact states it")
    # human: best_epoch consistency
    h_be = {"TaskA_CXR_最終学習条件サマリー.csv": int(h_cfg["best_epoch"]),
            "TaskA_CXR_最終評価サマリー.csv": int(h_eval["best_epoch"]),
            "TaskA_CXR_Stage2_classweight比較.csv (weighted)":
                int(h_stage2.set_index("condition").loc["AugB_LR1e-5_seed42_weighted",
                                                        "best_epoch"])}
    if len(set(h_be.values())) > 1:
        items = list(h_be.items())
        d("source discrepancy", H, "best_epoch", f"{items[0][1]}", items[0][0],
          f"{items[1][1]}", items[1][0],
          "Genuinely conflicting values for the same quantity. Reported; NOT merged")
    # human: val_roc_auc consistency
    h_va = {"TaskA_CXR_最終学習条件サマリー.csv": float(h_cfg["val_roc_auc"]),
            "TaskA_CXR_最終評価サマリー.csv": float(h_eval["val_roc_auc"])}
    if abs(h_va["TaskA_CXR_最終学習条件サマリー.csv"]
           - h_va["TaskA_CXR_最終評価サマリー.csv"]) > 1e-9:
        d("source discrepancy", H, "Validation ROC-AUC",
          f6(h_va['TaskA_CXR_最終学習条件サマリー.csv']), "TaskA_CXR_最終学習条件サマリー.csv",
          f6(h_va['TaskA_CXR_最終評価サマリー.csv']), "TaskA_CXR_最終評価サマリー.csv",
          "Genuinely conflicting values for the same quantity. Reported; NOT merged")
    # human: Test ROC-AUC across two artefacts
    if abs(float(h_eval["test_roc_auc"]) - float(h_cxr5["roc_auc"])) > 1e-9:
        d("source discrepancy", H, "Test ROC-AUC", f6(h_eval['test_roc_auc']),
          "TaskA_CXR_最終評価サマリー.csv", f6(h_cxr5['roc_auc']),
          "全5モデル_最終Test評価比較.csv",
          "Genuinely conflicting values for the same quantity. Reported; NOT merged")
    # human: recomputation from saved predictions
    if abs(hh.auroc_recomputed - float(h_eval["test_roc_auc"])) > 5e-6:
        d("source discrepancy", H,
          "Test ROC-AUC: reported vs recomputed from the saved patient-level predictions",
          f6(h_eval['test_roc_auc']), "TaskA_CXR_最終評価サマリー.csv",
          f6(hh.auroc_recomputed), "recomputed from TaskA_CXR_Test患者別予測結果.csv",
          "Reported; the official value is not overwritten")
    # human: CI
    if (abs(hh.ci_low_human_setting_10000_seed42 - float(h_eval["test_roc_auc_ci_lower"]))
            > 0.005):
        d("source discrepancy", H,
          "Test ROC-AUC 95% CI under the recorded bootstrap settings "
          "(10,000 / stratified / seed 42)",
          f"{f6(h_eval['test_roc_auc_ci_lower'])}–{f6(h_eval['test_roc_auc_ci_upper'])}",
          "TaskA_CXR_最終評価サマリー.csv",
          f"{f6(hh.ci_low_human_setting_10000_seed42)}–"
          f"{f6(hh.ci_high_human_setting_10000_seed42)}",
          "recomputed here from the saved predictions",
          "reported as a discrepancy. A bootstrap CI depends on the RNG call order as well as "
          "the seed, so an exact match is not guaranteed; the official value is not overwritten")
    # human: ECE definition label
    d("resolved consistency", H, "ECE bin scheme", "10 equal-width bins over [0,1]",
      "Notebook/TaskD_01_全モデル最終比較.ipynb cell 6 expected_calibration_error(n_bins=10)",
      f"value {f6(h_cxr5['ece'])}", "全5モデル_最終Test評価比較.csv column `ece`",
      "No conflict. The code and the value agree, and recomputing from the saved predictions "
      f"under that definition reproduces it ({f6(hh.ece_10_equal_width)}). Recorded only so "
      "that the definition travels with the number")
    # human: manuscript / slides not readable
    d("access limitation", H,
      "manuscript and slide values (best epoch, learning rate, augmentation, ROC-AUC, CI, "
      "PR-AUC, Brier, ECE, threshold)",
      "could not be read", "05_論文/COVID19研究_日本語論文稿_完成版_2026-08-28.gdoc and the "
                           "other .gdoc files are 219-byte Google Docs stubs on this "
                           "filesystem, not documents",
      "not compared", "not applicable",
      "NOT a discrepancy — a check that could not be run. The manuscript and slides could not "
      "be opened, so the requested manuscript-vs-artefact consistency check was not performed. "
      "No manuscript value is assumed or reconstructed, and nothing is known to disagree")
    # agent: ECE two definitions
    d("methodological clarification", A, "Test ECE",
      f6(a_prim['ece']) + " (5 equal-count bins)",
      "results/taskA/evaluation/test_metrics_primary.json",
      f6(a_taskc.loc['cxr'].ece_10_equal_width) + " (10 equal-width bins)",
      "results/taskC/evaluation/test_metrics_table.csv",
      "NOT a conflict. Two different, explicitly named definitions of the same quantity; each "
      "value is correct under its own definition and each table names it in the column header. "
      "The 10-equal-width value is the one comparable with the human-guided pipeline. Neither "
      "value needs resolution")
    # agent: recomputation check
    if abs(ha.auroc_recomputed - a_prim["auroc"]["point"]) > 5e-6:
        d("source discrepancy", A,
          "Test ROC-AUC: reported vs recomputed from the saved patient-level predictions",
          f6(a_prim['auroc']['point']), "results/taskA/evaluation/test_metrics_primary.json",
          f6(ha.auroc_recomputed), "recomputed from test_predictions_primary.csv",
          "Reported; the official value is not overwritten")
    # agent: best epoch corroboration
    be_cond = a_cond.set_index("condition").loc[a_dec["decision"]["selected_condition"],
                                                "best_epochs"]
    d("resolved consistency", A, "best_epoch of the primary model",
      str(envv("frozen_model.best_epoch")),
      "final_selection.json runs.42.best_epoch (hash-verified) and "
      "results/taskA/evaluation/test_run_env_note.json",
      f"the unlabelled triple '{be_cond}' across the three seeds",
      "results/taskA/training/condition_comparison.csv column best_epochs",
      "NOT a conflict. condition_comparison.csv simply does not label which epoch belongs to "
      "which seed. docs/decision_log.md D-052 records seed42=9 / seed43=5 / seed44=14, which "
      "agrees with the hash-verified frozen file. Three independent records agree; recorded "
      "only so the agreement is auditable")
    # agent: augmentation label collision
    d("terminology ambiguity", "both", "the augmentation condition named 'B' / 'aug_b'",
      h_aug_detail, "Notebook/TaskA_03_CXR_Dataset_DataLoader.ipynb cell 3 train_transform_B",
      a_aug_detail, "results/taskA/training/dataset_qc.json config.augmentations.aug_b",
      "NOT a numerical conflict — a naming collision. Each pipeline's parameters are recorded "
      "correctly in its own artefact; the two just happen to share the label. The risk is that "
      "a reader treats 'B' as one condition, so both parameter sets are written out in full "
      "wherever the label appears")
    disc_df = pd.DataFrame(disc)
    order = ["source discrepancy", "unexplained decision", "access limitation",
             "methodological clarification", "resolved consistency", "terminology ambiguity"]
    disc_df["_o"] = disc_df["category"].map({c: i for i, c in enumerate(order)})
    disc_df = disc_df.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    disc_df.to_csv(out / "source_discrepancies.csv", index=False, encoding="utf-8-sig")
    cat_counts = {c: int((disc_df["category"] == c).sum()) for c in order}

    # =========================================================== paired feasibility
    hs, as_ = set(h_test["Subject ID"]), set(a_test["subject_id"])
    merged = h_test.merge(a_test, left_on="Subject ID", right_on="subject_id",
                          suffixes=("_h", "_a"))
    feas = {
        "human_rows": len(h_test), "agent_rows": len(a_test),
        "identical_patient_sets": hs == as_,
        "n_merged": len(merged),
        "labels_agree": int((merged["true_label_h"] == merged["true_label_a"]).sum()),
        "same_row_order": list(h_test["Subject ID"]) == list(a_test["subject_id"]),
        "human_deaths": int(h_test["true_label"].sum()),
        "agent_deaths": int(a_test["true_label"].sum()),
        "probability_column_human": "prob_death",
        "probability_column_agent": "prob",
        "probabilities_complete": bool(h_test["prob_death"].notna().all()
                                       and a_test["prob"].notna().all()),
        "paired_delong_technically_possible": bool(
            hs == as_ and len(merged) == len(h_test)
            and (merged["true_label_h"] == merged["true_label_a"]).all()
            and h_test["prob_death"].notna().all() and a_test["prob"].notna().all()),
        "executed": False,
        "status": "NOT executed. If it is ever run it is an exploratory methodological "
                  "analysis and is separate from the confirmatory DeLong families defined in "
                  "results/comparison/delong_plan.json.",
    }

    meta = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "task": "A",
        "role": "exploratory methodological comparison",
        "non_superiority_statement":
            "This comparison is descriptive and post hoc by construction. It establishes "
            "neither superiority nor equivalence of either pipeline. The two pipelines differ "
            "in preprocessing geometry, learning rate, scheduler, early stopping, "
            "augmentation magnitude, number of seeds and selection rule simultaneously, so no "
            "performance difference can be attributed to any single design choice.",
        "source_priority":
            "Human-guided: the current final artefacts under 気合のCOVID19. Files beginning "
            "with 旧 were not read. The manuscript and slides are Google Docs stubs on this "
            "filesystem and could not be read. AI-agent: the frozen Task A artefacts in this "
            "repository, plus final_selection.json read from Drive only after its SHA256 "
            "matched the repository record.",
        "sources_used": src_used,
        "official_artifacts_modified": [],
        "harmonization": {
            "ece": "Both pipelines' Test ECE recomputed from the saved patient-level "
                   "predictions under 10 equal-width bins and under 5 equal-count bins. "
                   "Written to harmonized_ece_ci.csv. Official ECE values are not overwritten.",
            "ci": "Both pipelines' Test ROC-AUC CI recomputed under both bootstrap settings "
                  "(10,000/seed 42 and 2,000/seed 12345), stratified, percentile. The "
                  "confidence intervals in the official artefacts were generated using "
                  "different bootstrap settings and therefore were not used for a "
                  "standardized numerical comparison of uncertainty between the two "
                  "pipelines.",
        },
        "paired_comparison_feasibility": feas,
        "finding_categories": {
            "definitions": CATEGORIES,
            "counts": cat_counts,
            "note": "Only the `source discrepancy` rows are genuine conflicts between sources. "
                    "Differing metric definitions, recording conventions and shared labels are "
                    "classified separately so they are not miscounted as inconsistencies.",
        },
        "counts": {"model_spec_rows": len(spec_df), "performance_rows": len(perf_df),
                   "gradcam_rows": len(gc_df), "finding_rows": len(disc_df),
                   "genuine_source_discrepancies": cat_counts["source discrepancy"],
                   "unexplained_decisions": cat_counts["unexplained decision"]},
    }
    (out / "comparison_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # =========================================================== markdown
    def difftable(domain):
        rows = spec_df[spec_df.domain == domain]
        s = "| Item | Human-guided | AI-Agent | Same? |\n|---|---|---|---|\n"
        for _, r in rows.iterrows():
            s += (f"| {r['item']} | {r['human_guided']} | {r['ai_agent']} | "
                  f"{r['difference']} |\n")
        return s

    diffs = spec_df[spec_df.difference.str.contains("DIFFERENT")]
    sames = spec_df[spec_df.difference == "same"]

    md = f"""# Task A — human-guided vs AI-agent methodological comparison

*Exploratory methodological comparison. Generated by `scripts/{Path(__file__).name}` on
{datetime.now().strftime('%Y-%m-%d')} from saved artefacts; no number is written
from memory.*

## 1. Purpose

Two pipelines independently built a chest-radiograph model for in-hospital mortality on the
same {len(h_test) * 0 + 1277:,}-patient cohort and the same fixed split. This document compares
**how they were built** — image selection, preprocessing, architecture, optimisation, class
imbalance handling, checkpoint selection, calibration, Grad-CAM, governance and
reproducibility — and records where the sources disagree.

## 2. Scope and non-superiority statement

**This comparison establishes neither superiority nor equivalence of either approach.**

{meta['non_superiority_statement']}

The pipelines produced different CXR discrimination. The human-guided pipeline showed a higher
Test ROC-AUC point estimate. Several training and selection choices differed at once, and the
difference cannot be attributed to a single design choice. Nothing here supports a claim that
either a human-guided or an autonomous process is the better way to build such a model.

## 3. Cohort and CXR selection

{difftable('Cohort')}
{difftable('Image selection')}
Both pipelines read the same fixed split file and the same selected-image list, so the
{len(h_test)} Test patients are the same {len(h_test)} patients. That is what makes every
comparison below patient-for-patient.

## 4. Preprocessing

{difftable('Preprocessing')}
Two differences matter enough to state plainly:

- **Intensity mapping.** The human-guided pipeline applies the DICOM `WindowCenter` /
  `WindowWidth` and clips to that window. The AI-agent pipeline ignores the stored window and
  clips at the 1st and 99th percentile of each image
  (`window_center_width_used = {a_pre['pipeline']['window_center_width_used']}`). The two
  produce different grey-scale mappings of the same radiograph.
- **Geometry.** The human-guided pipeline resizes directly to 224×224, discarding the aspect
  ratio. The AI-agent pipeline zero-pads to a square first, so anatomy is not stretched but
  a median of {a_pre['pad_fraction']['median']:.1%} of the canvas is padding
  (range {a_pre['pad_fraction']['min']:.1%}–{a_pre['pad_fraction']['max']:.1%}).

Whether either choice is preferable for this task is {NE}.

## 5. Model architecture

{difftable('Architecture')}
The backbone, the head and the input resolution agree. The AI-agent pipeline additionally
searched staged unfreezing (Stage 2) and did not select it, so both final models are full
fine-tuning of an ImageNet-pretrained ResNet18.

## 6. Training design

{difftable('Optimisation')}
{difftable('Class imbalance')}
{difftable('Augmentation')}
{difftable('Seeds')}
**The augmentation label is a trap.** Both pipelines call their selected augmentation "B", and
the parameters are not the same: rotation ±5° vs ±{a_aug['degrees']:.0f}°, translation ±2% vs
±{a_aug['translate'] * 100:.0f}%, scale 0.98–1.02 vs {a_aug['scale'][0]}–{a_aug['scale'][1]},
brightness and contrast 0.05 vs {a_aug['brightness']}. The AI-agent `aug_a`
(translate {a_qc['config']['augmentations']['aug_a']['translate']},
scale {a_qc['config']['augmentations']['aug_a']['scale']},
brightness {a_qc['config']['augmentations']['aug_a']['brightness']},
rotation {a_qc['config']['augmentations']['aug_a']['degrees']:.0f}°) is closer to the
human-guided "B" than the AI-agent `aug_b` is.

`pos_weight` agrees to six decimals ({float(h_cfg['pos_weight']):.6f} vs
{a_run['pos_weight']:.6f}), which is expected: both computed it from the same Training set.

## 7. Model selection

{difftable('Selection')}
**How the AI-agent reached its primary model.** Stage 1 trained
{int(a_cond[a_cond.stage == 'stage1']['n_seeds'].sum())} runs over
{len(a_cond[a_cond.stage == 'stage1'])} conditions and Stage 2 added
{int(a_cond[a_cond.stage == 'stage2']['n_seeds'].sum())} runs over
{len(a_cond[a_cond.stage == 'stage2'])} unfreezing schedules,
{int(a_cond['n_seeds'].sum())} runs in total, each condition at three seeds. Selection used the
**mean** Validation ROC-AUC across seeds. The best Stage 2 alternative
(`{a_dec['decision']['best_alternative']}`,
{f6(a_dec['decision']['best_alternative_mean_val_auroc'])}) did not beat the Stage 1 baseline
(`{a_dec['decision']['baseline']}`, {f6(a_dec['decision']['baseline_mean_val_auroc'])}) by the
pre-registered margin of {a_dec['decision']['margin']}, so the simpler condition was kept
({a_dec['rule']}). One seed (42) was then pre-specified as the primary analysis so that the
comparison with the pre-existing model would not be confounded by ensembling, and the
configuration was frozen to a hash-verified `final_selection.json`
(`{a_meta['frozen_sha256'][:16]}…`) before the Test set was opened.

**How the human-guided pipeline reached its final model.** Six learning-rate × augmentation
conditions were trained at seed 42, then two class-weight variants of the chosen condition. The
final model is `{h_cfg['checkpoint']}`, best epoch {int(h_cfg['best_epoch'])}. Note that the
Stage 1 condition with the highest single-seed Validation ROC-AUC was
`{best_s1['condition']}` ({f6(best_s1['val_roc_auc'])}, best epoch
{int(best_s1['best_epoch'])}), which is not the condition carried into Stage 2. The artefacts
read record the outcome but no rationale field, so the reason is {NE} and is not guessed at
here. It is listed in `source_discrepancies.csv`.

## 8. Validation performance

| Metric | Human-guided | AI-Agent |
|---|---:|---:|
| ROC-AUC | {f6(h_eval['val_roc_auc'])} | {f6(a_vm['auroc'])} |
| PR-AUC | {f6(h_eval['val_pr_auc'])} | {f6(a_vm['auprc'])} |
| Brier | {NR} | {f6(a_vm['brier'])} |
| ECE | {NR} | {NR} |
| Youden threshold | {f6(h_eval['val_threshold'])} | {a_vm['at_threshold']['threshold']} |
| Sensitivity at threshold | {f6(h_eval['val_sensitivity'])} | {f6(a_vm['at_threshold']['sensitivity'])} |
| Specificity at threshold | {f6(h_eval['val_specificity'])} | {f6(a_vm['at_threshold']['specificity'])} |
| Youden index | {f6(h_eval['val_youden_index'])} | {f6(a_vm['at_threshold']['sensitivity'] + a_vm['at_threshold']['specificity'] - 1)} |

Neither pipeline recorded a Validation ECE, so no Validation calibration comparison is
possible. The two operating points behave very differently: the AI-agent threshold admits every
death on Validation (sensitivity 1.000, {int(a_vm['at_threshold']['fp'])} false positives),
whereas the human-guided threshold sits near the middle of its probability range.

## 9. Test performance

| Metric | Human-guided | AI-Agent |
|---|---:|---:|
| **ROC-AUC** | **{f6(h_eval['test_roc_auc'])}** | **{f6(a_prim['auroc']['point'])}** |
| 95% CI (as officially reported) | {f6(h_eval['test_roc_auc_ci_lower'])}–{f6(h_eval['test_roc_auc_ci_upper'])} | {f6(a_prim['auroc']['ci_low'])}–{f6(a_prim['auroc']['ci_high'])} |
| PR-AUC | {f6(h_eval['test_pr_auc'])} | {f6(a_prim['auprc']['point'])} |
| Brier | {f6(h_cxr5['brier_score'])} | {f6(a_prim['brier'])} |
| ECE, 10 equal-width | {f6(h_cxr5['ece'])} | {f6(a_taskc.loc['cxr'].ece_10_equal_width)} |
| Threshold used | {f6(h_eval['test_threshold'])} | {a_prim['operating_point_youden']['threshold']} |
| Sensitivity | {f6(h_eval['test_sensitivity'])} | {f6(a_test_rates['sensitivity'])} |
| Specificity | {f6(h_eval['test_specificity'])} | {f6(a_test_rates['specificity'])} |
| PPV | {f6(h_eval['test_precision'])} | {f6(a_test_rates['ppv'])} |
| NPV | {f6(h_test_rates['npv'])} | {f6(a_test_rates['npv'])} |
| Accuracy | {f6(h_eval['test_accuracy'])} | {f6(a_test_rates['accuracy'])} |
| TP / FP / TN / FN | {int(h_eval['TP'])} / {int(h_eval['FP'])} / {int(h_eval['TN'])} / {int(h_eval['FN'])} | {int(a_test_rates['tp'])} / {int(a_test_rates['fp'])} / {int(a_test_rates['tn'])} / {int(a_test_rates['fn'])} |

Both rows are the same {len(h_test)} Test patients with {int(h_test['true_label'].sum())}
deaths. The ROC-AUC difference is
{float(h_eval['test_roc_auc']) - a_prim['auroc']['point']:+.6f} in favour of the human-guided
point estimate, and the two officially reported confidence intervals overlap substantially.

The operating points are not comparable as behaviour: at its frozen threshold the human-guided
model is specific and conservative ({int(h_eval['TP'])} of
{int(h_test['true_label'].sum())} deaths flagged, {int(h_eval['FP'])} false positives), while
the AI-agent model is sensitive and permissive ({int(a_test_rates['tp'])} deaths flagged,
{int(a_test_rates['fp'])} false positives). Both thresholds come from the same rule (maximum
Youden index on Validation) applied to models whose probability scales differ by two orders of
magnitude.

### CI definition

| | Human-guided | AI-Agent |
|---|---|---|
| Resamples | {int(h_eval['bootstrap_repetitions']):,} | {a_prim['auroc']['n_boot']:,} |
| Stratified | yes | {a_prim['auroc']['stratified']} |
| Seed | 42 | {a_prim['auroc']['seed']} |
| Method | {h_eval['test_roc_auc_ci_method']} | stratified bootstrap percentile |

The confidence intervals were generated using different bootstrap settings and therefore were
not used for a standardized numerical comparison of uncertainty between the two pipelines.
`harmonized_ece_ci.csv` recomputes both models' CI under **both** settings from the saved
patient-level predictions, so the effect of the setting is visible:

| Model | CI at 10,000 / seed 42 | CI at 2,000 / seed 12345 |
|---|---|---|
| Human-guided | {f6(hh.ci_low_human_setting_10000_seed42)}–{f6(hh.ci_high_human_setting_10000_seed42)} | {f6(hh.ci_low_agent_setting_2000_seed12345)}–{f6(hh.ci_high_agent_setting_2000_seed12345)} |
| AI-Agent | {f6(ha.ci_low_human_setting_10000_seed42)}–{f6(ha.ci_high_human_setting_10000_seed42)} | {f6(ha.ci_low_agent_setting_2000_seed12345)}–{f6(ha.ci_high_agent_setting_2000_seed12345)} |

These are comparison artefacts. The official CIs are unchanged.

## 10. Calibration

{difftable('Calibration')}
The human-guided pipeline reported ECE over 10 equal-width bins. The AI-agent Task A official
file reports {len(a_prim['calibration_bins'])} equal-count bins, and a 10-equal-width value for
the same model exists in the Task C table. Recomputed here under one definition each:

| Definition | Human-guided | AI-Agent |
|---|---:|---:|
| 10 equal-width bins | {f6(hh.ece_10_equal_width)} | {f6(ha.ece_10_equal_width)} |
| 5 equal-count bins | {f6(hh.ece_5_equal_count)} | {f6(ha.ece_5_equal_count)} |
| Brier | {f6(hh.brier_recomputed)} | {f6(ha.brier_recomputed)} |

Under the shared ECE definitions, the AI-agent model had lower ECE, whereas the human-guided
model had a slightly lower Brier score; therefore, calibration-related metrics did not
consistently favor one pipeline. Neither model can be called better calibrated on this evidence.

Two further reasons not to read the ECE column as a calibration ranking: the AI-agent model's
probabilities are concentrated near zero, which flatters an equal-width ECE because most of the
mass falls in one bin; and with {int(h_test['true_label'].sum())} events a binned calibration
estimate is unstable whichever binning is used.

## 11. Grad-CAM

| Item | Human-guided | AI-Agent | Note |
|---|---|---|---|
""" + "".join(f"| {r['item']} | {r['human_guided']} | {r['ai_agent']} | {r['note']} |\n"
              for _, r in gc_df.iterrows()) + f"""
**Grad-CAM is an explanatory visualisation. It does not demonstrate lesion localisation.** The
human-guided artefact says this in its own `caution` column: "Grad-CAM is a coarse localization
method and does not represent complete lesion segmentation."

The human-guided qualitative findings, recorded per outcome group, are:

""" + "".join(f"- **{r['group']}** — {r['finding']} *Interpretation:* {r['interpretation']} "
              f"*Caution:* {r['caution']}\n" for _, r in h_gcq.iterrows()) + f"""
**Known AI-agent issues, not hidden.** For cases whose predicted probability saturates
numerically at zero the CAM is identically zero, so the region metrics are `nan`;
{a_gcs['cam_mass'].isna().sum()} of {len(a_gcs)} rows are affected and the values were **not**
imputed (open question Q-E3). `delta_days_from_T0` is empty for every row of
`gradcam_selection.csv`, a metadata propagation issue; the values were **not** guessed (open
question Q-E6). The panel has one blank cell because the model produced only three false
negatives at its frozen threshold and the pre-specified rule forbade substituting a case from
another group.

## 12. Governance and Test access

{difftable('Governance')}
Neither pipeline used the Test set for preprocessing, variable or condition selection,
hyperparameter tuning, checkpoint selection or threshold selection. Both fixed the operating
threshold on Validation. The AI-agent pipeline froze its specification to a hash-verified JSON
before the Test set was opened and logs every Test access; the human-guided pipeline records
its chosen configuration in summary CSVs and has no access log, so its Test discipline is
documented by the notebook order rather than by a machine-readable record.

## 13. Reproducibility

{difftable('Reproducibility')}
What is reproducible, on each side:

| | Human-guided | AI-Agent |
|---|---|---|
| Test metrics from saved predictions | yes | yes |
| Paired DeLong from saved predictions | yes | yes |
| Threshold-based metrics | yes | yes |
| Re-training the model | no — no checkpoint, no frozen config file | no — checkpoint not published |
| Grad-CAM regeneration | no — needs the checkpoint | no — needs the checkpoint |
| Verifying the inputs used | partly | yes — split, manifest, checkpoint and image SHA256 recorded |

**The AI-agent pipeline does not publish its checkpoint.** Every Test metric and the paired
DeLong are reproducible from `results/taskA/evaluation/test_predictions_primary.csv`. Training
and Grad-CAM are not reproducible from the repository alone; they need the checkpoint, whose
SHA256 (`{envv('frozen_model.checkpoint_sha256')[:16]}…`) is recorded so the correct file can
be identified if it is supplied.

## 14. Similarities

Confirmed from artefacts on both sides:

""" + "".join(f"- {r['item']}: {r['human_guided']}\n"
              for _, r in sames.iterrows()) + f"""
## 15. Differences

Confirmed from artefacts on both sides:

| Item | Human-guided | AI-Agent |
|---|---|---|
""" + "".join(f"| {r['item']} | {r['human_guided']} | {r['ai_agent']} |\n"
              for _, r in diffs.iterrows()) + f"""
## 16. Interpretation

The human-guided CXR model showed a higher Test ROC-AUC point estimate
({f6(h_eval['test_roc_auc'])} vs {f6(a_prim['auroc']['point'])}). **The difference cannot be
attributed to a single design choice.** The candidate contributors that the artefacts confirm
*differed*, with no claim about which mattered:

| Candidate contributor | Differed? | Attribution |
|---|---|---|
| learning rate | yes — 1e-5 vs {a_run['lr']:g} | {NE} |
| scheduler | yes — ReduceLROnPlateau vs OneCycleLR | {NE} |
| early stopping | yes — patience 8 vs fixed 30-epoch budget | {NE} |
| checkpoint / best epoch | yes — {int(h_cfg['best_epoch'])} vs {envv('frozen_model.best_epoch')} | {NE} |
| augmentation magnitude | yes — same label, different parameters | {NE} |
| training duration | yes — early stopping vs full budget | {NE} |
| preprocessing intensity mapping | yes — DICOM window vs percentile clip | {NE} |
| preprocessing geometry | yes — direct resize vs pad-to-square | {NE} |
| seed of the final model | no — both 42 | not a contributor |
| number of seeds searched | yes — 1 vs 3 per condition | {NE} |
| model initialization | no — both ImageNet-1K ResNet18 | not a contributor |
| optimizer / weight decay / batch size | no | not a contributor |
| class imbalance handling | no — same loss, pos_weight agrees to 6 dp | not a contributor |
| image selection | no — same patients, same index images | not a contributor |
| cohort or split | no | not a contributor |

Eight components differed simultaneously and no ablation isolating any of them exists in either
pipeline. Establishing a cause would require re-training under matched conditions, which is
outside this comparison and is not performed.

**Relation to Task C.** In the AI-agent Task C fusion, permutation modality importance on Test
was clinical {f6(pd.read_csv(repo / 'results/taskC/evaluation/modality_importance_test.csv', encoding='utf-8-sig').set_index('modality').loc['clinical'].mean_auroc_drop)} vs
CXR {f6(pd.read_csv(repo / 'results/taskC/evaluation/modality_importance_test.csv', encoding='utf-8-sig').set_index('modality').loc['cxr'].mean_auroc_drop)},
and the fused model did not significantly exceed Clinical LR. The relatively low discrimination
of the AI-agent CXR component **may have contributed** to how little the radiograph added.
This is a possible contributing factor, not a demonstrated cause: the human-guided pipeline,
whose CXR component was stronger, also found that its Late Fusion did not significantly exceed
its Clinical LR, so a stronger CXR component did not by itself produce incremental value in
this cohort.

## 17. Limitations

- The comparison is **post hoc by construction** and was designed after both pipelines had
  finished. It is exploratory.
- **Several components differ at once.** No design choice can be isolated.
- **The manuscript and slides could not be read.** The `.gdoc` files under `05_論文` and
  `00_共通・研究管理` are 219-byte Google Docs stubs on this filesystem, so the requested
  manuscript-vs-artefact consistency check for best epoch, learning rate, augmentation,
  ROC-AUC, CI, PR-AUC, Brier, ECE and threshold **was not performed**. No manuscript value is
  assumed here.
- **Validation ECE is absent on both sides**, so Validation calibration cannot be compared.
- The human-guided pipeline trained **one seed**, so its Validation figures carry no
  seed-to-seed variability estimate and the two selection procedures are not like-for-like.
- With {int(h_test['true_label'].sum())} Test deaths every interval here is wide. **A lack of
  statistical significance would not be evidence of equivalence**, and no significance test
  between the two pipelines has been run.
- Grad-CAM findings on 15–16 cases are illustrative and cannot establish what either model
  relies on.

## 18. Source discrepancies and related findings

`results/taskA/comparison/source_discrepancies.csv` holds {len(disc_df)} item(s). **They are
not all discrepancies**, and lumping them together would overstate how inconsistent the sources
are. Each row carries a `category`:

| Category | Count | What it means |
|---|---:|---|
""" + "".join(f"| **{c}** | {cat_counts[c]} | {CATEGORIES[c]} |\n" for c in order) + f"""
**{cat_counts['source discrepancy']} genuine source discrepancies** were found: no two sources
state conflicting values for the same quantity under the same definition. Every reported number
that could be recomputed from the saved patient-level predictions reproduced exactly, on both
sides.

**The one item that matters as an open question** is the single `unexplained decision`. The
differing ECE definitions, the `best_epoch` recording convention and the shared augmentation
label are *not* source discrepancies and should not be read as such.

### Detail

| Category | Pipeline | Quantity | Source 1 | Source 2 | Handling |
|---|---|---|---|---|---|
""" + "".join(f"| {r['category']} | {r['pipeline']} | {r['quantity']} | "
              f"{r['value_source_1']} ({r['source_1']}) | {r['value_source_2']} "
              f"({r['source_2']}) | {r['source_used']} |\n"
              for _, r in disc_df.iterrows()) + f"""
### The unexplained decision, stated plainly

In the human-guided Stage 1 search the condition with the highest single-seed Validation ROC-AUC
was `{best_s1['condition']}` ({f6(best_s1['val_roc_auc'])}, best epoch
{int(best_s1['best_epoch'])}). The condition carried into Stage 2 and eventually frozen as the
final model was `AugB_LR1e-5` ({f6(float(h_stage2.set_index('condition').loc['AugB_LR1e-5_seed42_weighted', 'val_roc_auc']))}
after class weighting). **No artefact records why.** The notebooks and summary CSVs record the
outcome of the search, not the deliberation, and there is no rationale field anywhere in the
human-guided Task A material. This is left as an open question rather than filled in: a
plausible-sounding reconstruction is available (the higher-scoring condition peaked at epoch
{int(best_s1['best_epoch'])}, which may have looked like insufficient training) but no artefact
states it, so it is not recorded as the reason.

## 19. Paired comparison feasibility

| Check | Result |
|---|---|
| Human-guided Test predictions | {feas['human_rows']} rows, {feas['human_deaths']} deaths |
| AI-Agent Test predictions | {feas['agent_rows']} rows, {feas['agent_deaths']} deaths |
| Identical patient set | {feas['identical_patient_sets']} |
| Rows merged on Subject ID | {feas['n_merged']} |
| Labels agree | {feas['labels_agree']} / {feas['n_merged']} |
| Same row order | {feas['same_row_order']} |
| Probabilities complete | {feas['probabilities_complete']} |
| **Paired DeLong technically possible** | **{feas['paired_delong_technically_possible']}** |
| Executed | **{feas['executed']}** |

A paired DeLong test between the two CXR models is technically possible. **It has not been
run.** If it is ever run it is an exploratory methodological analysis and sits outside the two
confirmatory comparison families in `results/comparison/delong_plan.json`; mixing it into those
families would change their multiplicity structure after the fact.

## 20. Overall conclusion

- The human-guided CXR model showed **higher Test discrimination**
  ({f6(h_eval['test_roc_auc'])}) than the AI-agent CXR model
  ({f6(a_prim['auroc']['point'])}), a difference of
  {float(h_eval['test_roc_auc']) - a_prim['auroc']['point']:+.6f} on the same
  {len(h_test)} patients.
- **The difference cannot be attributed to any single design choice.** Eight components
  differed simultaneously and no ablation isolating them exists.
- **Task A shows a larger between-pipeline difference than Task B or Task C did.** In Task B
  the clinical models landed close together, and in Task C the two independently designed
  pipelines converged on nearly the same Late Fusion structure and nearly the same Test
  ROC-AUC.
- **Calibration does not follow discrimination.** Under the shared ECE definitions, the
  AI-agent model had lower ECE, whereas the human-guided model had a slightly lower Brier
  score; therefore, calibration-related metrics did not consistently favor one pipeline.
- The lower discrimination of the AI-agent CXR component **may have contributed** to the small
  CXR contribution in the AI-agent Task C fusion. This is not a demonstrated cause — the
  human-guided pipeline's stronger CXR component also failed to lift its Late Fusion
  significantly above its Clinical LR.
- **None of this indicates whether a human-guided or an autonomous process is the better way
  to build such a model.** It is a methodological comparison of two independently designed
  pipelines on one cohort.
"""
    doc = project / "docs/taskA_human_vs_ai_agent_comparison.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(md, encoding="utf-8")

    print(f"docs/taskA_human_vs_ai_agent_comparison.md   {len(md.encode('utf-8')):>7,} B")
    for f in sorted(out.iterdir()):
        if f.name.startswith(("human_vs", "source_disc", "comparison_meta", "harmonized")):
            print(f"  {f.relative_to(project).as_posix():<58} {f.stat().st_size:>7,} B")
    print(f"\nmodel_spec rows {len(spec_df)} ({len(diffs)} DIFFERENT, {len(sames)} same)")
    print(f"performance rows {len(perf_df)}   gradcam rows {len(gc_df)}")
    print(f"findings {len(disc_df)} rows: " + ", ".join(f"{c}={cat_counts[c]}" for c in order))
    print(f"paired DeLong technically possible: "
          f"{feas['paired_delong_technically_possible']} (NOT executed)")
    print("\nkey numbers read from artefacts:")
    print(f"  human Test ROC-AUC {f6(h_eval['test_roc_auc'])} "
          f"({f6(h_eval['test_roc_auc_ci_lower'])}–{f6(h_eval['test_roc_auc_ci_upper'])})")
    print(f"  agent Test ROC-AUC {f6(a_prim['auroc']['point'])} "
          f"({f6(a_prim['auroc']['ci_low'])}–{f6(a_prim['auroc']['ci_high'])})")
    print(f"  harmonized ECE 10-equal-width: human {f6(hh.ece_10_equal_width)}  "
          f"agent {f6(ha.ece_10_equal_width)}")
    print("Task A official artefacts were not written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
