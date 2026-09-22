"""Grad-CAM for the frozen Task A primary model on 16 representative Test cases.

This is explanation only. It does not train, does not select a model, does not recompute a
threshold and does not touch the Test evaluation itself: the primary model, the checkpoint and
the Youden threshold all come from final_selection.json, and the predictions come from the
Test predictions file produced by the single Test evaluation.

Case selection (fixed in advance, D-045; amended 2026-09-21, D-061):
    TP : highest predicted probability, 4 cases
    FP : highest predicted probability, 4 cases
    TN : lowest  predicted probability, 4 cases
    FN : lowest  predicted probability, min(4, available) cases
    ties broken by Subject ID ascending; no case is substituted by hand and no other group is
    enlarged to reach 16. The Test set contains only 3 FN cases at the frozen threshold, so all
    3 are used and the total is 15; the 4x4 panel keeps one empty cell.

Target layer: ResNet18 `layer4[-1]` (models/taskA_resnet18.gradcam_target_layer).

Everything is written into a temporary staging directory first and moved to --out-dir only
after all 16 cases succeed, so a failure never leaves a half-finished deliverable.

Usage:
    python scripts/18_taskA_gradcam_primary_test.py \
        --frozen  "<...>/AIagent_taskA_runs/final_selection.json" \
        --test-predictions "<...>/AIagent_taskA_test/test_predictions_primary.csv" \
        --image-dir "<...>/AIagent_taskA_png512_16bit/images" \
        --out-dir "<...>/06_GradCAM/AIagent_taskA_primary_test"
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.data.taskA_dataset import TaskACXRDataset, load_index  # noqa: E402
from covid_mortality.evaluation.gradcam import (  # noqa: E402
    GradCAM, annotate, grid, overlay, region_metrics, side_by_side)
from covid_mortality.models.taskA_resnet18 import (  # noqa: E402
    ModelConfig, build_model, gradcam_target_layer)

EXPECTED_CONDITION = "lr3e-4_aug_b"
EXPECTED_SEED = 42
N_PER_GROUP = 4
GROUPS = ("TP", "FP", "TN", "FN")
# minimum number of cases required per group before the run may proceed (D-061):
# TP/FP/TN must have 4; FN may have fewer and then all available FN cases are used.
MIN_PER_GROUP = {"TP": 4, "FP": 4, "TN": 4, "FN": 1}
PANEL_CELLS = 16


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_cases(pred: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, dict]:
    """Apply the fixed selection rule; returns the selected rows and the available group sizes.

    TP/FP/TN take 4 cases each; FN takes min(4, available). No group is enlarged to compensate
    for a small FN group.
    """
    df = pred.copy()
    df["predicted_label"] = (df.prob >= threshold).astype(int)
    df["outcome_group"] = np.select(
        [(df.true_label == 1) & (df.predicted_label == 1),
         (df.true_label == 0) & (df.predicted_label == 1),
         (df.true_label == 0) & (df.predicted_label == 0),
         (df.true_label == 1) & (df.predicted_label == 0)],
        ["TP", "FP", "TN", "FN"], default="?")
    available = {g: int((df.outcome_group == g).sum()) for g in GROUPS}
    chosen = []
    for group in GROUPS:
        g = df[df.outcome_group == group]
        ascending = group in ("TN", "FN")           # TN/FN: lowest probability first
        take = min(N_PER_GROUP, available[group])   # FN may legitimately be fewer than 4
        g = g.sort_values(["prob", "subject_id"], ascending=[ascending, True]).head(take)
        g = g.assign(selection_rank_within_group=range(1, len(g) + 1))
        chosen.append(g)
    return pd.concat(chosen).reset_index(drop=True), available


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--frozen", required=True, help="final_selection.json")
    ap.add_argument("--test-predictions", required=True)
    ap.add_argument("--image-dir", required=True, help="16-bit PNG cache (AI-agent)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-overwrite", action="store_true")
    args = ap.parse_args()

    project = Path(args.project)
    frozen_path, pred_path = Path(args.frozen), Path(args.test_predictions)
    image_dir, out_dir = Path(args.image_dir), Path(args.out_dir)
    warnings: list[str] = []
    progress: list[str] = []

    def stop(msg: str) -> int:
        print(f"STOP: {msg}")
        print("progress: " + (" -> ".join(progress) if progress else "nothing completed"))
        return 1

    # ---------- pre-flight -----------------------------------------------------------
    if not frozen_path.exists():
        return stop(f"final_selection.json not found: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    primary = frozen["primary_analysis"]
    model_name = primary["model"]
    if not model_name.startswith(EXPECTED_CONDITION) or not model_name.endswith(f"seed{EXPECTED_SEED}"):
        return stop(f"primary model in the frozen file is {model_name!r}, expected "
                    f"{EXPECTED_CONDITION}/seed{EXPECTED_SEED}")
    threshold = primary.get("threshold", {}).get("value")
    if threshold is None:
        return stop("threshold (Youden) missing from final_selection.json")
    checkpoint = Path(primary["checkpoint"])
    if not checkpoint.exists():
        return stop(f"checkpoint not found: {checkpoint}")
    ckpt_hash = sha256(checkpoint)
    if ckpt_hash != primary["checkpoint_sha256"]:
        return stop(f"checkpoint hash changed: {checkpoint}")
    progress.append("frozen selection verified")

    if not pred_path.exists():
        return stop(f"Test predictions not found: {pred_path}")
    pred = pd.read_csv(pred_path, encoding="utf-8-sig")
    for col in ("subject_id", "true_label", "prob"):
        if col not in pred.columns:
            return stop(f"{pred_path.name}: missing column {col!r}; has {list(pred.columns)}")
    pred["subject_id"] = pred.subject_id.astype(str).str.strip()
    pred["true_label"] = pred.true_label.astype(int)
    if len(pred) != 128 or not pred.subject_id.is_unique:
        return stop(f"expected 128 unique Test patients, got {len(pred)} rows "
                    f"({pred.subject_id.nunique()} unique)")
    if set(pred.true_label.unique()) - {0, 1}:
        return stop("true_label must be 0/1")
    progress.append(f"test predictions loaded (n={len(pred)}, events={int(pred.true_label.sum())})")

    # thresholds are never recomputed here: the frozen value is applied as is
    cases, available = select_cases(pred, float(threshold))
    short = {g: available[g] for g in GROUPS if available[g] < MIN_PER_GROUP[g]}
    if short:
        return stop(f"group(s) below the required minimum {MIN_PER_GROUP}: {short} "
                    f"(threshold {threshold}); nothing was written")
    selection_note = ""
    if available["FN"] < N_PER_GROUP:
        selection_note = (f"FN が Test set に {available['FN']} 例しか存在しなかったため、"
                          f"全 {available['FN']} 例を採用し、合計 {len(cases)} 例とした。"
                          f"16 例に合わせるために他群から追加症例は選んでいない。")
        warnings.append(selection_note)
    progress.append(f"{len(cases)} cases selected (available {available})")

    index = load_index(project)          # verifies the fixed cohort/split numbers
    test_ids = set(index[index.split == "test"].PatientID)
    if not set(cases.subject_id) <= test_ids:
        return stop("selected cases are not all in the fixed Test split")
    meta_cols = ["PatientID", "delta_days_from_T0", "StudyInstanceUID", "SeriesInstanceUID",
                 "SOPInstanceUID", "source_filepath"]
    cases = cases.merge(index[meta_cols], left_on="subject_id", right_on="PatientID", how="left")
    cases["split"] = "test"
    cases["threshold_used"] = float(threshold)
    missing_images = [p for p in cases.subject_id if not (image_dir / f"{p}.png").exists()]
    if missing_images:
        return stop(f"cached image missing for {missing_images}")
    progress.append("images located for all 16 cases")

    if out_dir.exists() and any(out_dir.iterdir()) and not args.allow_overwrite:
        return stop(f"{out_dir} already exists and is not empty; re-run with --allow-overwrite "
                    f"only if replacing the deliverable is intended")

    # ---------- produce everything in a staging directory ----------------------------
    stage = Path(tempfile.mkdtemp(prefix="gradcam_stage_"))
    try:
        (stage / "individual").mkdir()
        (stage / "panels").mkdir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model(ModelConfig(pretrained=False)).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        dataset = TaskACXRDataset(index, image_dir, "test")
        row_of = {pid: i for i, pid in enumerate(dataset.patient_ids())}
        progress.append(f"model loaded on {device.type}")

        records, tiles, per_group_tiles = [], [], {g: [] for g in GROUPS}
        with GradCAM(model, gradcam_target_layer(model)) as cam_fn:
            for _, case in cases.iterrows():
                sample = dataset[row_of[case.subject_id]]
                x = sample["image"].unsqueeze(0).to(device)
                cam, prob_model = cam_fn(x)
                if abs(prob_model - float(case.prob)) > 5e-3:
                    warnings.append(f"{case.subject_id}: probability recomputed here "
                                    f"({prob_model:.4f}) differs from the stored Test prediction "
                                    f"({float(case.prob):.4f})")
                raw = np.array(Image.open(image_dir / f"{case.subject_id}.png")).astype(np.float32) / 65535
                gray = np.array(Image.fromarray((raw * 255).astype(np.uint8)).resize(
                    cam.shape[::-1], Image.BILINEAR)).astype(np.float32) / 255
                metrics = region_metrics(cam, gray)

                base = Image.fromarray((gray * 255).astype(np.uint8)).convert("RGB")
                panel = side_by_side(base, overlay(gray, cam))
                caption = [f"{case.subject_id}  {case.outcome_group}  rank {int(case.selection_rank_within_group)}",
                           f"p={float(case.prob):.3f}  thr={float(threshold):.3f}  "
                           f"true={int(case.true_label)}  pred={int(case.predicted_label)}",
                           f"CAM: padding {metrics['frac_in_padding']:.2f}  "
                           f"border {metrics['frac_in_border_band']:.2f}  "
                           f"central {metrics['frac_in_central_50']:.2f}"]
                img = annotate(panel, caption)
                out_name = f"{case.outcome_group}_{int(case.selection_rank_within_group)}_{case.subject_id}.png"
                img.save(stage / "individual" / out_name)
                tiles.append(img)
                per_group_tiles[case.outcome_group].append(img)
                records.append({
                    "subject_id": case.subject_id, "split": "test",
                    "true_label": int(case.true_label), "predicted_probability": float(case.prob),
                    "threshold_used": float(threshold), "predicted_label": int(case.predicted_label),
                    "outcome_group": case.outcome_group,
                    "selection_rank_within_group": int(case.selection_rank_within_group),
                    "delta_days_from_T0": case.get("delta_days_from_T0"),
                    "StudyInstanceUID": case.get("StudyInstanceUID"),
                    "SeriesInstanceUID": case.get("SeriesInstanceUID"),
                    "SOPInstanceUID": case.get("SOPInstanceUID"),
                    "source_image_path": str(image_dir / f"{case.subject_id}.png"),
                    "source_dicom_path": case.get("source_filepath"),
                    "output_image_path": f"individual/{out_name}",
                    "probability_recomputed_here": round(prob_model, 6),
                    "n_available_in_group": available[case.outcome_group],
                    "selection_note": selection_note, **{
                        k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}})
        progress.append("Grad-CAM computed for 16 cases")

        # the 4x4 panel keeps its 16 cells: a missing case leaves an empty cell, it is never
        # replaced by a case from another group
        grid(tiles, cols=4, cells=PANEL_CELLS).save(stage / "panels" / "gradcam_panel_4x4.png")
        for g in GROUPS:
            grid(per_group_tiles[g], cols=4, cells=N_PER_GROUP).save(
                stage / "panels" / f"gradcam_panel_{g.lower()}.png")

        rec = pd.DataFrame(records)
        rec.to_csv(stage / "gradcam_selection.csv", index=False, encoding="utf-8-sig")

        # notes: objective per-case observations; the qualitative reading is left to the researcher
        lines = ["# Grad-CAM 所見メモ（Task A primary model）", "",
                 f"- モデル：{model_name}（checkpoint `{checkpoint.name}`, sha256 `{ckpt_hash[:16]}…`）",
                 f"- 閾値：{threshold:.6f}（final_selection.json の Youden。Test では再計算していない）",
                 "- 対象層：ResNet18 `layer4[-1]`",
                 "- 指標の定義：`padding` は正方形化のためのゼロ埋め領域、`border` は写野の外周 10% の帯"
                 "（コリメーション縁・L/R マーカー・PORTABLE の文字が入りやすい位置）、"
                 "`central` は写野中央 50%×50%（正面胸部では肺野・縦隔にあたる）に落ちた CAM 質量の割合。", ""]
        lines += [f"- 症例数：{len(rec)} 例（TP {available['TP']} 例中 "
                  f"{int((rec.outcome_group == 'TP').sum())}、FP {available['FP']} 例中 "
                  f"{int((rec.outcome_group == 'FP').sum())}、TN {available['TN']} 例中 "
                  f"{int((rec.outcome_group == 'TN').sum())}、FN {available['FN']} 例中 "
                  f"{int((rec.outcome_group == 'FN').sum())}）"]
        if selection_note:
            lines += ["", f"> **{selection_note}** 4×4 パネルは 1 セルを空欄のままとしている。"]
        lines += ["", "## 症例ごとの所見", ""]
        for g in GROUPS:
            lines.append(f"### {g}（{int((rec.outcome_group == g).sum())} 例 / Test 内 {available[g]} 例）")
            if g == "FN" and selection_note:
                lines.append(f"{selection_note}")
            for r in rec[rec.outcome_group == g].itertuples():
                flag = []
                if r.frac_in_padding > 0.05:
                    flag.append("**ゼロ埋め領域への注目が 5% 超**")
                if r.frac_in_border_band > 0.35:
                    flag.append("**外周帯への注目が大きい**")
                if r.frac_in_central_50 < 0.30:
                    flag.append("中央部への注目が乏しい")
                lines.append(
                    f"- `{r.subject_id}`（rank {r.selection_rank_within_group}, p={r.predicted_probability:.3f}, "
                    f"true={r.true_label}, pred={r.predicted_label}, T0 との差 {r.delta_days_from_T0} 日）："
                    f"CAM 質量は padding {r.frac_in_padding:.2f} / border {r.frac_in_border_band:.2f} / "
                    f"central {r.frac_in_central_50:.2f}、重心 (x={r.centroid_x:.2f}, y={r.centroid_y:.2f})、"
                    f"ピーク (x={r.peak_x:.2f}, y={r.peak_y:.2f})。"
                    + ("　" + "、".join(flag) if flag else ""))
            lines.append("")
        lines += [
            "## 総括（定量）", "",
            f"- 16 例平均：padding {rec.frac_in_padding.mean():.3f}、border {rec.frac_in_border_band.mean():.3f}、"
            f"central {rec.frac_in_central_50.mean():.3f}",
            f"- ゼロ埋め領域への注目が 5% を超えた症例：{int((rec.frac_in_padding > 0.05).sum())} / 16",
            f"- 外周帯への注目が 35% を超えた症例：{int((rec.frac_in_border_band > 0.35).sum())} / 16",
            f"- 群別の central 平均：" + "、".join(
                f"{g} {rec[rec.outcome_group == g].frac_in_central_50.mean():.3f}" for g in GROUPS),
            "", "## 総括（読影）", "",
            "上の数値は「どこに CAM 質量が落ちたか」を機械的に測ったもので、解剖学的な妥当性の判断ではない。",
            "肺野・心陰影・下肺野・末梢陰影のどこを見ているか、焼き込み文字やマーカーに引っ張られていないかは、",
            "`panels/gradcam_panel_4x4.png` と `individual/` の画像を確認したうえで研究責任者が記入する。", ""]
        (stage / "gradcam_notes.md").write_text("\n".join(lines), encoding="utf-8")

        meta = {
            "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": Path(__file__).name, "script_sha256": sha256(Path(__file__)),
            "model": model_name, "condition": EXPECTED_CONDITION, "seed": EXPECTED_SEED,
            "checkpoint": str(checkpoint), "checkpoint_sha256": ckpt_hash,
            "final_selection_json": str(frozen_path), "final_selection_sha256": sha256(frozen_path),
            "threshold_used": float(threshold), "threshold_source": "final_selection.json (Youden)",
            "threshold_recomputed_on_test": False,
            "test_predictions": str(pred_path), "test_predictions_sha256": sha256(pred_path),
            "image_dir": str(image_dir),
            "image_sha256": {r.subject_id: sha256(Path(r.source_image_path)) for r in rec.itertuples()},
            "split_manifest_sha256": sha256(project / "data/splits/COVID19_固定患者split_1277.csv"),
            "index_manifest_sha256": sha256(
                project / "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv"),
            "gradcam_layer": "resnet18.layer4[-1]",
            "gradcam_method": "Selvaraju et al. 2017 (gradient-weighted class activation mapping)",
            "selection_rule": {"TP": "highest probability, 4", "FP": "highest probability, 4",
                               "TN": "lowest probability, 4",
                               "FN": "lowest probability, min(4, available)",
                               "tie_break": "Subject ID ascending",
                               "no_substitution": "no group is enlarged to reach 16 cases",
                               "minimum_required_per_group": MIN_PER_GROUP,
                               "reference": "decision_log D-045, amended D-061"},
            "cases_available_per_group": available,
            "cases_used_per_group": {g: int((rec.outcome_group == g).sum()) for g in GROUPS},
            "selection_note": selection_note,
            "panel_cells": PANEL_CELLS, "panel_empty_cells": PANEL_CELLS - len(rec),
            "n_cases": len(rec), "device": device.type,
            "torch": torch.__version__, "python": sys.version.split()[0],
            "warnings": warnings,
            "outputs": sorted(str(p.relative_to(stage)).replace("\\", "/")
                              for p in stage.rglob("*") if p.is_file()),
        }
        (stage / "gradcam_run_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        out_dir.mkdir(parents=True, exist_ok=True)
        for item in stage.iterdir():
            target = out_dir / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), str(target))
        progress.append(f"outputs moved to {out_dir}")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        print("FAILED: nothing was written to the output directory")
        print("progress: " + " -> ".join(progress))
        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print(json.dumps({"model": model_name, "threshold": float(threshold),
                      "cases": {g: list(rec[rec.outcome_group == g].subject_id) for g in GROUPS},
                      "prob_range": [float(rec.predicted_probability.min()),
                                     float(rec.predicted_probability.max())],
                      "mean_frac_central_50": float(rec.frac_in_central_50.mean()),
                      "mean_frac_padding": float(rec.frac_in_padding.mean()),
                      "warnings": warnings, "out_dir": str(out_dir)},
                     ensure_ascii=False, indent=2))
    print("\nfiles: " + ", ".join(meta["outputs"][:6]) + f" … ({len(meta['outputs'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
