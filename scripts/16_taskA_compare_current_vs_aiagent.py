"""Exploratory paired comparison of the current Task A CXR model and the AI-agent Task A model.

This compares two sets of Test predictions that already exist. It never retrains, never
re-evaluates the Test set with a model, and never changes a model, threshold or
hyper-parameter. It only reads the two prediction files and computes paired statistics.

Status of the comparison: **post-hoc / exploratory**. It was not part of the pre-registered
hypothesis tests of the study, so the p-value is unadjusted and is reported as descriptive
evidence, not as a confirmatory result.

DeLong implementation: src/covid_mortality/evaluation/delong.py -- DeLong, DeLong &
Clarke-Pearson (1988) with the fast O(N log N) formulation of Sun & Xu (2014), midranks for
ties, normal approximation for the two-sided p-value and the Wald CI of the difference.
Bootstrap CIs for the single-model metrics use a stratified patient-level resample with a
fixed seed (see BOOTSTRAP_SEED) so the numbers are reproducible.

Outputs (written into --out-dir, never overwritten unless --allow-overwrite):
    taskA_current_vs_aiagent_predictions.csv
    taskA_current_vs_aiagent_metrics.csv
    taskA_current_vs_aiagent_delong.json
    taskA_current_vs_aiagent_comparison.md

Usage:
    python scripts/16_taskA_compare_current_vs_aiagent.py \
        --current "<...>/TaskA_CXR_Test患者別予測結果.csv" \
        --aiagent "<...>/AIagent_taskA_test/test_predictions_primary.csv" \
        --out-dir "<...>/AIagent_vs_current_TaskA_comparison"
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.evaluation.delong import delong_roc_test  # noqa: E402
from covid_mortality.evaluation.metrics import (  # noqa: E402
    average_precision, bootstrap_ci, brier_score, roc_auc)

N_PATIENTS, N_EVENTS = 128, 17
BOOTSTRAP_SEED, N_BOOT = 20260921, 2000
ID_CANDIDATES = ["Subject ID", "subject_id", "SubjectID", "PatientID", "patient_id",
                 "to_patient_id", "ID"]
PROB_CANDIDATES = ["prob", "probability", "pred_prob", "predicted_probability", "y_prob",
                   "prob_primary", "y_pred_prob", "pred", "score", "pred_proba", "proba"]
LABEL_CANDIDATES = ["true_label", "label", "y_true", "outcome", "died", "mortality",
                    "last.status", "status", "death", "event"]
# label values are normalised to 0/1 with these maps (case-insensitive)
POSITIVE_VALUES = {"1", "1.0", "deceased", "died", "death", "true", "yes", "positive"}
NEGATIVE_VALUES = {"0", "0.0", "discharged", "alive", "survived", "false", "no", "negative"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick(columns, candidates, what: str, path: Path) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(f"{path.name}: no {what} column found; looked for {candidates}, "
                   f"file has {list(columns)}")


def load_predictions(path: Path, tag: str, id_col: str | None = None,
                     prob_col: str | None = None, label_col: str | None = None) -> pd.DataFrame:
    """Read one prediction file. Column names can be given explicitly when the file uses
    names this script does not know; otherwise they are detected from the candidate lists."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    id_col = id_col or pick(df.columns, ID_CANDIDATES, "subject id", path)
    prob_col = prob_col or pick(df.columns, PROB_CANDIDATES, "probability", path)
    label_col = label_col or pick(df.columns, LABEL_CANDIDATES, "true label", path)
    for c in (id_col, prob_col, label_col):
        if c not in df.columns:
            raise KeyError(f"{path.name}: column {c!r} not present; file has {list(df.columns)}")
    out = df[[id_col, label_col, prob_col]].copy()
    out.columns = ["subject_id", f"{tag}_true_label", f"{tag}_prob"]
    out["subject_id"] = out.subject_id.astype(str).str.strip()

    def to01(v):
        s = str(v).strip().lower()
        if s in POSITIVE_VALUES:
            return 1
        if s in NEGATIVE_VALUES:
            return 0
        return np.nan

    out[f"{tag}_true_label"] = out[f"{tag}_true_label"].map(to01)
    if out[f"{tag}_true_label"].isna().any():
        bad = sorted({str(v) for v in df[label_col][out[f"{tag}_true_label"].isna()]})[:5]
        raise ValueError(f"{path.name}: could not interpret label values {bad}; "
                         f"pass --{tag}-label-col or extend POSITIVE_VALUES/NEGATIVE_VALUES")
    out[f"{tag}_true_label"] = out[f"{tag}_true_label"].astype(int)
    out[f"{tag}_prob"] = pd.to_numeric(out[f"{tag}_prob"], errors="raise").astype(float)
    if not ((out[f"{tag}_prob"] >= 0) & (out[f"{tag}_prob"] <= 1)).all():
        raise ValueError(f"{path.name}: column {prob_col!r} has values outside [0, 1]; "
                         f"is it a probability?")
    out.attrs["columns_used"] = {"id": id_col, "prob": prob_col, "label": label_col}
    return out


def verify(cur: pd.DataFrame, agent: pd.DataFrame) -> dict:
    checks = {
        "current_n_rows": len(cur), "aiagent_n_rows": len(agent),
        "current_is_128": len(cur) == N_PATIENTS, "aiagent_is_128": len(agent) == N_PATIENTS,
        "current_no_duplicate_ids": bool(cur.subject_id.is_unique),
        "aiagent_no_duplicate_ids": bool(agent.subject_id.is_unique),
        "subject_ids_identical": set(cur.subject_id) == set(agent.subject_id),
        "n_only_in_current": len(set(cur.subject_id) - set(agent.subject_id)),
        "n_only_in_aiagent": len(set(agent.subject_id) - set(cur.subject_id)),
    }
    merged = cur.merge(agent, on="subject_id", how="inner").sort_values("subject_id")
    checks["merged_n"] = len(merged)
    checks["labels_identical"] = bool(
        (merged.current_true_label == merged.aiagent_true_label).all()) if len(merged) else False
    if checks["labels_identical"]:
        y = merged.current_true_label.to_numpy()
        checks["events"] = int(y.sum())
        checks["survivors"] = int((1 - y).sum())
        checks["events_is_17"] = int(y.sum()) == N_EVENTS
        checks["survivors_is_111"] = int((1 - y).sum()) == N_PATIENTS - N_EVENTS
    checks["all_passed"] = all(checks[k] for k in
                               ["current_is_128", "aiagent_is_128", "current_no_duplicate_ids",
                                "aiagent_no_duplicate_ids", "subject_ids_identical",
                                "labels_identical", "events_is_17", "survivors_is_111"]
                               if k in checks) and checks.get("events_is_17", False)
    return checks, merged


def model_metrics(y, p, name: str) -> dict:
    auroc = bootstrap_ci(y, p, roc_auc, N_BOOT, stratified=True, seed=BOOTSTRAP_SEED)
    auprc = bootstrap_ci(y, p, average_precision, N_BOOT, stratified=True, seed=BOOTSTRAP_SEED)
    return {"model": name, "n": int(len(y)), "events": int(y.sum()),
            "roc_auc": auroc["point"], "roc_auc_ci_low": auroc["ci_low"],
            "roc_auc_ci_high": auroc["ci_high"],
            "pr_auc_average_precision": auprc["point"], "pr_auc_ci_low": auprc["ci_low"],
            "pr_auc_ci_high": auprc["ci_high"],
            "brier_score": brier_score(y, p),
            "mean_predicted_probability": float(np.mean(p)),
            "observed_event_rate": float(np.mean(y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="current model Test predictions CSV")
    ap.add_argument("--aiagent", required=True, help="AI-agent model Test predictions CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-overwrite", action="store_true")
    for tag in ("current", "aiagent"):          # explicit column names when detection fails
        ap.add_argument(f"--{tag}-id-col", default=None)
        ap.add_argument(f"--{tag}-prob-col", default=None)
        ap.add_argument(f"--{tag}-label-col", default=None)
    args = ap.parse_args()

    cur_path, agent_path, out_dir = Path(args.current), Path(args.aiagent), Path(args.out_dir)
    for p in (cur_path, agent_path):
        if not p.exists():
            print(f"STOP: input not found: {p}")
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {n: out_dir / n for n in
               ("taskA_current_vs_aiagent_predictions.csv", "taskA_current_vs_aiagent_metrics.csv",
                "taskA_current_vs_aiagent_delong.json", "taskA_current_vs_aiagent_comparison.md")}
    existing = [p.name for p in outputs.values() if p.exists()]
    if existing and not args.allow_overwrite:
        print(f"STOP: outputs already exist ({', '.join(existing)}). "
              f"Re-run with --allow-overwrite only if that is intended.")
        return 1

    cur = load_predictions(cur_path, "current", args.current_id_col, args.current_prob_col,
                           args.current_label_col)
    agent = load_predictions(agent_path, "aiagent", args.aiagent_id_col, args.aiagent_prob_col,
                             args.aiagent_label_col)
    checks, merged = verify(cur, agent)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["all_passed"]:
        print("STOP: patient-level correspondence could not be verified; nothing was written.")
        return 1

    y = merged.current_true_label.to_numpy()
    p_cur = merged.current_prob.to_numpy(dtype=float)
    p_agent = merged.aiagent_prob.to_numpy(dtype=float)

    preds = pd.DataFrame({"Subject ID": merged.subject_id, "true_label": y,
                          "current_model_prob": p_cur, "aiagent_model_prob": p_agent})
    preds.to_csv(outputs["taskA_current_vs_aiagent_predictions.csv"], index=False,
                 encoding="utf-8-sig")

    metrics = pd.DataFrame([model_metrics(y, p_cur, "current_TaskA_CXR"),
                            model_metrics(y, p_agent, "aiagent_TaskA_CXR_lr3e-4_aug_b_seed42")])
    metrics["comparison_type"] = "exploratory/post-hoc"
    metrics["generated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metrics.to_csv(outputs["taskA_current_vs_aiagent_metrics.csv"], index=False,
                   encoding="utf-8-sig")

    d = delong_roc_test(y, p_cur, p_agent)     # difference = current - AI-agent
    delong_payload = {
        "n": d["n"], "events": d["events"], "survivors": int(d["n"] - d["events"]),
        "auroc_current": d["auc_a"], "auroc_aiagent": d["auc_b"],
        "auroc_difference": d["difference"], "difference_definition": "current - AI-agent",
        "se": d["se"], "z": d["z"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
        "ci_level": 0.95, "p_value": d["p_value"], "p_value_type": "two-sided, unadjusted",
        "test": "paired DeLong",
        "implementation": "DeLong et al. 1988; fast algorithm of Sun & Xu 2014 (midranks); "
                          "src/covid_mortality/evaluation/delong.py",
        "comparison_type": "exploratory/post-hoc",
        "pre_registered": False,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_files": {
            "current": {"path": str(cur_path), "filename": cur_path.name, "sha256": sha256(cur_path),
                        "columns_used": cur.attrs["columns_used"]},
            "aiagent": {"path": str(agent_path), "filename": agent_path.name,
                        "sha256": sha256(agent_path), "columns_used": agent.attrs["columns_used"]}},
        "verification": checks,
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOTSTRAP_SEED, "stratified": True},
    }
    outputs["taskA_current_vs_aiagent_delong.json"].write_text(
        json.dumps(delong_payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    m = metrics.set_index("model")
    cur_name, agent_name = m.index[0], m.index[1]
    md = f"""# Task A CXR モデルの比較：current model と AI-agent model

- 作成：{delong_payload['timestamp']}
- **位置づけ：post-hoc / exploratory comparison**。研究計画で事前登録した主要仮説検定ではない。p 値は多重比較未補正
- 再学習・Test の再評価は行っていない。既に得られている Test 予測のみを使用した
- Test を見てモデル・閾値・ハイパーパラメータを変更していない

## 1. 比較の目的
同一の Test 128 例に対して、従来の Task A CXR モデルと、AI agent が独立に設計・学習した
Task A CXR モデルの判別性能を、患者単位で対応させて比較する。

## 2. 比較したモデル

| | current model | AI-agent model |
|---|---|---|
| 予測ファイル | `{cur_path.name}` | `{agent_path.name}` |
| SHA256 | `{delong_payload['input_files']['current']['sha256'][:16]}…` | `{delong_payload['input_files']['aiagent']['sha256'][:16]}…` |
| 学習条件 | 既存 Task A の設定 | lr 3e-4 / augmentation aug_b / seed 42（単一 seed、事前固定） |

## 3. 患者単位の対応の確認

| 確認項目 | 結果 |
|---|---|
| 両ファイルとも 128 例 | {checks['current_n_rows']} / {checks['aiagent_n_rows']} |
| Subject ID が全例一致 | {checks['subject_ids_identical']}（片側のみ: current {checks['n_only_in_current']}、AI-agent {checks['n_only_in_aiagent']}） |
| 重複 Subject ID なし | current {checks['current_no_duplicate_ids']} / AI-agent {checks['aiagent_no_duplicate_ids']} |
| true_label が全例一致 | {checks['labels_identical']} |
| 死亡 17 例・生存 111 例 | 死亡 {checks['events']} / 生存 {checks['survivors']} |

## 4. 判別性能（同一 128 例）

| 指標 | current model | AI-agent model |
|---|---|---|
| ROC-AUC (95% CI) | {m.loc[cur_name, 'roc_auc']:.4f} ({m.loc[cur_name, 'roc_auc_ci_low']:.4f}–{m.loc[cur_name, 'roc_auc_ci_high']:.4f}) | {m.loc[agent_name, 'roc_auc']:.4f} ({m.loc[agent_name, 'roc_auc_ci_low']:.4f}–{m.loc[agent_name, 'roc_auc_ci_high']:.4f}) |
| PR-AUC / Average Precision (95% CI) | {m.loc[cur_name, 'pr_auc_average_precision']:.4f} ({m.loc[cur_name, 'pr_auc_ci_low']:.4f}–{m.loc[cur_name, 'pr_auc_ci_high']:.4f}) | {m.loc[agent_name, 'pr_auc_average_precision']:.4f} ({m.loc[agent_name, 'pr_auc_ci_low']:.4f}–{m.loc[agent_name, 'pr_auc_ci_high']:.4f}) |
| Brier score | {m.loc[cur_name, 'brier_score']:.4f} | {m.loc[agent_name, 'brier_score']:.4f} |

CI は患者単位の層別 bootstrap（{N_BOOT} 反復、seed {BOOTSTRAP_SEED}）。
PR-AUC は Average Precision（step-wise）であり、PR 曲線の台形積分ではない。

## 5. paired DeLong test（同一患者で対応）

| 項目 | 値 |
|---|---|
| current ROC-AUC | {d['auc_a']:.6f} |
| AI-agent ROC-AUC | {d['auc_b']:.6f} |
| 差（current − AI-agent） | {d['difference']:+.6f} |
| 95% CI | {d['ci_low']:+.6f} … {d['ci_high']:+.6f} |
| two-sided unadjusted p 値 | {d['p_value']:.4f} |
| 検定 | paired DeLong（DeLong 1988 / Sun & Xu 2014 の高速実装、同順位は midrank） |

## 6. 解釈上の注意
- 本比較は **exploratory** であり、事前に固定した主要仮説検定ではない。p 値は補正していない
- Test の死亡は 17 例と少なく、**ROC-AUC の CI は広い**。点推定の差をそのまま性能差の大きさとして読み取らない
- 両モデルは学習条件が複数の点で異なるため、この比較からは**どの設計要素が差に寄与したかは特定できない**
- AI-agent model は単一 seed（42）の主解析であり、seed による変動は別に報告している
- 差が認められた場合でも「current model が統計学的に優れている」と断定せず、**同一 Test set 上での exploratory paired comparison の結果**として記述する
- 逆に p 値が大きい場合も「同等である」ことの証明にはならない

## 7. 再現方法
```bash
python scripts/16_taskA_compare_current_vs_aiagent.py \\
    --current "{cur_path}" \\
    --aiagent "{agent_path}" \\
    --out-dir "{out_dir}"
```
乱数を使うのは bootstrap CI のみで、seed {BOOTSTRAP_SEED} に固定している。DeLong 検定は解析的で乱数を使わない。
"""
    outputs["taskA_current_vs_aiagent_comparison.md"].write_text(md, encoding="utf-8")

    print("\n" + json.dumps({k: delong_payload[k] for k in
                             ("auroc_current", "auroc_aiagent", "auroc_difference", "ci_low",
                              "ci_high", "p_value", "test", "comparison_type")},
                            ensure_ascii=False, indent=2))
    print("\nwritten:")
    for p in outputs.values():
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
