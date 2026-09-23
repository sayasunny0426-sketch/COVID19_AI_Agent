# README の数値の出典

README に載せた数値はすべて実ファイルから読み出したものです。
本ファイルは `scripts/20_prepare_github_repo.py` が生成します。

## リポジトリ内のファイルから読んだ値

| README の記載 | 値 | 出典ファイル |
|---|---|---|
| コホート / split / 死亡 | 1277 / 1021-128-128 / 135-17-17 | `results/taskA/preprocessing/fixed_split_1277.csv` |
| split の SHA256 | `626061a532ab7fa05eeb0198019222e8919d04e5e4df720230538149389599f8` | 同上 |
| Δ日分布 | {-1: 1055, 0: 215, -2: 7} | `results/taskA/preprocessing/index_cxr_manifest_1277.csv` |
| 前処理枚数 / round-trip QC | 1277 / True | `results/taskA/preprocessing/preprocess_summary.json` |
| 最良条件 / 3 seed 平均 Val AUROC | lr3e-4_aug_b / 0.872637 | `results/taskA/training/condition_selection_decision.json` |
| Stage 2 の改善 / margin | -0.002826 / 0.005 | 同上 |

## Google Drive の正本から読んだ値

| README の記載 | 値 | 出典ファイル |
|---|---|---|
| Test ROC-AUC (95% CI) | 0.834658 (0.734473–0.923172) | `results/taskA/evaluation/test_metrics_primary.json` |
| Test PR-AUC (95% CI) | 0.457562 (0.309488–0.712620) | 同上 |
| Brier / ECE | 0.105808 / 0.051147 | 同上 |
| 主操作点の閾値 | 0.008507705 | 同上（Validation 由来・凍結） |
| 副次操作点の閾値 | 0.016904498 | 同上 |
| 副次解析（seed 別・ensemble・T0 当日除外） | — | `results/taskA/evaluation/test_metrics_secondary.json`, `test_metrics_table.csv` |
| current ROC-AUC | 0.907260 | `results/taskA/comparison/taskA_current_vs_aiagent_metrics.csv` |
| ROC-AUC 差 / SE / 95% CI | 0.072602 / 0.033459 / 0.007022–0.138182 | `results/taskA/comparison/taskA_current_vs_aiagent_delong.json` |
| paired DeLong p (two-sided, unadjusted) | 0.030017 | 同上 |
| Grad-CAM 症例構成 | TP 4・FP 4・TN 4・FN 3 = 15 | `results/taskA/gradcam/gradcam_run_meta.json` |
| Grad-CAM の閾値 | 0.008507705 (recomputed_on_test = False) | 同上 |

## 照合できた整合性

- Test 予測の AI-agent 側 ROC-AUC は、`test_metrics_primary.json` (0.834658) と DeLong 側 (0.834658) で一致
- Grad-CAM の群別該当数（TP 14・FP 34・TN 77・FN 3）は、主操作点の混同行列 TP/FP/TN/FN = 14/34/77/3 と一致
- Grad-CAM が参照した Test 予測・凍結 selection の SHA256 は `test_run_meta.json` の値と一致（fd93d6fe83c0a6a0…）

## 収録・除外の対応

| Drive 上の場所 | 本リポジトリ | 収録 | 内容・除外理由 | bytes | sha256（正本） | identical |
|---|---|---|---|---|---|---|
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_predictions_primary.csv` | `results/taskA/evaluation/test_predictions_primary.csv` | 収録 | 患者単位の主解析予測（Subject ID・true_label・確率・閾値・判定） | 7683 | `76dae650320f9e20…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_predictions_all_variants.csv` | `results/taskA/evaluation/test_predictions_all_variants.csv` | 収録 | seed42/43/44・ensemble・primary を 1 ファイルに統合 | 9296 | `3a634791e596abf2…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_metrics_primary.json` | `results/taskA/evaluation/test_metrics_primary.json` | 収録 | 主解析の指標・CI・2 つの操作点・calibration bins | 1873 | `0cefce9420bafe0c…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_metrics_secondary.json` | `results/taskA/evaluation/test_metrics_secondary.json` | 収録 | 副次解析（seed 別・3 seed ensemble・T0 当日除外の感度分析） | 10422 | `579391eaf17911c7…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_metrics_table.csv` | `results/taskA/evaluation/test_metrics_table.csv` | 収録 | 上記を 1 表にまとめた比較表 | 1722 | `14fb82344e5ed9c4…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_roc_curve_points.csv` | `results/taskA/evaluation/test_roc_curve_points.csv` | 収録 | ROC 曲線の座標（図の再描画用） | 16640 | `bbe22477aa04a404…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_pr_curve_points.csv` | `results/taskA/evaluation/test_pr_curve_points.csv` | 収録 | PR 曲線の座標（図の再描画用） | 5779 | `6c63d102e083468b…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_calibration_bins.csv` | `results/taskA/evaluation/test_calibration_bins.csv` | 収録 | calibration 5 分位の集計 | 242 | `ed60d34aec6e0932…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_run_meta.json` | `results/taskA/evaluation/test_run_meta.json` | 収録 | 実行環境・凍結 selection の SHA256・使用した閾値 | 1044 | `4111736ae215c4b6…` | 加工のため不一致（伏字化／列削除） |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_access_log.jsonl` | `results/taskA/evaluation/test_access_log.jsonl` | 収録 | Test set を 1 回だけ使用したことの記録（438 B） | 438 | `d2afbf57b3622f6e…` | 加工のため不一致（伏字化／列削除） |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_predictions_seed42.csv` | — | 除外 | test_predictions_all_variants.csv に同一値が含まれる（照合済み・冗長） | 2913 | `9b015d8d61362cc1…` | — |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_predictions_seed43.csv` | — | 除外 | 同上 | 2848 | `890425412606f47f…` | — |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_predictions_seed44.csv` | — | 除外 | 同上 | 2948 | `f8b68262a54fe573…` | — |
| `01_TaskA_CXR/05_Evaluation/AIagent_taskA_test/test_predictions_ensemble.csv` | — | 除外 | 同上 | 2856 | `20a08c0a5f7a55bd…` | — |
| `01_TaskA_CXR/05_Evaluation/AIagent_vs_current_TaskA_comparison.zip::taskA_current_vs_aiagent_metrics.csv` | `results/taskA/comparison/taskA_current_vs_aiagent_metrics.csv` | 収録 | paired DeLong 比較の正本（ZIP から展開） | 182 | `9e9a17bdb2db160c…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_vs_current_TaskA_comparison.zip::taskA_current_vs_aiagent_predictions.csv` | `results/taskA/comparison/taskA_current_vs_aiagent_predictions.csv` | 収録 | paired DeLong 比較の正本（ZIP から展開） | 4330 | `ba8961fd339afc17…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_vs_current_TaskA_comparison.zip::taskA_current_vs_aiagent_delong.json` | `results/taskA/comparison/taskA_current_vs_aiagent_delong.json` | 収録 | paired DeLong 比較の正本（ZIP から展開） | 983 | `5c73a23661cf17c7…` | ✓ 正本と同一 |
| `01_TaskA_CXR/05_Evaluation/AIagent_vs_current_TaskA_comparison.zip::taskA_current_vs_aiagent_comparison.md` | `results/taskA/comparison/taskA_current_vs_aiagent_comparison.md` | 収録 | paired DeLong 比較の正本（ZIP から展開） | 1263 | `30b0b11800d9d57f…` | ✓ 正本と同一 |
| `01_TaskA_CXR/06_GradCAM/AIagent_taskA_primary_test/gradcam_selection.csv` | `results/taskA/gradcam/gradcam_selection.csv` | 収録 | 15 症例の選択根拠と CAM 領域指標（除去列: source_image_path, source_dicom_path, output_image_path, StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID） | 11465 | `94ac1882ae7d8099…` | 加工のため不一致（伏字化／列削除） |
| `01_TaskA_CXR/06_GradCAM/AIagent_taskA_primary_test/gradcam_notes.md` | `results/taskA/gradcam/gradcam_notes.md` | 収録 | 症例別所見と定量サマリー | 7670 | `877ad9cceeea067c…` | ✓ 正本と同一 |
| `01_TaskA_CXR/06_GradCAM/AIagent_taskA_primary_test/gradcam_run_meta.json` | `results/taskA/gradcam/gradcam_run_meta.json` | 収録 | モデル・閾値・入力 SHA256・選択規則 | 4753 | `f4544e9801ce6ff6…` | 加工のため不一致（伏字化／列削除） |
| `01_TaskA_CXR/06_GradCAM/AIagent_taskA_primary_test/panels/gradcam_panel_4x4.png` | `results/taskA/gradcam/gradcam_panel_4x4.png` | 収録 | 15 症例すべてを 1 枚に収めた代表図（1 セル空欄） | 1206376 | `f37f3be4c7ac6320…` | ✓ 正本と同一 |
| `01_TaskA_CXR/06_GradCAM/AIagent_taskA_primary_test/(individual|panels)/*.png` | — | 除外 | 個別 19 枚は 4x4 パネルと同一症例の高解像度版（計 2.3 MB）。SHA256 のみ収録 |  | `—` | — |

<!-- TASKBC_PROVENANCE_BEGIN (scripts/44_repo_documentation.py) -->

---

# Task A / B / C 主要数値の authoritative artifact 対応



以下は `scripts/44_repo_documentation.py` が生成します（上の Task A 部分は `scripts/20_prepare_github_repo.py` が生成）。**表の値はすべて実行時に artifact から読み出したもので、記憶や推測によるものはありません。**



「読み出した場所」列は、ファイル内のどのキー・どの行から取ったかを示します。

### Task A

凍結モデルの仕様は `test_run_env_note.json` に集約してあります（`test_run_meta.json` は凍結 artifact のため変更していません）。

| 数値 | 値 | authoritative artifact | 読み出した場所 |
|---|---|---|---|
| 凍結条件 / seed / best epoch | lr3e-4_aug_b / 42 / 9 | `results/taskA/evaluation/test_run_env_note.json` | `frozen_model.condition, random_seeds.model_seed_primary, frozen_model.best_epoch` |
| checkpoint SHA256 | 26d6ed95844e5ff6… | `results/taskA/evaluation/test_run_env_note.json` | `frozen_model.checkpoint_sha256` |
| Test ROC-AUC (95% CI) | 0.834658 (0.734473–0.923172) | `results/taskA/evaluation/test_metrics_primary.json` | `auroc.point / ci_low / ci_high` |
| Test PR-AUC | 0.457562 | `results/taskA/evaluation/test_metrics_primary.json` | `auprc.point` |
| Test Brier | 0.105808 | `results/taskA/evaluation/test_metrics_primary.json` | `brier` |
| Test ECE（5 equal-count bin） | 0.051147 | `results/taskA/evaluation/test_metrics_primary.json` | `ece` |
| Test ECE（10 equal-width bin） | 0.097227 | `results/taskC/evaluation/test_metrics_table.csv` | `model=cxr の ece_10_equal_width 列` |
| 操作点の閾値 | 0.008507705 | `results/taskA/evaluation/test_metrics_primary.json` | `operating_point_youden.threshold` |
| bootstrap 設定 | 2,000 resample / stratified / seed 12345 | `results/taskA/evaluation/test_metrics_primary.json` | `auroc.n_boot / stratified / seed` |
| Grad-CAM 症例構成 | TP 4 / FP 4 / TN 4 / FN 3 = 15 | `results/taskA/gradcam/gradcam_run_meta.json` | `cases_used_per_group, n_cases` |

### Task B

**ECE の定義に注意**：Task B の公式表は 5 equal-count bin です。同じモデルを 10 equal-width bin で評価した値は Task C の表にあります。

| 数値 | 値 | authoritative artifact | 読み出した場所 |
|---|---|---|---|
| 最終変数 / 係数数 / EPV | 10 / 12 / 11.25 | `results/taskB/variable_selection/final_variables.json` | `primary.variables / n_features / events_per_parameter` |
| 変数選択規則 | backward AIC subject to <= 13 coefficients | `results/taskB/variable_selection/final_variables.json` | `primary.rule` |
| LR 条件 | C=1.0_cw=balanced | `results/taskB/modeling/final_selection_taskB.json` | `models.lr.condition` |
| LR Test ROC-AUC (95% CI) | 0.946476 (0.895601–0.984632) | `results/taskB/evaluation/test_metrics_table.csv` | `model=lr の auroc / auroc_ci_low / auroc_ci_high` |
| LR Test PR-AUC / Brier / ECE(5 equal-count) | 0.770898 / 0.097765 / 0.161595 | `results/taskB/evaluation/test_metrics_table.csv` | `model=lr の auprc / brier / ece_5bin` |
| XGBoost Test ROC-AUC (95% CI) | 0.928988 (0.873847–0.970866) | `results/taskB/evaluation/test_metrics_table.csv` | `model=xgb の auroc / auroc_ci_low / auroc_ci_high` |
| XGBoost Test PR-AUC / Brier / ECE(5 equal-count) | 0.657762 / 0.073210 / 0.022064 | `results/taskB/evaluation/test_metrics_table.csv` | `model=xgb の auprc / brier / ece_5bin` |
| MLP Test ROC-AUC (95% CI) | 0.924218 (0.859565–0.977213) | `results/taskB/evaluation/test_metrics_table.csv` | `model=mlp の auroc / auroc_ci_low / auroc_ci_high` |
| MLP Test PR-AUC / Brier / ECE(5 equal-count) | 0.747650 / 0.160305 / 0.296720 | `results/taskB/evaluation/test_metrics_table.csv` | `model=mlp の auprc / brier / ece_5bin` |
| XGBoost native missing（secondary） Test ROC-AUC (95% CI) | 0.924218 (0.857446–0.973516) | `results/taskB/evaluation/test_metrics_table.csv` | `model=xgb_native の auroc / auroc_ci_low / auroc_ci_high` |
| XGBoost native missing（secondary） Test PR-AUC / Brier / ECE(5 equal-count) | 0.682196 / 0.069241 / 0.027144 | `results/taskB/evaluation/test_metrics_table.csv` | `model=xgb_native の auprc / brier / ece_5bin` |
| 実行環境（package versions） | numpy 2.5.1 / pandas 3.0.3 / sklearn 1.9.1 / torch 2.14.0+cpu / xgboost 3.4.1 | `results/taskB/evaluation/test_run_meta.json` | `packages` |

### Task C

閾値は Validation 由来で、Test では再計算していません。

| 数値 | 値 | authoritative artifact | 読み出した場所 |
|---|---|---|---|
| 融合式 | p_fused = w * p_clinical + (1 - w) * p_cxr | `results/taskC/modeling/final_selection_taskC.json` | `fusion.formula` |
| 重み（clinical / CXR） | 0.75 / 0.25 | `results/taskC/modeling/final_selection_taskC.json` | `fusion.w_clinical / fusion.w_cxr` |
| frozen spec SHA256 | 2abebc28d4e4e87e… | `results/taskC/modeling/final_selection_taskC.json` | `（ファイル全体のハッシュ）` |
| Validation ROC-AUC（fused） | 0.920509 | `results/taskC/validation/validation_metrics_taskC.json` | `fused.auroc` |
| Test ROC-AUC (95% CI) | 0.944886 (0.892422–0.983042) | `results/taskC/evaluation/test_metrics_taskC.json` | `auroc.point / ci_low / ci_high` |
| Test PR-AUC | 0.780516 | `results/taskC/evaluation/test_metrics_taskC.json` | `auprc.point` |
| Test Brier | 0.078034 | `results/taskC/evaluation/test_metrics_taskC.json` | `brier` |
| Test ECE（10 equal_width bin） | 0.128711 | `results/taskC/evaluation/test_metrics_taskC.json` | `ece / calibration_definition` |
| 閾値（Validation 由来・Test 再計算なし） | 0.32592158196659415（recomputed_on_test = False） | `results/taskC/evaluation/test_metrics_taskC.json` | `threshold.value / recomputed_on_test` |
| 感度 / 特異度 | 0.882353 / 0.810811 | `results/taskC/evaluation/test_metrics_taskC.json` | `operating_point.sensitivity / specificity` |
| TP / FP / TN / FN | 15 / 21 / 90 / 2 | `results/taskC/evaluation/test_metrics_taskC.json` | `operating_point.tp / fp / tn / fn` |
| modality importance（clinical / CXR、post hoc） | 0.350539 / 0.003551 | `results/taskC/evaluation/modality_importance_test.csv` | `mean_auroc_drop` |
| secondary（logit空間 / stacking、探索的） | 0.932697 / 0.935877 | `results/taskC/evaluation/secondary_exploratory_test.json` | `models.*.auroc` |
| 実行環境（package versions） | numpy 2.5.1 / pandas 3.0.3 / sklearn 1.9.1 / torch 2.14.0+cpu / xgboost 3.4.1 | `results/taskC/evaluation/test_run_meta.json` | `packages` |

### 横断比較・統計

paired DeLong は 2 family 構成で、family ごとに Holm 補正を行っています。`Late Fusion vs Clinical LR` は両 family に属するため、unadjusted p は 1 つ、Holm 調整済み p は 2 つあります。

| 数値 | 値 | authoritative artifact | 読み出した場所 |
|---|---|---|---|
| 全5モデル Test 指標（単一定義） | ECE = 10 equal-width bin、n=128 / deaths=17 | `results/taskC/evaluation/test_metrics_table.csv` | `全行` |
| 患者単位 全モデル予測 | 128 行 × 7 列 | `results/comparison/test_predictions_all_models.csv` | `subject_id, true_label, prob_cxr, prob_clinical_lr, prob_clinical_xgboost, prob_clinical_mlp, prob_late_fusion` |
| DeLong Family A: Late Fusion vs Clinical LR | ΔAUC -0.001590 / p(unadj) 0.769705 / Holm 0.769705 | `results/comparison/delong_results.csv` | `family=A, model_2=clinical_lr` |
| DeLong Family A: Late Fusion vs Clinical XGBoost | ΔAUC +0.015898 / p(unadj) 0.227127 / Holm 0.539785 | `results/comparison/delong_results.csv` | `family=A, model_2=clinical_xgboost` |
| DeLong Family A: Late Fusion vs Clinical MLP | ΔAUC +0.020668 / p(unadj) 0.179928 / Holm 0.539785 | `results/comparison/delong_results.csv` | `family=A, model_2=clinical_mlp` |
| DeLong Family B: Late Fusion vs Clinical LR | ΔAUC -0.001590 / p(unadj) 0.769705 / Holm 0.769705 | `results/comparison/delong_results.csv` | `family=B, model_2=clinical_lr` |
| DeLong Family B: Late Fusion vs CXR ResNet18 | ΔAUC +0.110228 / p(unadj) 0.002079 / Holm 0.004158 | `results/comparison/delong_results.csv` | `family=B, model_2=cxr` |
| DeLong の事前規定（2 family + Holm） | Test を開く前に固定 | `results/comparison/delong_plan.json` | `（全体）` |

### 正本が競合しないことの確認



| 確認項目 | 結果 |

|---|---|

| comparison artifact が official Test metrics を上書きしていないか | 上書きなし。`results/task{B,C}/comparison/` は `pipeline` 列と `source` 列を持ち、AI-Agent 行は official artifact の再掲であることを明示している |

| harmonized 値が official 値を置き換えていないか | 置き換えていない。`harmonized_ece.csv` / `harmonized_ci.csv` は比較専用で、official 表は未変更 |

| 同一数値が複数ファイルにある場合の正本 | `docs/artifact_index.md` 末尾の「Which file is the source of truth for a given number?」表に明記 |



Task A / Task B / Task C の official artifact は、本ドキュメントの作成により一切変更していません。

<!-- TASKBC_PROVENANCE_END -->
