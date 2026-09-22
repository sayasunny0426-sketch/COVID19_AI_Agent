# results/taskA

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
