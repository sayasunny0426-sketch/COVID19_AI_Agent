# COVID-19 in-hospital mortality prediction from admission chest radiographs (Task A)

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
| 固定コホート | **1277 例** |
| Train / Validation / Test | **1021 / 128 / 128** |
| 死亡数（Train / Val / Test） | **135 / 17 / 17** |
| 1 患者あたりの画像 | index CXR 1 枚 |
| 前処理済み画像 | 1277 枚（16-bit PNG 512×512、round-trip QC 完全一致：True） |

固定 split は `results/taskA/preprocessing/fixed_split_1277.csv`（SHA256 `626061a532ab7fa0…`）。
**split は再作成せず、全 Task で共通に使用します。**

## 3. 予測時点と CXR の選択規則（検証済）

- **T0 = `visit_start_datetime`**（日付のみ。データに正確な admission time が存在しないため）
- CXR は **撮影日が T0−2 日 〜 T0 当日**のものを対象とし、T0 より後の日付は使用しない
- 対象日が複数ある場合は T0 に最も近い日、その日に複数 Series があれば `AcquisitionTime` が
  最も早い Series、同時刻なら `SeriesNumber` 最小（tie-break のみ。撮影順とは解釈しない）
- frontal 判定：`Modality` が CR/DX → `StudyDescription` で胸部検査を特定 → `SeriesDescription` で AP/PA
- 選ばれた index CXR の T0 からの日数差：0 日 215 例、-1 日 1055 例、-2 日 7 例

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
  計 30 run を **Validation のみ**で比較し、3 seed 平均 Validation AUROC が最大の条件を採用（検証済：lr3e-4_aug_b、0.872637）
- Stage 2 は baseline を上回らなかった（最良でも **-0.002826**、事前に固定した
  practical tolerance / parsimony margin 0.005 未満）ため、より単純な Stage 1 条件を採用
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

## 7. Test 性能

主解析 `lr3e-4_aug_b` / seed 42、Test 128 例（死亡 17 例、有病率 0.133）。

| 指標 | 値 | 95% CI |
|---|---|---|
| ROC-AUC | **0.835** | 0.734 – 0.923 |
| PR-AUC（Average Precision） | **0.458** | 0.309 – 0.713 |
| Brier score | 0.106 | — |
| ECE（5 分位） | 0.051 | — |

操作点（いずれも **Validation で決定した閾値を凍結して適用**。Test では再計算していません）

| 操作点 | 閾値 | 感度 | 特異度 | PPV | NPV | TP/FP/TN/FN |
|---|---|---|---|---|---|---|
| 主：Validation Youden 最大 | 0.008508 | 0.824 | 0.694 | 0.292 | 0.963 | 14/34/77/3 |
| 副次（exploratory）：Validation 感度 ≥ 0.80 のうち最も高い閾値 | 0.016904 | 0.706 | 0.766 | 0.316 | 0.944 | 12/26/85/5 |

副次操作点の感度 0.706 は **Test での実測値**です。閾値は Validation で感度 ≥ 0.80 を満たすよう
選んだもので、Test で感度 0.80 を保証するものではありません。

副次解析（主解析を置き換えるものではありません）

| モデル | ROC-AUC | PR-AUC |
|---|---|---|
| seed 43 | 0.836 | 0.584 |
| seed 44 | 0.845 | 0.475 |
| 3 seed ensemble | 0.846 | 0.499 |
| 感度分析：index CXR が T0 当日でない 106 例 | 0.891 | 0.561 |

AUPRC は step-wise の Average Precision であり、PR 曲線の台形積分ではありません。
95% CI は患者単位の層別 bootstrap（2000 反復、seed 12345）です。
出典：`results/taskA/evaluation/`（`test_metrics_primary.json`, `test_metrics_table.csv`）。
実行環境と凍結 selection の SHA256 は `test_run_meta.json`、Test を 1 回だけ使用した記録は
`test_access_log.jsonl` にあります。

## 8. Grad-CAM

- 対象層：`resnet18.layer4[-1]`、主解析モデル（lr3e-4_aug_b/seed42）に対してのみ実施
- 閾値 0.008508（`final_selection.json` の Youden。**Test で再計算していない**：
  `threshold_recomputed_on_test = False`）
- 症例：**TP 4・FP 4・TN 4・FN 3 = 計 15 例**
  （各群の Test 内該当数：TP 14・FP 34・TN 77・FN 3）。
  FN は Test set に 3 例しか存在しないため全 3 例を採用し、**他群からの補充は行っていない**。
  4×4 パネルは 1 セルを空欄
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

## 9. 既存モデルとの探索的比較

同一の Test 128 例（死亡 17 例）。Subject ID 128/128 一致、重複なし、true_label 全一致。

| | current model | AI-agent model |
|---|---|---|
| ROC-AUC | 0.9073 | 0.8347 |
| PR-AUC | 0.6229 | 0.4576 |
| Brier score | 0.1017 | 0.1058 |

- ROC-AUC 差（current - AI-agent）：**0.0726**（SE 0.0335、95% CI 0.0070 – 0.1382）
- paired DeLong、two-sided、**unadjusted** p = **0.0300**

**post-hoc / exploratory comparison** です。事前登録した主要仮説検定ではなく、p 値は多重比較を補正していません。
Test の死亡は 17 例で CI は広く、両モデルは学習条件が複数の点で異なるため、
**どの設計要素が差に寄与したかは特定できません**。「current model が統計学的に優れている」とは結論しません。
出典：`results/taskA/comparison/`（`taskA_current_vs_aiagent_delong.json`, `..._metrics.csv`）。

## 10. リポジトリ構成

```
docs/
notebooks/
results/
  taskA/
scripts/
src/
  covid_mortality/
tests/
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
python scripts/18_taskA_gradcam_primary_test.py --project . --frozen <final_selection.json> \
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
  UID は `scripts/02b` と `scripts/18` が DICOM から再生成します（PUBLISH_UIDS = True で収録に切り替え可）
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

生成日：2026-09-22（`scripts/20_prepare_github_repo.py` により作成）
