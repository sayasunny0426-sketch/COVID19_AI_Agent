# Task A 最終 Test 評価の計画（保存する出力と評価項目）

- 作成日：2026-09-21　状態：**提示のみ。Test 評価は未実行**
- 前提：`final_selection.json` を凍結してから実行する。実行は **1 回のみ**
- 主解析：`lr3e-4_aug_b / seed 42`（D-056）。副次：seed 43 / 44 の個別結果、3 seed ensemble

---

## 1. 実行の前提条件（満たさなければ実行しない）

| # | 条件 |
|---|---|
| 1 | `final_selection.json` が存在する |
| 2 | 記録された checkpoint・config・Validation 予測の SHA256 がすべて一致する |
| 3 | split manifest と index CXR manifest の SHA256 が一致する |
| 4 | Test の患者集合が 128 例・死亡 17 例で、Train / Validation と重複しない |
| 5 | 閾値は `final_selection.json` に記録済みの値を使う（Test では再計算しない） |
| 6 | 既に Test 予測ファイルが存在する場合は上書きしない（明示フラグと記録が必要） |

実行のたびに `test_access_log.jsonl` に、日時・実行者・コード hash・凍結ファイル hash・生成物を追記する。

---

## 2. 保存する出力

### 2.1 患者単位の予測（最重要）
| ファイル | 内容 |
|---|---|
| `test_predictions_primary.csv` | **主解析**。列：`subject_id`, `true_label`, `prob`, `model`(=lr3e-4_aug_b/seed42), `threshold_used`, `predicted_label`, `delta_days_from_T0`（L-1 の感度分析用） |
| `test_predictions_seed42.csv` / `seed43.csv` / `seed44.csv` | 各 seed 単独（副次・再現性確認） |
| `test_predictions_ensemble.csv` | 3 seed 平均（副次） |
| `test_predictions_all_variants.csv` | 上記を `subject_id` で横に結合したもの。Task C での結合と DeLong 検定に使う |

いずれも **1 患者 1 行・128 行**。行の順番ではなく `subject_id` で結合する。

### 2.2 指標
| ファイル | 内容 |
|---|---|
| `test_metrics_primary.json` | 主解析の全指標（§3） |
| `test_metrics_secondary.json` | seed 43 / 44 / ensemble の同じ指標 |
| `test_metrics_table.csv` | 論文用の一覧（モデル × 指標、95% CI 付き） |

### 2.3 曲線・較正の元データ（図そのものではなく数値）
| ファイル | 内容 |
|---|---|
| `roc_curve_points.csv` | FPR / TPR / threshold（モデル別） |
| `pr_curve_points.csv` | Precision / Recall / threshold |
| `calibration_bins.csv` | 予測確率の bin ごとの実測割合・件数 |
| `bootstrap_distribution.csv` | bootstrap 2,000 反復の AUROC・AUPRC の値（CI の再計算と検定に使う） |

### 2.4 図
`roc_curve.png/pdf`、`pr_curve.png/pdf`、`calibration_plot.png/pdf`、`learning_curve_primary.png/pdf`（Loss と Accuracy、課題要件）。図はすべて §2.3 の CSV から再生成できる。

### 2.5 Grad-CAM
| ファイル | 内容 |
|---|---|
| `gradcam/<subject_id>_<category>.png` | TP・FP は予測確率の高い順に 4 例、TN・FN は低い順に 4 例（同値は PatientID 昇順）。計 16 例 |
| `gradcam_selection.csv` | 選ばれた症例、カテゴリ、予測確率、true label、選択順位 |
| `gradcam_notes.md` | 焼き込み文字・マーカー・画像端への注目の有無（近道学習の監視） |

対象層は `layer4[-1]`。**凍結後の 1 回のみ**、主解析モデルで実施する。

### 2.6 実行の記録
`test_run_meta.json`（日時、コード hash、`final_selection.json` の hash、環境、GPU、所要時間）、`test_access_log.jsonl`。

---

## 3. 評価項目

### 3.1 主評価
- **AUROC**（95% CI：患者単位の**層別** bootstrap 2,000 反復、percentile 法。層別は少数イベントで class composition を保ち CI を安定させるため）
- 参考として DeLong 法の CI も併記

### 3.2 副次評価
- **AUPRC**（= Average Precision、step-wise。台形積分は使わない：D-049）と 95% CI
- **Brier score**、**calibration**（intercept・slope、calibration plot、ECE）
- 凍結済み閾値（Youden）での **感度・特異度・PPV・NPV・accuracy**、混同行列
- 副次の操作点（Validation 感度 0.80 の閾値）での同じ指標
- 有病率（Test の死亡 17/128 = 13.3%）を併記し、PPV/NPV の解釈に使う

### 3.3 再現性・副次解析
- seed 42 / 43 / 44 の個別 AUROC・AUPRC と、その範囲
- 3 seed ensemble の AUROC・AUPRC（主解析との差を提示）
- **L-1 の感度分析**：index CXR が T0 当日の患者を除いた部分集合での AUROC（Q-D7。該当数が少なければ数値のみ報告し、解釈は限定する）

### 3.4 提示の方針
- Test の死亡は 17 例であり、**CI は広い**。点推定の差を過度に解釈しない
- 「有意差なし」を「同等」と読み替えない
- モデル間の比較検定（paired DeLong + Holm）は **Task C まで揃ってから**行う。Task A 単独では CI の提示にとどめる

---

## 4. 実行後に行わないこと
- 結果を見てからの閾値・モデル・前処理の変更（行う場合は事後解析として明示し、`change_log.csv` に記録）
- Test 予測を用いた条件選択、Task B / C の設計変更

---

## 5. 未確定事項

| # | 内容 |
|---|---|
| ~~Q-D3~~ | **解決（D-057）**：主解析は Validation の Youden 最大（同率なら最も低い閾値）、Test では再計算しない。副次は「Validation 感度 ≥ 0.80 のうち最も高い閾値」で、exploratory secondary operating point として扱う |
| Q-D6 | calibration・Brier・PR-AUC を全モデルで報告するか（Task B / C との一貫性） |
| Q-D7 | L-1 の感度分析を実施するか |
| 実行環境 | Test 評価も Colab で実行する（GPU 不要だが、環境を揃えるため同じ Notebook 経由を推奨） |
