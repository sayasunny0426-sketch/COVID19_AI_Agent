# 実装計画

- 改訂：v0.8（2026-09-18）
  - D-037 を反映した：患者集合と index CXR の両方が完全に一致することを合格条件とし、1 例でも食い違えば停止して差分表を報告する
- v0.7（2026-09-18）
  - D-036 を反映した：コホートを作り直して既存の 1,277 例と照合し、一致すれば既存の split をそのまま使う。一致しなければ止まって報告する
- v0.6（2026-09-18）
  - D-035 に従って作業の順序を改訂した（CXR 監査 → index CXR → コホート固定 → split 固定 → Train だけで臨床の再監査 → 候補選定 → 補完・変換 → clinical model）
  - v0.5 では Drive のルートを「気合のCOVID19」とした（D-027）
  - frontal の判定方法を反映した（D-028）
  - DICOM を複製しない方針を反映した（D-029）
  - Q-B9(b)(c) の確認手順を反映した（D-030）
  - v0.4 では index CXR の選択ルール（D-025, D-026）を反映した
  - v0.3 では研究者の決定 [R] を反映した：v1 のコホートは使わない。T0 = `visit_start_datetime`。1,277 例の固定コホートと split
  - v0.2 は基準資料 [T] と [FB] との照合を反映した版
- ステータス：**計画段階（未承認）**。コード、データ監査、前処理、学習はいずれも未実施
- 出典の表記
  - **[T]** 課題文
  - **[FB]** 教員フィードバック（最優先）
  - **[U]** 最初の依頼文
  - **[R]** 研究者の決定
  - **[提案]** Claude の提案
  - **[未確定]** 未確定。番号は [open_questions.md](open_questions.md) の番号
- 要件、研究者の決定、limitation は [requirements.md](requirements.md) を参照

---

## 0. 前提

| 項目 | 内容 | 出典 |
|---|---|---|
| 目的 | T0 時点の情報から院内死亡を予測する | [T] [FB §8] |
| T0 | `visit_start_datetime`（正確な admission time はデータにない） | [R] D-020 |
| CXR の対象 | 撮影日（calendar date）が T0−2 日から T0 当日まで。T0 より後の日付の CXR は使わない。T0 当日の CXR は一律に「入院後」とはみなさない | [R] D-021, D-022 |
| 同日内の順序 | `AcquisitionTime` は同じ日の中での順序を決めるためだけに使う | [R] D-023 |
| index CXR の選択 | frontal CXR（T0−2 日〜T0 当日）→ T0 に最も近い撮影日 → その日の中で `AcquisitionTime` が最も早い Series → 同じ時刻なら `SeriesNumber` が最小のもの（tie-breaker のみで、撮影順とは解釈しない）。T0 より後の日付は代わりとしても使わない | [R] D-025, D-026 |
| frontal の判定 | `Modality` が CR/DX → `StudyDescription` で胸部検査の候補を特定 → `SeriesDescription` で frontal の AP / PA に限定。`ViewPosition` は主な判定基準にしない | [R] D-028 |
| 未決定の細部 | `AcquisitionTime` の欠測（Q-B9(b)）と、Series 内に複数の画像がある場合（Q-B9(c)）。監査で件数を確認し、該当例があれば研究責任者が決める | [R] D-030 |
| 固定コホート | **1,277 例**。Train 1,021（死亡 135）、Validation 128（死亡 17）、Test 128（死亡 17） | [R] D-024（Claude は未検証） |
| v1 | v1 の 1,299 例コホートと v1 の split は**使わない**。v1 の数値は履歴としてのみ扱う | [R] D-019 |
| 実行環境 | Google Colab（無料）、GPU T4 以上 | [T] |
| 保存先 | Google Drive の**「気合のCOVID19」フォルダ**。今回の成果物はすべてこの下に保存する。中は `00_governance`、`revision_v2_*` などに分ける（名前は Q-A3）。旧結果（v1）は移動も上書きもしない | [R] D-027、[FB §12] |
| 既存データ | DICOM は別フォルダに複製しない。既存の保存場所を確認してからアクセス方法を決める（Q-A4） | [R] D-029 |
| コード管理 | このリポジトリ（GitHub）。Colab から clone またはマウントして実行する | [提案]、[未確定 Q-A2] |
| 主な limitation | L-1：T0 当日の CXR が T0 の前か後かを判定できない | [R] requirements.md §6 |

---

## 1. 全工程（D-035 で順序を改訂）

**CXR の画像選定で最終解析コホートを固定し、その後に split を固定する。臨床データの監査と候補変数の選定は、固定コホートの Training set だけで行う。** 2026-09-18 に全 1,384 例で行った臨床データの監査と候補選定は「予備監査」として `docs/preliminary_pre_cohort_audit/` に保存しており、最終的な判断には使わない。

**現状**：外付け SSD が使えるようになるまで、工程 01 以降（CXR 監査、臨床側の追加解析）は開始しない（D-035）。

| # | 工程 | 主な作業 | 判断に使うデータ |
|---|---|---|---|
| 00 | 環境・版管理の準備 | Colab と Drive（気合のCOVID19）の構成。`00_governance` と `revision_v2_*` を作る。v1 の結果には触らない [FB §12] | – |
| 01 | **CXR 画像監査** | DICOM ヘッダの索引化（PatientID、UID、Modality、StudyDescription、SeriesDescription、StudyDate、AcquisitionDate/Time、SeriesNumber、PhotometricInterpretation、Instance 数など）。frontal の判定（D-028）に使う値と分類の対応表。`AcquisitionTime` の欠測と、Series 内に複数の画像がある件数（Q-B9(b)(c)：該当があれば研究責任者に判断を求める）。DICOM の日付と T0 の時間軸が整合しているか。重複画像 | 全患者（ラベルは見ない） |
| 02 | **index CXR 選定** | D-021, D-025, D-026, D-028 のルールで index CXR を選ぶ。T0 より後の日付がないことを確認する。T0−2、T0−1、T0 当日ごとの患者数を集計する（L-1） | 全患者（ラベルは見ない） |
| 03 | **コホートの再構築と照合（D-036）** | 確定済みのルール（D-020〜D-023, D-025, D-026, D-028）でコホートを作り直し、除外の流れ（理由ごとの人数）を記録する。**合格条件（D-037）**：(1) 患者の集合が既存の 1,277 例と完全に一致する。(2) 既存 manifest に index CXR の識別子があれば、患者ごとの index CXR も完全に一致する（StudyInstanceUID、SeriesInstanceUID、SOPInstanceUID、撮影日のうち manifest にあるものをすべて照合）。**どちらかが満たされなければ、1 例の食い違いでもここで止める**。そのときは差分表を報告する。患者集合の差分は「患者、除外理由、ルールの段階」、index CXR の差分は「subject_id、既存の index CXR、再構築した index CXR、撮影日、ルールの段階」の形で出す。推測での補正も、既存 manifest に合わせた選び直しもしない | 全患者（ラベルは見ない） |
| 04 | **既存の固定 split の採用（D-036）。2026-09-20 に完了（D-043）：`data/splits/COVID19_固定患者split_1277.csv`、SHA256 `626061a5…`。以後この split を全 Task で共通使用し、再割付しない** | 03 で完全に一致した場合に限り、既存の split（Train 1,021 / Validation 128 / Test 128）を**そのまま使う。作り直さない**。split manifest の SHA256 を記録し、次の点を確かめる：split 間で患者の重複がない、合計が 1,277 例、各 split の人数と死亡数（135 / 17 / 17）が D-024 と一致する。一致しなければ報告して止める | 割付のみ（死亡数の照合は manifest の記載値との一致確認だけ） |
| — | 研究プロトコル | 04 の後に `study_protocol.md` を作り、05〜07 で決まった事項を追記するたびに版を上げる | – |
| 05 | **臨床データの再監査（Train のみ）** | 列の型、欠測率、値の範囲、生理学的にあり得ない値（事前に決めた範囲による規則）[FB §6]、測定時点（Q-B3）。予備監査（`preliminary_pre_cohort_audit/`）の観点は参考にしてよいが、数値は Train で計算し直す | **Train** |
| 06 | **候補変数の選定** | 臨床的知識、先行研究、既存 risk score、T0 時点の利用可能性、欠測、リーク、冗長性に基づく候補表 [FB §2]。EPV の上限 [FB §1]。stepwise（仕様は Q-B4）[FB §3] | **Train** |
| 07 | **欠測補完・変換** | Train で fit：分布の確認（ヒストグラム、Q–Q、Shapiro–Wilk）[FB §4] → 変数ごとに平均値か中央値かを決める → 補完、スケーリング、One-Hot。variable と feature の対応表 [FB §5] | **Train** |
| 08 | 評価・チェック基盤 | bootstrap CI、DeLong（paired）、Holm、calibration、ECE、ROC/PR、Grad-CAM、リークチェック。合成データでテストする（コードだけなので 01〜07 と並行して作れる） | 合成データ |
| 09 | **Task B：Clinical models** | LR（class weight なしとあり [FB §11]）、XGBoost または LightGBM（欠測を補完する方式とそのまま扱う方式の比較 [FB §4]）、MLP [T] | Train で fit、Val で選択 |
| 10 | CXR 前処理 | pydicom、MONOCHROME1 の反転、正規化、224×224、3 channel [T Lv.2] | Train（統計量を使う場合） |
| 11 | Task A：CXR model | ResNet18（ImageNet の重み）の fine-tuning [T]。**事前に固定した複数の学習率**で学習し、全条件の履歴を保存する [FB §7] | Train で fit、Val で選択 |
| 12 | Task C：Late Fusion | 選んだ臨床モデルと CXR model の出力を decision level で統合する [FB §9]。重みや meta-model は Train/Val で決める | Train/Val |
| 13 | 凍結 | 最終モデル、checkpoint、閾値（Val で決める）、fusion の設定を記録し、各ファイルの hash を残す | Val |
| 14 | 最終 Test 評価 | 1 回だけ実行する。AUC（95% CI）、Holm で補正した DeLong 検定、ROC 図、Grad-CAM、Feature Importance | **Test** |
| 15 | 報告 | [T] の提出物一式、limitation（L-1〜L-4）。`change_log.csv` と成果物の更新チェックリストを確認する [FB §12] | – |

**工程番号の対応**：§1.1 の監査項目の番号（01-x〜04-x）と §2 の表の番号は v0.5 のまま残している（D-030 と open_questions から参照されているため）。新しい工程との対応は次のとおり。
- 旧 02（DICOM）と旧 04-1・04-2 → 新 01
- 旧 04-3〜04-6 → 新 02
- 旧 01 → 新 03・04
- 旧 03（臨床）→ 新 05（Train のみ）
- 旧 06・09 → 新 06
- 旧 08 → 新 07
- 旧 07 → 新 08
- 旧 11 → 新 09
- 旧 10〜16 → 新 10〜15

### 1.1 データ監査の項目（工程 01〜04）

転帰を見る項目は **Train のみ** で行う。それ以外の項目ではラベルを見ない。

**01 固定コホートと split**
- 01-1：manifest ファイル（コホート、split、index CXR）の SHA256 を記録する
- 01-2：人数（1,277；1,021 / 128 / 128）と死亡数（135 / 17 / 17）が D-024 と一致するか
- 01-3：subject_id が一意であり、split 間で重複がないか（3 通りの積集合がすべて空）
- 01-4：split を作った方法（層別の有無、seed）の記録があるか
- 01-5：生データから D-020〜D-023 のルールでコホートを作り直すと、manifest の患者集合と完全に一致するか。一致しない場合、その差分と理由
- 01-6：元の母集団から 1,277 例に至るまでの除外の流れ（理由ごとの人数）を作れるか

**02 DICOM**
- 02-1：コホートの全患者で、index CXR の DICOM を読み込めるか
- 02-2：frontal の判定（D-028）に使う値の一覧を作る。`Modality` の分布、`StudyDescription` の値ごとの件数（胸部検査の候補かどうか）、`SeriesDescription` の値ごとの件数（frontal の AP / PA かどうか）。値と分類の対応表を `00_governance` に保存し、研究責任者の確認を受けて固定する。コホートを作ったときの対応表があれば、それと照合する。`ViewPosition` は判定に使わない（件数の参考集計にとどめる [提案]）
- 02-2b：選んだ Series に複数の画像（Instance）がある患者の数（Q-B9(c)）。**該当例があれば一覧を作り、研究責任者に判断を求める。扱いは Claude が決めない**（D-030）
- 02-3：PhotometricInterpretation（MONOCHROME1 か 2 か）、BitsStored、画素サイズ、画像サイズの分布
- 02-4：同じ画像や非常によく似た画像が、異なる患者に存在しないか（pixel hash）
- 02-5：画像内の文字やマーカー（目視で標本を確認する）

**03 臨床データ**（→ 新工程 05。**Train のみ**で行う。2026-09-18 に全データで行ったものは予備監査：[preliminary_pre_cohort_audit/clinical_audit_report.md](preliminary_pre_cohort_audit/clinical_audit_report.md)。最終的な判断には使わない）
- 03-1：列一覧、型、欠測率。欠測率は Train で計算する
- 03-2：各列を「T0 時点で利用可能」「T0 後の情報」「転帰そのもの、または転帰に由来する情報」に分類する。治療、ICU、人工呼吸、在院日数などは T0 後の情報にあたる
- 03-3：**臨床変数の測定・記録時刻の情報がデータにあるか**。ある場合は T0 との前後関係を確認する（Q-B3、L-3）
- 03-4：転帰の列の値の種類と件数（転院、ホスピス、不明などの扱い：Q-B2）
- 03-5：単位の一貫性、年齢の表し方（連続値か階級か）
- 03-6：生理学的にあり得ない値の件数。事前に決めた範囲による規則で数え、処理は記録する

**04 T0 と index CXR**
- 04-1：`visit_start_datetime` の形式、時刻の部分の有無、欠測、1 人の患者に複数の visit があるか
- 04-2：DICOM の日付と `visit_start_datetime` が同じ時間軸にあるか（de-identification による日付シフトの整合性）
- 04-3：index CXR の撮影日がすべて T0−2 日から T0 当日の範囲にあり、T0 より後の日付が 1 件もないこと
- 04-4：index CXR の撮影日が T0−2、T0−1、T0 当日にあたる患者数（全体と split 別。転帰別の集計は Train のみ）
- 04-5：選んだ撮影日に複数の frontal Series がある患者の数。同じ時刻の Series がある件数（`SeriesNumber` による tie-breaker が働いた件数）。`AcquisitionTime` が欠けている件数（Q-B9(b)）。**欠測があり、index の選択に影響する例があれば一覧を作り、研究責任者に判断を求める。扱いは Claude が決めない**（D-030）
- 04-6：manifest の index CXR が、D-025 の選択ルールで完全に再現できるか。再現できない患者がいれば、その一覧と理由

---

## 2. 各工程の入力と出力

保存先は Drive の `気合のCOVID19/revision_v2_*/` 以下で（D-027）、ローカルの `data/` と `outputs/` は同じ構造にそろえる [提案]。`G/` は `00_governance/` を表す。

| # | 入力 | 出力 |
|---|---|---|
| 00 | – | `G/analysis_plan.md`、`G/change_log.csv`、`G/environment.txt`、Drive のフォルダ構成 |
| 01 | 固定コホートと split の manifest、生データ | `G/cohort_manifest.csv`、`G/split_manifest.csv`（subject_id, split, label, index_cxr_id）とその SHA256、`G/cohort_verification.md`（照合の結果と、作り直したコホートとの差分）、`interim/exclusion_log.csv`、コホートのフローチャート |
| 02 | DICOM、cohort manifest | `interim/dicom_index.parquet`、`interim/duplicate_images.csv`、`interim/dicom_audit_summary.csv` |
| 03 | 臨床 CSV、cohort manifest | `interim/clinical_audit.csv`、`G/data_dictionary.md`（時点の分類を含む）、`G/implausible_value_rules.csv`、`interim/implausible_value_log.csv`（変数、症例、根拠、処理）[FB §6] |
| 04 | 02・03 の出力 | `interim/t0_audit.csv`（患者ごとの index CXR の撮影日と T0 の日数差、`AcquisitionTime`）、`tables/cxr_day_offset_distribution.csv` |
| 05 | 01〜04 の出力 | `G/study_protocol.md`（版つき） |
| 06 | 文献、data dictionary | `G/candidate_variables.csv`（FB §2 の列構成） |
| 08 | Train の特徴量 | `G/imputation_rationale.csv`（変数、分布の判定、Shapiro–Wilk の結果、採用した代表値、図へのパス）、`processed/clinical/preprocessor.joblib`、`G/variable_feature_map.csv` [FB §5]、分布図 |
| 09 | 前処理済み Train | `G/stepwise_log.csv`（各ステップ）、`G/final_variables.csv` |
| 10 | DICOM、cohort manifest | `processed/images/*.png`、`processed/image_manifest.csv` |
| 11〜13 | 前処理済みデータ、`G/model_settings.csv` | `runs/<task>/<model>/<run_id>/`（§5 参照）、`predictions/validation/val_pred_<model>.csv` |
| 14 | 各 run | `G/final_selection.json` |
| 15 | 凍結済みモデル、Test | `predictions/test/test_predictions_all_models.csv`、`metrics/test_metrics.csv`、`metrics/delong_holm.csv`、`figures/roc_all_models.*` と元データ CSV、`figures/gradcam/`、`figures/feature_importance/`、`logs/test_access_log.jsonl` |
| 16 | 上記すべて | 性能評価表、学習曲線、考察の材料、`G/output_update_checklist.csv` |

---

## 3. データリークと時点の不整合を防ぐチェックポイント

各チェックは `src/covid_mortality/checks/` に実装し、失敗したら処理を止める。

### A. 患者・ID
- A1：split 間で subject_id が重ならない（3 通りの積集合がすべて空）[U][T]
- A2：subject_id が一意であり、臨床と DICOM の ID が 1 対 1 に対応する
- A3：同じ画像や非常によく似た画像が、異なる患者 ID に存在しない [提案]
- A4：split manifest の hash が、以降のすべての工程で、工程 01 で記録した値と一致する [FB §12]
- A5：画像、特徴量、予測の subject_id 集合が split manifest（1,277 例）と一致する

### B. 時点（[FB §8]、[R]）
- B1：index CXR の撮影日が T0−2 日から T0 当日の範囲にあり、T0 より後の日付がない（D-021）
- B2：T0 当日の CXR について、T0 との前後を判定しない。`AcquisitionTime` を T0 との比較に使わない（D-022, D-023）。T0 当日の患者の割合を報告する（L-1）
- B2b：index CXR は D-025 の選択ルールで決まるものに限る。frontal であること（D-028 の判定）、T0 に最も近い撮影日であること、その日の中で `AcquisitionTime` が最も早いことを確認する。`SeriesNumber` は tie-breaker にだけ使う（D-026）
- B3：モデルに入れる臨床変数がすべて、data dictionary で「T0 時点で利用可能」に分類されている。治療、ICU、人工呼吸、在院日数、転帰に由来する列は除く
- B4：時刻が不明な臨床項目は、事前に決めたルールで扱う（Q-B3）
- B5：特徴量を列名の allowlist で指定する [提案]

### C. 前処理・変数選択
- C1：補完値（平均値か中央値か）の判定、Shapiro–Wilk 検定、スケーラ、One-Hot のカテゴリ水準は Train のみで決める [FB §4]
- C2：候補変数の欠測率と stepwise は Train のみで計算・実行する [FB §3]
- C3：生理学的にあり得ない値の判定は、事前に決めた範囲による規則で行い、データから閾値を推定しない。処理はすべての split に同じ規則で適用し、記録する [FB §6]
- C4：EDA と、転帰を使う監査は Train のみで行う [提案]

### D. モデル選択と Test の隔離（[U][FB §14]）
- D1：学習率、hyperparameter、early stopping、checkpoint は Val（または Train 内 CV）で選ぶ
- D2：閾値、fusion の重み、meta-model は Train/Val で決める
- D3：Test の読み込みは工程 15 のみ。`final_selection.json` と hash が一致しない場合は実行しない [提案]
- D4：Test へのアクセスは記録する。Test 評価の後に変更した場合は `change_log.csv` に記録し、事後解析として明示する [FB §12]

### E. 比較の妥当性
- E1：すべてのモデルを同じ Test 患者集合（128 例）で評価する
- E2：モデル間の結合は subject_id をキーとした join で行う
- E3：Late Fusion に使う単体モデルと、比較対象として報告する単体モデルを同じにする
- E4：Late Fusion の meta-model を Train の予測で学習する場合、Train の予測は in-sample で過大評価されているため使わない。out-of-fold 予測を使うか、Val で決める [提案]

### F. 近道学習の監視 [提案]
- F1：ポータブル AP と PA の比率を転帰別に確認する（Train のみ）。Grad-CAM で画像内の文字やマーカーへの注目がないかを確認する

---

## 4. モデル仕様（資料で決まっている部分と未確定の部分）

各モデルの設定は `G/model_settings.csv` に、[FB §13] の項目（初期化、構造、loss、optimizer、学習率、batch size、epoch、class weight、正則化、scheduler、early stopping、探索範囲、選択基準）で固定する。v1 の設定（[FB §7]）を引き継ぐ場合は、項目ごとに decision log に記録する。v1 の `pos_weight` などコホートに依存する値は、今回の Train から計算し直す。

### Task A：CXR
- ResNet18、ImageNet の重み、最終層の出力は 1 [T]。「fine-tuning」と呼ぶのはこのモデルだけ [FB §7]
- 学習率は複数を事前に固定する（1×10⁻⁴ を含む）[FB §7]。範囲、scheduler、patience は [未確定 Q-C1]
- 全条件について、train/val の loss、ROC-AUC、PR-AUC の推移を保存する [FB §7]。[T] の要件として Accuracy の推移も保存する（閾値を明記）
- 不均衡データへの対応：pos_weight または WeightedRandomSampler [T Lv.2]
- Grad-CAM：成功例と失敗例 [T]。定義は [未確定 Q-D4]

### Task B：Clinical
- LR：class weight なしと、balanced（または指定した重み）の 2 条件 [FB §11]。重みの指定方法は [未確定 Q-C2]
- XGBoost または LightGBM [T]。欠測を補完したデータで学習する方式と、欠測をそのまま扱う方式を比較する [FB §4]
- MLP（PyTorch）[T]
- 3 モデルで同じ最終変数セットを使うかどうか [未確定 Q-B5]
- Feature Importance [T]：手法と計算に使うデータは [未確定 Q-D5]
- 調整を Val で行うか、Train 内 CV で行うか [未確定 Q-C3]

### Task C：Late Fusion [FB §9]
- 統合する臨床モデル、統合対象（確率か logit か）、統合方法（単純平均、重み付き平均、meta-model）は [未確定 Q-C4]
- 重みや meta-model は Train/Val のみで決める。Test は使わない
- Early Fusion を副解析として残すか [未確定 Q-C5]。残す場合は、Train で fine-tune した ResNet から Train 画像の特徴量を取り出すと過適合した特徴量になる、という点に対処する必要がある [提案]

---

## 5. 再現性のために保存する成果物

### `00_governance/`（[FB §12] で明示）
- 解析計画（`study_protocol.md`）、変更履歴（`change_log.csv`：発見日、問題、原因、影響する Task とファイル、修正方法、再実行の要否、確認結果）
- 固定コホート（1,277 例）と split の manifest とその hash、特徴量定義（候補変数表、variable と feature の対応表、最終変数）、seed、評価ルール
- 成果物の更新チェックリスト（論文、Notebook、表、図、スライド）
- [提案] 追加：`decision_log`、`environment.txt`、`cohort_verification.md`、補完の根拠表、生理学的にあり得ない値の処理記録、stepwise の記録

### 各 run（[FB §12] で明示）
- `config.json`、`metrics.json`、学習履歴、best checkpoint、実行済み Notebook、実行日時、ソース hash
- [提案] 追加：`val_predictions.csv`（subject_id, true_label, prob）、`hparam_search.csv`（全試行）、last checkpoint、split manifest の hash

### 最終評価
- Test 予測：subject_id、true_label、各モデルの確率（患者 1 行）[U]。index CXR の撮影日と T0 の日数差（−2、−1、0）の列も含める（L-1 の感度分析 Q-D7 に備える）[提案]
- AUC と 95% CI（bootstrap [T]。反復回数は [未確定 Q-D2]）、paired DeLong と Holm 補正（family の定義を明記）[T][FB §10]
- 全モデルの ROC 曲線を 1 枚にまとめた図と、描画に使った元データ [T]
- LR の class weight 比較：PR-AUC、感度、特異度、Brier、ECE、calibration plot [FB §11]（他のモデルにも出すのは [提案]）
- Grad-CAM（成功例と失敗例）、Feature Importance（上位の因子）[T]

---

## 6. 実装順序

| Phase | 内容 | 補足 |
|---|---|---|
| A | 00〜02：版管理の準備、**CXR 画像監査、index CXR 選定** | 外付け SSD が使えるようになってから開始する（D-035） |
| B | 03〜04：**コホートを再構築して 1,277 例と照合 → 一致すれば既存の split を採用**（D-036） | **関門**：患者集合、または（manifest に識別子がある場合の）index CXR が 1 例でも一致しなければ、split にも後続の解析にも進まず、差分表を報告する（D-037）。既存のコホートと split の manifest の置き場所は Q-A4 |
| C | 05〜07：**Train だけで**臨床データの再監査 → 候補変数の選定 → 欠測補完・変換 | 予備監査の数値は使わない |
| D | 08：評価・チェック基盤 | コードだけなので A〜C と並行して作れる。モデルより先に完成させる |
| E | 09：Task B | 計算が軽いため、パイプラインの確認を兼ねる |
| F | 10〜11：CXR 前処理、Task A | 複数の学習率で学習する。Colab の T4 で時間がかかる |
| G | 12：Task C の Late Fusion | E と F の結果に依存する |
| H | 13〜15：凍結 → Test 評価 → 報告 | Test に触れるのはここで 1 回だけ |

各 Phase の終わりに結果を報告し、承認を得てから次に進む。

---

## 7. 設計上のリスクと limitation

- **L-1（時点の曖昧さ）**：正確な admission time がないため、T0 当日の CXR が T0 の前か後かを判定できない。T0 当日の CXR には、T0 の数時間後に撮影されたものが含まれる可能性がある。T0 当日の患者の割合を報告し、感度分析（Q-D7）を検討する
- **検出力**：Test は 128 例（死亡 17 例）。AUC の 95% CI は広く、paired DeLong で差を検出できない可能性が高い。「有意差なし」を「同等」と解釈しないよう、考察に明記する [提案]
- **Validation の小ささ**：Val の死亡は 17 例。Val で選ぶ項目（学習率、checkpoint、XGB の設定、閾値、fusion の重み）が多いほど、Val への過適合と選択のばらつきが大きくなる。[FB §7] は、v1 で CXR の epoch 1 が best になった原因の候補の 1 つとして Val の小ささを挙げている
- **EPV の制約**：Train の死亡 135 例を基準にすると、説明変数は約 13 が目安になる（全体の 169 例を基準にすると約 17。どちらを使うかは Q-B4）[FB §1]。One-Hot で展開すると、すぐにこの数を超える
- **stepwise**：[FB] で提案された方法。選択後の係数や p 値は楽観的になりやすいため、報告では選択の過程を明示する [提案]
- **Colab の制約**：セッションが切れるため、checkpoint と履歴は各 epoch で Drive に保存し、途中から再開できるようにする [提案]
