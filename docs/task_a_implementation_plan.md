# Task A（CXR モデル）実装計画

- 作成日：2026-09-20　版：v0.1（**未承認。学習は未実施**）
- 対象：入院時 CXR（index CXR）1 枚から院内死亡を予測する ResNet18 モデル
- 参照範囲：Google Drive「気合のCOVID19」内の現行ファイルのみ。「旧」で始まるファイル、過去版資料、「心機一転COVID19」は参照しない
- 最優先の制約：教員フィードバック（[FB]）
- 表記：**[確定]** すでに決まっている／**[決定]** 本計画で Claude が決める／**[要確認]** 追加確認が必要

---

## 0. 固定条件（[確定]）

| 項目 | 値 | 記録 |
|---|---|---|
| 最終解析コホート | 1,277 例 | D-039, D-040 |
| 固定 split | Train 1,021 / Val 128 / Test 128 | D-043 |
| 死亡数 | 135 / 17 / 17 | D-043 |
| split の扱い | 変更・再作成・再割付しない。患者単位を維持 | D-043 |
| 画像 | 選択済み CXR 1,277 枚。PatientID・SeriesInstanceUID・SOPInstanceUID まで既存と完全一致 | D-041 |
| Test | 最終評価のみ。前処理選択、augmentation 選択、モデル選択、hyperparameter 調整、checkpoint 選択、閾値決定に使わない | [U][T][FB §14] |
| 課題要件 | ResNet18（転移学習可）、ROC-AUC と 95% CI、学習曲線、Grad-CAM、再現可能なコード・設定・結果 | [T] |
| 用語 | fine-tuning と呼ぶのは本モデルのみ | [FB §7] |

---

## 1. 現在利用可能なデータ・成果物の監査

### 確認できたもの [確定]
| 対象 | 状態 |
|---|---|
| 選択済み DICOM 1,277 枚 | `My Drive\気合のCOVID19\01_TaskA_CXR\01_前処理\selected_dicom_1277`。読み込みエラー 0、`pixel_array` 復号エラー 0 |
| ヘッダー監査 | 全枚 MONOCHROME2、BitsStored 12、PixelRepresentation 0、RescaleSlope 1 / Intercept 0、WindowCenter 2048 / WindowWidth 4096（単一値）、VOILUTFunction 欠損、PresentationLUTShape IDENTITY、PixelSpacing 全枚あり |
| 画素 | uint16、実測範囲 0〜4095 |
| 装置 | CR 1,246（CARESTREAM HEALTH）／DX 31（Carestream）。DX のみ PixelIntensityRelationship = LOG、Sign = −1 |
| サイズ | 185 種類。横長 1,216 枚／縦長 61 枚。aspect ratio 0.639〜1.242 |
| 固定 split | `data/splits/COVID19_固定患者split_1277.csv`（SHA256 `626061a5…`） |
| index CXR manifest | `data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv`（SHA256 `e8c07db7…`） |
| コピー監査ログ | `CXR_1277_DICOMコピー監査ログ.csv`（Subject ID, Series UID, status, source, dest） |

### 確認できていないもの [要確認]
- 現行の前処理 Notebook / スクリプト / 設定ファイル（ローカルから見えない）
- `processed_png_1277` の内容（bit 深度、サイズ、正規化方法）
- v1 の学習ログ・checkpoint（再現性確認の参照用）

→ **本計画は、これらに依存しない形で設計する**。

**参照制限（2026-09-20 の研究者指示、[確定]）**：独立解析を保つため、**既存の `processed_png_1277` および既存 Task A の詳細な前処理仕様は、エージェントによる Task A が完了するまで参照しない**。既存解析の監査は最終比較フェーズで行う。したがって §15 の Q1（現行 Notebook の仕様）は、Task A 完了までは確認しない。

---

## 2. 画像前処理方針 [決定]

**方針：データから統計量を推定しない、画像ごとに閉じた決定的な処理にする。** これにより Train で fit する対象がなくなり、split をまたぐ情報の流れが原理的に生じない。

| 工程 | 採用する方法 | 理由 |
|---|---|---|
| 読み込み | `pydicom` で `pixel_array`（uint16） | 復号エラー 0 を確認済み |
| Modality LUT | `RescaleSlope`/`Intercept` を形式的に適用 | 全枚 1 / 0 のため値は不変だが、将来の再利用と明示性のため実装する |
| Photometric | `MONOCHROME1` なら反転、`MONOCHROME2` はそのまま | 現データは全枚 MONOCHROME2 で発火しないが、条件分岐は残す |
| windowing | **DICOM の WindowCenter/Width は使わない** | 全枚 2048/4096 で 12 bit 全域を指す既定値であり、コントラスト正規化にならない |
| intensity normalization | **画像ごとの percentile clipping（1–99%）→ [0,1] へ min–max 変換** | 焼き込み文字（白）やコリメーション領域（黒）の極端値に影響されにくい。データセット統計を使わないためリークが生じない |
| foreground crop | **行わない（v1 の仕様と独立に判断）** | 胸部領域の自動抽出は誤検出時に肺野を切る危険がある。代わりに padding で全体を保持する。将来、Train のみで効果を検証する余地は残す |
| rotation correction | **行わない** | 傾いた画像は存在するが（例：A074083）、自動補正は失敗時に不可逆な歪みを生む。傾きは augmentation の範囲内で吸収する |
| aspect ratio | **保持する**。長辺に合わせて正方形に zero padding | 単純リサイズは心胸郭比などの形態情報を歪める。横長 1,216 枚・縦長 61 枚と比率のばらつきが大きい |
| resize | padding 後に 512×512 へ（キャッシュ）→ 学習時に 224×224 | 224×224 は ResNet18 / ImageNet の標準 [T][FB §7]。512 キャッシュにより、後で解像度を変える検討が DICOM 再復号なしで可能 |
| 中間形式 | **16 bit PNG（512×512）** | 12 bit の情報を保ったまま、Colab での DICOM 復号を毎 epoch 繰り返さずに済む。8 bit 化は情報損失のため採らない |
| 中間形式の QC | **round-trip QC を必須とする**：DICOM から前処理した値 → PNG 保存 → PNG 再読込 で、dtype、dynamic range、画素値の変換が想定どおりかを全枚確認する。想定外なら**実装を止めて報告する** | 16 bit PNG は読み書きのライブラリによって dtype や値域の扱いが変わりうるため、学習前に保証しておく |
| grayscale → 3 channel | 同じ channel を 3 枚複製 | ImageNet 事前学習の入力仕様に合わせる |
| ImageNet normalization | **適用する**（mean/std は ImageNet の既定値） | ImageNet 事前学習重みを使うため、入力分布を事前学習時に合わせる。Train から統計量を推定する案は、リーク対策の手間に対して利得が小さい |
| Train / Val / Test | **決定的処理は完全に同一**。augmentation は Train のみ | [FB §7]、研究設計上の原則 |

**前処理を Validation で選ばない**：Val の死亡は 17 例で、前処理の候補を Val の AUROC で選ぶと Val に過適合しやすい。前処理は上記のとおり技術的根拠で事前に固定し、比較対象にしない。

---

## 3. Dataset / DataLoader 設計 [決定]

- 入力は `index_cxr_manifest_window_T0m2_T0.csv` と固定 split を `PatientID` で内部結合したもの。**1 患者 1 画像**
- `__getitem__` は 16 bit PNG を読み、float32 [0,1] → augmentation（Train のみ）→ 224×224 → 3 channel → ImageNet 正規化
- ラベルは split manifest の `true_label`（deceased=1）
- `num_workers` あり。worker ごとに seed を固定（`worker_init_fn`）
- サンプリングは通常のシャッフル（不均衡対策は損失側で行う。§7 参照）
- **整合性チェック**：データセット構築時に、split 間の患者重複 0、総数 1,277、各 split の人数と死亡数が D-043 と一致することを assert する

---

## 4. ResNet18 モデル設計 [決定]

| 項目 | 内容 | 理由 |
|---|---|---|
| アーキテクチャ | `torchvision` ResNet18、`IMAGENET1K_V1` 重み | [T] の要件。[FB §7] で fine-tuning の対象と定義済み |
| 出力層 | `Linear(512, 1)`（ロジット 1 出力） | 二値分類。`BCEWithLogitsLoss` の `pos_weight` を使えるため |
| 入力 | 3×224×224 | ImageNet 事前学習の仕様 |
| 学習範囲 | 全層 fine-tuning を主とし、比較条件として「1 epoch だけ head のみ学習 → 全層解凍」を置く | [FB §7] は「早期 overfitting」「warm-up」「凍結層」を検討事項として挙げている |

---

## 5. 学習戦略 [決定]

| 項目 | 値 | 理由 |
|---|---|---|
| 損失 | `BCEWithLogitsLoss(pos_weight)` | §7 |
| optimizer | AdamW | 転移学習で安定。weight decay を分離して扱える |
| weight decay | 1×10⁻⁴ | 小規模データでの過学習抑制。極端な値は避ける |
| 学習率 | **事前に固定した 4 条件：1×10⁻⁵ / 3×10⁻⁵ / **1×10⁻⁴** / 3×10⁻⁴** | [FB §7] が「1×10⁻⁴ を含む複数の学習率で再学習」を求めている |
| batch size | 32 | Colab T4 のメモリに収まり、1,021 例で 32 iteration/epoch となり勾配推定が安定 |
| epoch | 最大 30 | [FB §7] の v1 と同水準。early stopping で打ち切る |
| scheduler | cosine annealing（warm-up 1 epoch） | 学習率を単調に下げるため挙動が再現しやすい。Val 依存の `ReduceLROnPlateau` は Val の揺らぎ（死亡 17 例）に反応しやすいため採らない |
| early stopping | Val AUROC を監視、patience 8、最小改善 0.005 | v1 は patience 5 / 0.001 で epoch 1 が best になった。patience を伸ばし、改善幅の閾値を上げて、偶然の揺らぎで止まりにくくする |
| 混合精度 | 使用（AMP） | T4 での学習時間短縮。数値差は seed 管理と併せて記録する |
| seed | 3 seed（42, 43, 44） | Val の揺らぎの大きさを定量化するため。単一 seed の結果で条件を選ばない |

**比較する候補条件（Validation のみで比較）**
1. 学習率 4 条件 × augmentation 2 条件（§6）= 8 条件 × 3 seed = **24 run**
2. 上位条件に対してのみ、warm-up + 段階的解凍を追加検証（2 条件 × 3 seed = 6 run）

条件数を絞るのは、Val の死亡が 17 例しかなく、多数の条件を Val で比較すると Val への過適合が避けられないため。**全条件の学習履歴を保存する**（[FB §7]）。

---

## 6. augmentation [決定]

| 条件 | 内容 |
|---|---|
| **Aug-A（弱）** | 平行移動 ±3%、scale 0.97–1.03、brightness / contrast 0.95–1.05 |
| **Aug-B（中）** | 平行移動 ±5%、回転 ±7°、scale 0.95–1.05、brightness / contrast 0.90–1.10 |

- **左右反転は行わない**。心臓の位置など左右の解剖学的情報が失われ、焼き込みの L/R マーカーとも矛盾するため
- 上下反転・大きな回転も行わない
- 回転 ±7°（Aug-B）は、実際に傾いた画像が存在することへの対応
- augmentation は **Train のみ**。Val / Test には適用しない

---

## 7. class imbalance 対策 [決定]

- **主方式：`pos_weight` = （Train の生存数）/（Train の死亡数）= 886 / 135 ≈ 6.56**（Train からのみ算出）
- 副方式として `WeightedRandomSampler` を置かない。sampler は epoch ごとに実質的な学習分布が変わり、学習曲線の解釈と再現が難しくなるため
- Val / Test はリサンプリングしない。評価は実際の有病率のまま行う
- 評価指標は AUROC を主とし、PR-AUC を併記する（不均衡下での挙動を見るため）

---

## 8. Validation によるモデル選択・checkpoint 選択 [決定]

| 段階 | 規則 |
|---|---|
| epoch 内 | 各 epoch 終了時に Val AUROC・Val loss・PR-AUC を記録 |
| checkpoint | **Val AUROC 最大の epoch**。同値なら Val loss が小さい方。さらに同値なら epoch 番号が小さい方 |
| 条件選択 | 各条件の **3 seed の Val AUROC の平均**で比較し、最大の条件を採用。平均が同点なら標準偏差が小さい方 |
| 最終モデル | 採用条件の **3 seed の予測確率の平均（seed ensemble）** | 単一 seed の偶然の当たりを避け、Val の揺らぎに対して頑健にする |
| 予測の保存 | **各 seed 単独の患者単位予測と性能を必ず保存する**（`val_predictions_seed<NN>.csv` / `test_predictions_seed<NN>.csv` と、それぞれの指標）。**ensemble の予測は別ファイル**（`*_ensemble.csv`）に保存する | 最終比較の段階で、単一 seed モデルと ensemble を区別して評価できるようにするため |
| 記録 | 全 run の `history.csv`、全条件の比較表、選択理由を残す [FB §7, §13] |

**epoch 1 問題への対応**：v1 で epoch 1 が best になった件は、patience の延長、最小改善幅、cosine scheduler、seed 平均での条件選択によって、偶然の揺らぎが選択に与える影響を小さくする。原因の断定はしない。

---

## 9. 最終 Test 評価方法 [決定]

1. §8 までで決めた条件・checkpoint・閾値・seed ensemble の構成を `final_selection.json` に**凍結**し、各ファイルの SHA256 を記録する
2. Test 評価スクリプトは、凍結記録があり hash が一致する場合のみ実行できる
3. Test 予測は**一度だけ**生成し、`subject_id, true_label, prob` を保存する（他 Task と `subject_id` で結合するため）
4. Test へのアクセスをログに記録する
5. 算出する指標：AUROC（95% CI）、PR-AUC、Brier score、calibration、Val で決めた閾値での感度・特異度・PPV・NPV
6. Test 評価後にモデルを変更した場合は、事後解析として明示する [FB §12]

---

## 10. ROC-AUC 95% CI の算出方法 [決定]

- **主方法：患者単位の層別 bootstrap（2,000 反復、percentile 法）** — [T] Lv.3 が bootstrap を指定。層別にする理由は、**イベントが少ない Test set で class composition（死亡・生存の構成比）を保ち、CI の推定を安定させるため**
- **副方法：DeLong 法の CI を併記** — 解析的な区間との整合を確認するため
- 乱数 seed を固定し、bootstrap 標本の指標も保存する
- Task C の比較では paired DeLong + Holm 補正を使う（Task A 単独の CI とは別）

---

## 11. Grad-CAM 設計 [決定]

| 項目 | 内容 | 理由 |
|---|---|---|
| 対象 layer | `layer4` の最終 conv（最後の残差ブロックの出力） | 空間情報を保った最終段で、Grad-CAM の標準的な選択 |
| 実装 | 勾配ベースの標準 Grad-CAM（外部実装に依存しない自前実装＋単体テスト） | 再現性の確保 |
| 対象データ | **開発中は Val のみ**。Test への適用は凍結後の最終評価時に一度だけ | 可視化結果を見てモデルを調整すると Test が汚染されるため |
| 症例選択 | 事前に規則を固定（閾値は Val で決めたもの）：**TP は予測確率の高い順に 4 例、FP も予測確率の高い順に 4 例、TN は予測確率の低い順に 4 例、FN も予測確率の低い順に 4 例**。同値の場合は PatientID の昇順で一意に決める | 「成功例・失敗例」を恣意的に選ばない [T]。FP は「自信を持って誤った陽性」、FN は「自信を持って見落とした陽性」を見るため、それぞれ確率の高い側・低い側から選ぶ |
| 確認事項 | 焼き込み文字（PORTABLE、L/R マーカー）や画像端に注目していないか | 近道学習の監視（[提案] F1） |

---

## 12. 保存する成果物 [決定]

**run ごと**（`runs/taskA/<condition>/<seed>/`）
- `config.json`（全設定）、`seed.json`、`history.csv`（epoch ごとの train/val loss、AUROC、PR-AUC、学習率、所要時間）
- `checkpoints/best.pt`、`last.pt`、選択基準の記録
- `val_predictions.csv`（subject_id, true_label, prob）
- `run_meta.json`（実行日時、ソース hash、データ hash、環境、GPU）

**Task A 全体**
- 前処理キャッシュ（16 bit PNG 512×512）と `preprocess_config.json`、画像ごとの処理記録
- 全条件の比較表（`condition_comparison.csv`）、学習曲線の図
- `final_selection.json`（凍結記録）
- `test_predictions.csv`、`test_metrics.json`、ROC / PR 曲線の元データ
- Grad-CAM 画像と対象症例の一覧
- `change_log.csv` への追記

---

## 13. data leakage 防止策 [決定]

| # | 対策 |
|---|---|
| L1 | split は固定ファイルから読み込むのみ。再割付しない。読み込み時に hash を照合 |
| L2 | 前処理は画像ごとに閉じた決定的処理で、データセット統計を使わない（§2） |
| L3 | augmentation は Train のみ |
| L4 | Val は選択にのみ使用。Test は凍結後の 1 回のみ、専用スクリプトからのみ読み込む |
| L5 | 1 患者 1 画像。重複画像が split をまたがないことを確認（既に確認済み） |
| L6 | Grad-CAM の Test 適用は凍結後 |
| L7 | 閾値は Val で決定 |
| L8 | Test アクセスログを残す |

---

## 14. 再現性確保 [決定]

- seed：Python / NumPy / PyTorch / CUDA、`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、DataLoader の worker seed
- 完全な決定性が AMP や一部 CUDA 演算で保証されない場合は、その旨を記録し、3 seed の結果の幅を併記する
- 環境（Python、torch、torchvision、CUDA、GPU 型番）、コードの commit hash、入力ファイルの SHA256 を各 run に保存
- Colab のセッション切断に備え、epoch ごとに checkpoint と history を Drive へ保存し、再開できるようにする

---

## 15. 想定されるリスク・未解決事項

| # | 内容 | 区分 |
|---|---|---|
| R1 | Test の死亡 17 例では AUROC の 95% CI が広く、Task C との差を検出する力が乏しい | [確定した制約] |
| R2 | Val の死亡 17 例では、条件選択・checkpoint 選択・閾値決定のばらつきが大きい。条件数を絞り、seed 平均で選ぶことで緩和するが解消はしない | [決定で緩和] |
| R3 | 焼き込み文字・PORTABLE 表記・装置差（CR 1,246 / DX 31）による近道学習 | Grad-CAM と撮影条件別の確認で監視 |
| R4 | 傾いた画像、コリメーション領域、黒い楔 | crop しない方針のため残る。Aug-B の回転で部分的に対応 |
| R5 | T0 当日の CXR が入院前か後か判定できない（L-1） | 研究全体の limitation |
| R6 | 決定的アルゴリズムと AMP の組み合わせで完全一致しない可能性 | 記録して幅を報告 |
| Q1 | 現行 Notebook / `processed_png_1277` の仕様（16 項目のうち 13 項目が不明） | **[要確認]** |
| Q2 | 前処理キャッシュを Drive のどこに置くか（既存の `processed_png_1277` を上書きしない） | **[要確認]** |
| Q3 | 計算資源の上限（Colab の GPU 時間）。24〜30 run が可能か | **[要確認]** |
| Q4 | 本計画の候補条件（学習率 4 × augmentation 2）で教員の意図（[FB §7] の「複数の学習率」）を満たすか | **[要確認]** |
| Q5 | Task C（Late Fusion）で使う画像特徴量の取り出し方（本計画では未決定） | 後続で決定 |

---

## 実行順序（承認後）

1. 前処理キャッシュ作成（1,277 枚 → 16 bit PNG）と処理記録の保存
2. Dataset / DataLoader と整合性チェックの実装、単体テスト
3. 学習コードの実装（seed 固定、履歴保存、再開機能）
4. 24 run の学習（Train + Val のみ）
5. 条件比較 → 追加 6 run → 最終条件の決定、閾値の決定
6. 凍結（`final_selection.json`）
7. **Test 評価（1 回のみ）** → 指標、95% CI、ROC 曲線、Grad-CAM
8. 成果物の整理と報告
