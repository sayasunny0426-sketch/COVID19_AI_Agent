# 未確定事項

- 改訂：v1.1（2026-09-18。Q-A7 を close（D-037））
- v1.0（2026-09-18。Q-A6 を (a) で close（D-036）。Q-A7 を追加）
- v0.9（2026-09-18。D-035 の作業順序の変更を反映し、臨床側の質問を保留とした。Q-A6 を追加）
- 前回の改訂：v0.8（2026-09-18。data dictionary との照合を受けて、Q-B11 と Q-B13 を更新し、Q-B16 を追加。Q-B15 は研究者の指示で削除した（辞書に use_category 列はなく、過去判断との比較は行わない）。Q-B3 と Q-B16 に辞書の定義からの注記を追加）
- 基準資料 [T]（課題文）、[FB]（教員フィードバック）、[R]（研究者の決定）で解決した項目は、[decision_log.md](decision_log.md) に移した
- ここに残しているのは、まだ決まっていない項目だけ
- 「確認先」の意味
  - **教員**：[FB] §14 で先生方への確認事項として挙がっているもの
  - **研究者**：研究責任者が決めるもの
  - **データ**：データ監査の結果を見てから決めるもの

> **作業順序の変更（D-035、2026-09-18）**：CXR 画像監査 → index CXR 選定 → コホート固定 → split 固定 → Train だけで臨床データを再監査 → 候補変数の選定 → 補完・変換 → clinical model。臨床側の質問（Q-B3, Q-B4, Q-B5, Q-B7, Q-B10〜Q-B14, Q-B16, Q-B17）は、Train での再監査のときまで**保留**する。質問に書かれている数値は、全 1,384 例での予備監査のもの（`preliminary_pre_cohort_audit/`）であり、再監査で置き換える。

## Close 済み

| # | 質問 | 結論 | 記録 |
|---|---|---|---|
| Q-A1 | v1 の固定コホートを引き継ぐか | 引き継がない。1,277 例のコホートと split（1,021 / 128 / 128）を新しい固定基準とする | D-019, D-024（2026-09-18 close） |
| Q-B1 | T0 とコホート固定のどちらを優先するか | T0 = `visit_start_datetime`。CXR は T0−2 日〜T0 当日。T0 より後の日付は使わない。T0 当日の CXR は一律に「入院後」とはみなさない（limitation L-1） | D-020〜D-023（2026-09-18 close） |
| Q-B6 | index CXR の選び方 | frontal CXR（T0−2 日〜T0 当日）→ T0 に最も近い撮影日 → その日の中で `AcquisitionTime` が最も早い Series → 同じ時刻なら `SeriesNumber` が最小のもの（tie-breaker のみ）。T0 より後の日付は代わりとしても使わない | D-025, D-026（2026-09-18 close） |
| Q-A6 | 既存のコホートと split を使うか、作り直すか | (a)：ルールからコホートを作り直し、1,277 例と完全に一致すれば既存の split（1,021 / 128 / 128）をそのまま使う。split は作り直さない。一致しなければ後続に進まず、差分を報告する（推測で補正しない） | D-036（2026-09-18 close） |
| Q-A7 | 照合での「完全一致」の範囲 | 患者集合の完全一致に加え、既存 manifest に index CXR の識別子があれば、患者ごとの index CXR（StudyInstanceUID、SeriesInstanceUID、SOPInstanceUID、撮影日のうち manifest にあるもの）も完全一致すること。1 例でも異なれば停止し、差分表を報告する。補正や選び直しはしない | D-037（2026-09-18 close） |
| Q-B9(a) | frontal の判定方法 | `Modality` が CR/DX → `StudyDescription` で胸部検査の候補を特定 → `SeriesDescription` で frontal の AP / PA に限定。`ViewPosition` は主な判定基準にしない | D-028（2026-09-18 close） |

## A. 環境と管理

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-A2 | コードをどこで管理し、どう Colab で実行するか（例：GitHub の private repo を Colab で clone し、`src/` を import する） | [T] は Colab を、[U] は GitHub で提出できる構成を求めている | 研究者 | |
| Q-A3 | 「気合のCOVID19」の下のフォルダ名と版の名前（`00_governance`、`revision_v2_20260814` のままにするか、日付を変えるか） | ルートは D-027 で確定。中の構成は [FB §12, §14] | 教員・研究者 | |
| Q-A4 | **一部解決（2026-09-20）**：DICOM は SSD の `D:\manifest-1628608914773\COVID-19-NY-SBU`。臨床 CSV と定義ファイルは `data/raw/clinical/`。既存コホートの照合候補は Drive「気合のCOVID19」の `00_共通・研究管理 / 03_患者選定 / CXR_最終胸部Xp候補一覧.xlsx`。Task A / 01_前処理に `selected_dicom_1277`、`processed_png_1277` がある。**残る課題**：(a) split manifest の最新版が特定できていない（split の照合には進まない）、(b) Colab からのアクセス方法（D-029：DICOM は複製しない） | D-029、D-036、D-037 | 研究者 | |
| Q-A5 | GitHub を public にするか private にするか。Subject ID を含む予測ファイルをリポジトリに含めてよいか | TCIA の利用規約を確認する必要がある | 研究者 | |

## B. 研究デザイン

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-B2 | 院内死亡の定義と対象集団。**監査の事実**：`last.status` は discharged 1,201 / deceased 183 の 2 値だけで、転院やホスピスは区別されていない。visit 種別は Inpatient 1,025（死亡 181）、**Emergency Room Visit 357（死亡 2）**、Outpatient 2。ER visit と Outpatient visit を「院内死亡」の対象に含めるか。1,277 例のコホートに含まれているか | 資料には記載がない（preliminary_pre_cohort_audit/clinical_audit_report §3。予備監査の値） | 研究者 | |
| Q-B3 | 臨床項目（バイタル、検査値、症状）の測定時点と T0 の関係。**監査の事実**：ファイルに測定時刻はない。MAP > SBP が 20 例、好中球数 > 白血球数が 13 例あり、項目ごとに測定時点が違う可能性がある。TCIA の説明文書で「入院時の値」「最初の値」などの定義を確認する必要がある。**辞書で分かったこと**：体温は入院時の値（`temperature.over38` の説明）。他のバイタルと検査値には時点の記載がない。また、`htn_v`（記録された SBP>140/90）と `ckd_v`（検査での GFR 低下）は、定義に今回の visit の測定値が使われた可能性がある | [FB §8]、L-3 | 研究者・データ | |
| Q-B4 | EPV の基準（**Train の死亡 135 例か、全体の 169 例か**）、数える単位（元の変数か、展開後のパラメータか）、stepwise の実施の有無・方向・基準（P 値、AIC、BIC） | [FB §14]。[FB] の 141 と 176 は v1 の数なので、今回の数に置き換えた | 教員 | |
| Q-B5 | LR、XGB、MLP で同じ最終変数セットを使うか | 資料には記載がない。同じにすれば比較が公平になる [提案] | 研究者 | |
| Q-B7 | 補完の判定方法（Shapiro–Wilk と分布図の併用）の確認。欠測 indicator を追加するか。XGB は補完済みデータを使うか、欠測をそのまま扱うか | [FB §14]。検討自体は Train/Val で行うことが [FB §4] で決まっている | 教員 | |
| Q-B9 | **D-025 の運用上の細部**（(a) は D-028 で close）：(b) `AcquisitionTime` が欠けている Series の扱い、(c) 選んだ Series に複数の画像（Instance）がある場合、どれを使うか | **D-030：推測で決めない**。監査（02-2b、04-5）で該当する症例数を確認する。該当例がある場合に限り、一覧を添えて研究責任者に判断を求める。該当例がなければ「該当なし」と記録して close する | データ → 研究者 | |
| Q-B10 | `visit_concept_name`（Inpatient / ER / Outpatient）を予測変数にするか。入院するかどうかの判断は T0 後に決まる可能性がある | 監査の事実：ER visit は検査値の欠測が多く、在院日数 1 日が中心 | 研究者 | |
| Q-B11 | NA の扱い。**辞書との照合で分かったこと**：カルテ抽出の 21 列は 270 例ですべて NA で、その多くは「カルテ抽出が行われていない」ことを表している可能性が高い。性別の NA は匿名化によるもの。**残る問題**：(a) カルテ抽出がない 270 例の併存症・症状をどう扱うか（欠測 indicator を付けると「抽出されなかったこと」自体が特徴量になる）、(b) 一部の列だけが NA の約 270 例、(c) Yes/NA だけの列（`kidney_transplant`、`kidney_replacement_therapy`）の NA が「なし」か「不明」か | FB-06 の欠測 indicator の設計に関わる | 研究者 | |
| Q-B12 | 単位（特に CRP：mg/dL か mg/L か。辞書にも記載なし）と、検出限界・上限に値が集中している変数の扱い（troponin 0.01 が 766 例、D-dimer 150 が 93 例、CRP 0.1 が 22 例、eGFR 120 が 149 例） | 監査の事実 | 研究者・データ | |
| Q-B13 | **一部解決**：`antibiotics_use_v` は受診前（辞書）、`therapeutic.exnox/heparin` は投与（treatment）。**残り**：`kidney_replacement_therapy`（慢性透析か入院中か）と `Other.anticoagulation.therapy`（常用薬か入院中の治療か）の時点 | 辞書に時点の記載がない | 研究者・データ | |
| Q-B16 | 一次候補（A）は 22 変数・24 パラメータで、EPV の目安（Train の死亡 135 例 → 約 13）を超えている。絞り込みの方法：stepwise（FB-04）のほかに、臨床的に合成する案 [提案] がある（例：併存症の数をまとめる（4C 型）、好中球/リンパ球比（NLR）、クレアチニンか eGFR のどちらか一方にする）。辞書の定義から分かる重なり（COPD の定義に喫煙歴を含む、CKD の定義に GFR 低下を含む）も考慮する | FB-03, FB-04 | 教員・研究者 | |
| Q-B14 | 生理学的にあり得ない値の範囲（preliminary_pre_cohort_audit/clinical_audit_report §8 の暫定案）を承認するか。範囲外の値（心拍数 < 20 が 2 例、呼吸数 > 80 が 2 例）と、項目間の矛盾（MAP > SBP が 20 例、好中球数 > 白血球数が 13 例）をどう扱うか | [FB §6]。原データの確認が必要 | 研究者 | |
| Q-B17 | **保留（D-035）**。stepwise の前の shortlist 案（D-034、[preliminary_pre_cohort_audit/candidate_shortlist_prestepwise_draft.csv](preliminary_pre_cohort_audit/candidate_shortlist_prestepwise_draft.csv)。予備監査の記録）を承認するか。別案（COPD を加える、troponin と D-dimer を入れ替える、BUN の代わりにクレアチニン、併存症の数、NLR）を採るか。欠測を判断に使った 6 変数（BUN、CRP、リンパ球数、D-dimer、troponin、BMI）は Train で欠測率を確認し直してから確定する | 欠測率は全データの記述統計しかない（split manifest がまだない） | 研究者・教員 | |
| Q-B8 | 論文での T0 の呼び方（「入院時」か「受診開始時（visit start）」か） | [T] は「入院時」と書いているが、T0 は `visit_start_datetime` である（L-2） | 研究者 | |

## C. モデル

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-C1 | CXR の学習率の候補、scheduler、early-stopping の patience。warm-up や凍結する層の範囲も比較条件に含めるか | [FB §7, §14]（1×10⁻⁴ を含むことは確定） | 教員・研究者 | |
| Q-C2 | LR の class weight を balanced にするか、重みを指定するか | [FB §14] | 教員 | |
| Q-C3 | Clinical model の調整を Val で行うか、Train 内 CV で行うか | [提案]。Val の死亡が 17 例しかなく、Val だけでの選択はばらつきが大きい | 研究者 | |
| Q-C4 | Late Fusion の仕様：統合する臨床モデル（LR、XGB、MLP のどれか）、統合対象（確率か logit か）、統合方法（単純平均、重み付き平均、meta-model）、重みの決め方 | [FB §9, §14] | 教員・研究者 | |
| Q-C5 | Early Fusion を副解析として残すか | [T] はどちらでもよいとし、[FB] は Late Fusion に変更するとしている | 研究者 | |
| Q-C6 | 学習の seed の数、最良の seed を選ぶか ensemble にするか | 資料には記載がない | 研究者 | |

## D. 評価

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-D1 | DeLong 検定で Task B の代表とする臨床モデルはどれか（または 3 つすべてと比べるか）。Holm 補正の family に何を含めるか。CXR vs Clinical の比較も含めるか | [T] は「Task C と Task A/B の比較」、[FB §10] は family を明示するよう求めている | 研究者 | |
| Q-D2 | Bootstrap の反復回数、層別するかどうか。DeLong の CI も併記するか | [T] は「Bootstrap 法等」とだけ書いている | 研究者 | |
| Q-D3 | 分類閾値のルール（Val での Youden 指数か、目標感度か）と、学習曲線の Accuracy に使う閾値 | [FB §11] は「閾値は Val で決める」とだけ書いている | 研究者 | |
| Q-D4 | Grad-CAM の「成功例」と「失敗例」の定義（TP/TN と FP/FN か）、使うデータ（凍結後の Test か）、例の数 | [T] | 研究者 | |
| Q-D5 | Feature Importance の手法（LR の標準化係数、XGB の gain または SHAP、MLP の permutation importance など）と、計算に使うデータ | [T] | 研究者 | |
| Q-D6 | calibration、Brier、PR-AUC を全モデルで報告するか（[FB] では LR の class weight 比較に限って挙げられている）。TRIPOD+AI に沿うか | [提案] | 研究者 | |
| Q-D7 | L-1 に対する感度分析：index CXR が T0 当日の患者を除き、T0−2 日〜T0−1 日の患者に限って Test 性能を再評価するか | [提案]。評価のみに使い、モデル選択には使わない。該当する患者の数によっては検出力が不足する | 研究者 | |

## E. 公開（GitHub 提出）

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-E1 | DICOM instance UID（StudyInstanceUID / SeriesInstanceUID / SOPInstanceUID）を公開リポジトリに含めてよいか | TCIA COVID-19-NY-SBU の再配布条件を本作業環境から確認できていない（**未確認**）。現在は D-065 により公開版から除去し、正本の SHA256 と再生成スクリプトのみを残している。含めてよい場合は `PUBLISH_UIDS = True` で復帰できる | 研究者・データ提供元の規約 | |
| Q-E2 | `gradcam_notes.md` の「総括（定量）」が分母を 15 ではなく **16** と記載している点をどう扱うか（「16 例平均」「8 / 16」「0 / 16」「群別 central 平均」） | 実際の症例数は 15（TP4・FP4・TN4・FN3）。正本をそのまま収録し、**数値は修正していない**。集計の再計算は Grad-CAM の再実行を伴うため、研究者の指示なしには行わない | 研究者 | **2026-09-22 修正済み（CL-006）**。gradcam_selection.csv から再集計し、分子が一致することを確認したうえで分母を修正。3 つの平均は元から 16 でも 15 でもなく「算出可能 10 例」の平均だったため、母集団を明示する記載に改めた。件数は 8 / 15、0 / 15。Grad-CAM の再生成・Test 再評価は行っていない |
| Q-E3 | 予測確率がほぼ 0 の症例（TN 4 例・FN 1 例）で CAM が全面ゼロとなり領域指標が `nan` になる件を、limitation として記載するか、正規化方法を変えて再計算するか | Grad-CAM は勾配重み付けのため、確率が数値的に 0 に飽和した症例では CAM が定義できない。現在は正本のまま収録し、README §8 に既知の未解決点として明記 | 研究者 | |
| Q-E4 | `gradcam_notes.md` の「## 総括（読影）」が生成時のプレースホルダのままで、D-063 で用意した読影総括（`scripts/19`）が**まだ追記されていない**。追記するか | Drive 上の正本の更新日時は生成時刻（2026-09-21 23:20:01）のままで、`### 読影総括（研究者記入）` の見出しが存在しない。追記は Drive 正本への書き込みになるため、指示なしには実行しない | 研究者 | **2026-09-22 追記済み（D-063 / D-068）**。`scripts/19_taskA_append_gradcam_notes.py` により `### 読影総括（研究者記入）` として append。既存の症例別所見・定量情報・実行記録は削除していない。追記前 SHA256 と追記日時をファイル末尾に記録 |
| Q-E5 | LICENSE を何にするか（未定。README は "to be determined" のまま） | 公開範囲（public / private）と併せて決める必要がある | 研究者 | |
| Q-E6 | `gradcam_selection.csv` の `delta_days_from_T0` が 15 例すべて空で、`gradcam_notes.md` の症例別記載が「T0 との差 None 日」になっている | Test 予測 CSV 側には正しい値（−1 / −2 / 0）があり、scripts/18 が選択表へ引き継いでいない**metadata propagation issue**。**推測で値を埋めない**。初回 commit の停止理由にはせず、既知の問題として記録を維持する（CL-006） | 研究者 | |

## F. Task B（臨床表データモデル）

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-F1 | EPV の目安の分母は、one-hot 展開前の元変数数か、展開後の推定パラメータ数か | 教員フィードバック §1 で提起され未回答。現在は**推定パラメータ数**（>= 13）で暫定固定している。過学習を規定するのは推定パラメータ数であるため | 教員 | |
| Q-F2 | stepwise 法の方向（前進・後退・双方向）と選択基準（P 値・AIC・BIC） | 教員フィードバック §3 で提起され未回答。現在は **backward + AIC** で暫定固定（候補が臨床的根拠で事前に絞られているため full model から始めるのが自然、P 値基準は多重性で不安定、BIC は n=1,021 で罰則が強い） | 教員 | |
| Q-F3 | 本コホートの「院内死亡」は ER 帰宅例に対してほぼ定義されない（Training で ER 262 例中 死亡 1 例、在院日数 中央値 1 日）。予測対象集団をどう定義・記述するか | 教員フィードバック §8 が提起した「予測時点と対象患者の再定義」と同じ論点。D-069 で主解析 1,277 例＋入院例 955 例の感度分析として対応しているが、論文での集団の記述は別途決める必要がある | 教員・研究者 | |
| Q-F4 | 性別の欠測（Training 19 例中 18 例が死亡）の機序 | data dictionary に de-identification で一部削除される旨の記載があるが、**本欠測がその処理に起因することは未確認**。非臨床的な欠測機序を反映している可能性がある（D-072）。Task A・C で臨床変数を使う場合にも該当 | 研究者・データ提供元 | |
| Q-F5 | CRP の単位が data dictionary に明記されていない | Training の中央値 8.4、最大 51。mg/dL とすれば臨床的に妥当な範囲だが確認が取れていない。単調変換のためモデル性能には影響しないが、係数の臨床的解釈には影響する | 研究者・データ提供元 | |
| Q-F6 | SpO2 に酸素投与の有無の情報がなく、同じ値でも臨床的意味が異なる | 4C・NEWS2 は室内気での SpO2 を想定している。本データでは補正できないため limitation として記載する方針 | 教員 | |

## G. リポジトリ全体監査（2026-09-23）

| # | 質問 | 背景 | 確認先 | 回答 |
|---|---|---|---|---|
| Q-G1 | `results/taskA/preprocessing/cxr_dicom_audit_summary.json` の `duplicate_candidates.examples` に、実在の TCIA **SeriesInstanceUID が 10 件**（5 例 × 2 series、SeriesNumber・AcquisitionTime を伴う）残っている。伏字化するか、このまま維持するか | 2026-09-23 のリポジトリ全体監査で検出。**従来の UID 検査は `.csv` のみを対象としており、`.json` を走査していなかったため見逃していた**。Q-E1 / D-065 で決めた「DICOM UID は現時点では除外状態を維持する」方針に抵触する。一方、修正は Task A official artifact の変更にあたり、当該ファイルは初回コミット `c5afc93` で既に push 済みのため履歴にも残る。repository は PRIVATE で外部公開はされていない | 研究者・データ提供元（TCIA 再配布条件） | **2026-09-23 保留（研究者判断）**。今回の documentation コミットでは対応せず、別件として保留する。Task A official artifact は変更していない。対応を決める際は、伏字化・履歴の書き換え・TCIA 再配布条件の確認の 3 点を併せて検討する |
| Q-G2 | 安全監査の UID 検査を `.json` / `.jsonl` / `.md` まで広げた恒久的なチェックを、`scripts/31` / `scripts/39` の prepare_commit 監査に組み込むか | 現在の prepare_commit スクリプトは secrets・local path・checkpoint・大容量ファイル・患者画像は全テキスト形式で検査しているが、UID 検査は独立した検査項目として実装されていない。Q-G1 は監査スクリプトの外側（手動監査）で初めて検出された | 研究者 | |
