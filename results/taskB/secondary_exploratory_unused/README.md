# results/taskB/secondary_exploratory_unused

**この資材は Task B の解析結果ではありません。いかなる結果の算出にも使われていません。**

## 位置づけ

固定コホート 1,277 例には、ED で評価され帰宅した Emergency Room Visit 例が含まれます
（Training では 262 例中 死亡 1 例、在院日数 中央値 1 日）。一時期、これらを除いた
`visit_concept_name == "Inpatient Visit"` の 955 例に限定する感度分析を実施する方針を
検討し、その**対象集団を定義するための集計表のみ**を作成しました。

**2026-09-23、研究者の判断により、この inpatient-only 感度分析は実施しないことが決定されました**
（decision_log D-081）。理由は Task A・Task B・Task C の比較可能性を優先し、
主解析を固定 1,277 例で統一するためです。

したがって本フォルダの内容は、

- 主解析の結果ではない
- 副次解析の結果でもない（解析自体を実施していない）
- 将来の追加解析候補として保存してある集団定義の集計表にすぎない

という位置づけです。**Primary の成果物と混在させないでください。**

## 収録ファイル

| ファイル | 内容 |
|---|---|
| `inpatient_sensitivity_flow.csv` | 臨床ファイル 1,384 例 → 固定コホート 1,277 例 → Inpatient Visit 955 例 の内訳 |
| `visit_type_composition.csv` | split 別・受診形態別の人数と死亡数 |

## 再生成方法

`scripts/23_taskB_missingness_mechanism.py --cohort-flow` および
`scripts/24_taskB_missing_indicator_decision.py --cohort-flow`
（フラグは既定オフ。このフラグを立てると Test のメタデータを読みます）

## 混同しやすい点（重要）

主解析の**欠測指示変数の採否基準 C2・C3 は、入院例の層内で関連が維持されるかを検証**します
（`results/taskB/preprocessing/missing_indicator_decision.csv`）。これは層別の**診断的検証**であり、
ここで中止した「入院例のみでモデルを再学習・再評価する感度分析」とは別物です。
C2・C3 は主解析の一部として維持されています。
