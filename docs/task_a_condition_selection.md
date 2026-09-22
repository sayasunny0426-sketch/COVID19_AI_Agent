# Task A 学習条件の選択（Stage 1 + Stage 2、全 30 run）

- 決定日：2026-09-21　記録：decision_log D-052 / D-053 / D-054 / D-055
- **Test set は使用していない。** 選択は Train（学習）と Validation（比較）のみで行った
- 機械可読の記録：[task_a_condition_comparison.csv](task_a_condition_comparison.csv)、
  [task_a_condition_comparison.decision.json](task_a_condition_comparison.decision.json)、
  run 単位の再生成スクリプト [scripts/13_taskA_condition_comparison.py](../scripts/13_taskA_condition_comparison.py)

---

## 1. 実行した全 30 run

| Stage | 条件数 | seed | run 数 | 内容 |
|---|---|---|---|---|
| Stage 1 | 8 | 42 / 43 / 44 | **24** | 学習率 4（1e-5 / 3e-5 / 1e-4 / 3e-4）× augmentation 2（aug_a 弱 / aug_b 中）。全層 fine-tuning、OneCycleLR |
| Stage 2 | 2 | 42 / 43 / 44 | **6** | Stage 1 最良条件から**解凍方法だけ**を変更。param group ごとの学習率（凍結中 0 → 解凍後 1 epoch で立ち上げ → cosine 減衰） |

Stage 1 の実測：総計算 1.45 時間、平均 3.6 分/run。

---

## 2. 比較結果（3 seed 平均、Validation のみ）

| Stage | condition | mean Val AUROC | SD | mean Val AUPRC | mean Val loss | best epochs |
|---|---|---|---|---|---|---|
| **Stage 1** | **lr3e-4_aug_b** | **0.872637** | **0.004569** | 0.537411 | 1.700158 | 5, 9, 14 |
| Stage 2 | lr3e-4_aug_b_warmup1 | 0.869811 | 0.008043 | 0.556175 | 1.712621 | 2, 3, 9 |
| Stage 1 | lr1e-4_aug_a | 0.866985 | 0.024952 | – | – | – |
| Stage 1 | lr3e-5_aug_b | 0.858506 | 0.013625 | – | – | – |
| Stage 2 | lr3e-4_aug_b_staged | 0.856209 | 0.008112 | 0.487404 | 1.653959 | 2, 3, 4 |
| Stage 1 | lr3e-5_aug_a | 0.854619 | 0.008632 | – | – | – |
| Stage 1 | lr3e-4_aug_a | 0.850380 | 0.005956 | – | – | – |
| Stage 1 | lr1e-4_aug_b | 0.848790 | 0.020880 | – | – | – |
| Stage 1 | lr1e-5_aug_b | 0.844727 | 0.006425 | – | – | – |
| Stage 1 | lr1e-5_aug_a | 0.839958 | 0.007842 | – | – | – |

（Stage 1 の AUPRC・loss・best epoch は最良条件についてのみ報告を受けている。run 成果物からの完全な再生成は §5 のスクリプトで行える）

---

## 3. 事前固定した選択規則（D-054、Stage 2 実行前に確定）

1. mean Validation AUROC の改善が Stage 1 baseline に対して **0.005 未満**の場合
   → 事前に固定した **practical tolerance / parsimony margin** として扱い、**より単純な Stage 1 条件を採用**する
2. **0.005 以上**改善した場合
   → mean Validation AUROC が最大の条件を候補とし、近接する条件については seed 間 SD、Val AUPRC、Val loss、学習曲線、best epoch の分布も確認して決める

結果を見てから規則を変えない。

---

## 4. 最終選択と理由

**採用：`lr3e-4_aug_b`（Stage 1 baseline）**
学習率 3e-4、augmentation aug_b（平行移動 ±5%、回転 ±7°、scale 0.95–1.05、明るさ・コントラスト 0.90–1.10）、全層 fine-tuning、OneCycleLR（warm-up 1 epoch → cosine）、AdamW、weight decay 1e-4、batch 32、最大 30 epoch、early stopping（Val AUROC、patience 8、min_delta 0.005）、`pos_weight` 6.563。

**理由**
1. **規則の適用**：Stage 2 の最良（warmup1）は baseline に対して **−0.002826** であり、改善していない。規則 1 に該当するため、より単純な Stage 1 条件を採用する。
2. **staged は明確に劣る**：−0.016428。AUPRC も 0.487 と最も低い。段階的解凍は、この規模のデータでは有効でなかった。
3. **副次指標でも覆らない**：warmup1 は AUPRC が baseline より高い（0.556 対 0.537）が、主評価は mean Val AUROC と事前に決めている。AUROC が改善していない以上、副次指標を理由に採用すると、規則を後から変えることになる。
4. **安定性**：baseline は seed 間 SD が 0.004569 と全条件で最小であり、Stage 2 の 2 条件（0.008）より小さい。
5. **早期収束の懸念は生じていない**：baseline の best epoch は 5 / 9 / 14 で、十分に学習が進んだ段階で最良になっている。Stage 2 の warmup1（2 / 3 / 9）と staged（2 / 3 / 4）はより早い epoch で頭打ちになっており、解凍を遅らせることが有利に働いていない。

**採用しなかった条件**：`lr3e-4_aug_b_warmup1`、`lr3e-4_aug_b_staged`（上記 1〜5 のとおり）。Stage 1 の他の 7 条件は mean Val AUROC が baseline を下回る。

---

## 5. 再利用・再現の方法

run 成果物（`04_Training/AIagent_taskA_runs/<condition>/seed<NN>/`）が参照できる環境では、次のコマンドで本表と決定を再生成できる。

```bash
python scripts/13_taskA_condition_comparison.py \
    --runs-root "<.../AIagent_taskA_runs>" \
    --out docs/task_a_condition_comparison.csv
```

出力：
- `task_a_all_runs.csv`：30 run すべての run 単位の記録（stage、condition、seed、best epoch、Val AUROC / AUPRC / loss、epochs_run、early stopping、unfreeze_schedule、lr、augment）
- `task_a_condition_comparison.csv`：条件単位の集計
- `task_a_condition_comparison.decision.json`：選択規則の適用結果

スクリプトは選択規則（margin 0.005）を実装しており、同じ run から同じ決定が導かれることを確認できる。

---

## 6. 次の段階（未実施）

1. 最終モデルの構成（単一 seed か 3 seed ensemble か）の決定 → [task_a_final_model_options.md](task_a_final_model_options.md)
2. 分類閾値の決定（Validation のみ）
3. 凍結記録 `final_selection.json` の作成
4. **Test 評価（1 回のみ）** ← ここまで Test は未使用
