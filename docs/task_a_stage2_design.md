# Task A Stage 2 設計案（warm-up / staged unfreezing）

- 作成日：2026-09-21　版：v0.1（**設計案。実行コードはまだ変更していない。未承認**）
- 位置づけ：Stage 1（24 run）完了後の追加条件。**Test set は一切使わない**
- 選択の根拠：主評価は 3 seed 平均の Validation AUROC。副次として Val AUPRC、Val loss、seed 間ばらつき、学習曲線、best epoch の分布

---

## 1. Stage 1 の結果（研究者提供、3 seed 平均 Val AUROC）

| 順位 | condition | mean | SD |
|---|---|---|---|
| 1 | **lr3e-4_aug_b** | **0.872637** | 0.004569 |
| 2 | lr1e-4_aug_a | 0.866985 | 0.024952 |
| 3 | lr3e-5_aug_b | 0.858506 | 0.013625 |
| 4 | lr3e-5_aug_a | 0.854619 | 0.008632 |
| 5 | lr3e-4_aug_a | 0.850380 | 0.005956 |
| 6 | lr1e-4_aug_b | 0.848790 | 0.020880 |
| 7 | lr1e-5_aug_b | 0.844727 | 0.006425 |
| 8 | lr1e-5_aug_a | 0.839958 | 0.007842 |

読み取れること（選択はしない）：
- 最良は lr3e-4_aug_b で、**seed 間のばらつきも最小**（SD 0.0046）。2 位の lr1e-4_aug_a は平均は近いが SD が 0.025 と 5 倍以上大きく、単一 seed での比較では順位が逆転しうる。
- 学習率の主効果は一貫しておらず（3e-4 が 1 位と 5 位、1e-4 が 2 位と 6 位）、augmentation との組み合わせで決まっている。Val の死亡が 17 例であることを踏まえると、**平均の差 0.006（1 位と 2 位）は実質的な差とみなせない**。

---

## 2. Stage 2 の 2 条件

**共通**：Stage 1 の最良条件（lr3e-4_aug_b）から、**解凍の仕方だけ**を変える。学習率 3e-4、augmentation aug_b、batch 32、最大 30 epoch、early stopping（Val AUROC、patience 8、min_delta 0.005）、seed 42/43/44、その他すべて Stage 1 と同一。

### 条件 S2-A：`lr3e-4_aug_b_warmup1`（head-only 1 epoch → 全層）
| 項目 | 内容 |
|---|---|
| epoch 1 | 分類 head（`fc`）のみ学習。backbone は凍結 |
| epoch 2 以降 | 全層を学習 |
| backbone の学習率 | 解凍時に 0 から 1 epoch かけて 3e-4 まで立ち上げ、その後 cosine で減衰 |
| head の学習率 | Stage 1 と同じ軌跡（1 epoch の warm-up → cosine 減衰） |

**根拠**：ImageNet 事前学習の重みに対して、初期化されたばかりの head から来る大きな勾配が最初に流れ込むと、事前学習の特徴が壊れやすい。head を先に学習して整合を取ってから backbone を動かす。[FB §7] が「epoch 1 が best になった原因の候補」として全層 fine-tuning による早期 overfitting と warm-up の検討を挙げていることに直接対応する。

### 条件 S2-B：`lr3e-4_aug_b_staged`（段階的解凍）
| epoch | 学習対象 |
|---|---|
| 1 | `fc` のみ |
| 2–3 | `fc` + `layer4` |
| 4 以降 | 全層 |

backbone の各群は、解凍された時点から 1 epoch かけて学習率を 0 → 3e-4 に立ち上げ、その後 cosine で減衰する。

**根拠**：CNN の浅い層は一般的なエッジ・テクスチャを表し、深い層ほどタスク固有になる。学習例が 1,021 例（死亡 135 例）と少ないため、浅い層まで一度に動かすと過学習しやすい。深い層から順に開放して、更新する自由度を段階的に増やす。S2-A が「解凍のタイミング」だけを変えるのに対し、S2-B は「解凍の範囲」も段階化する点が異なる。

---

## 3. 実装上、必ず直す必要がある点（重要）

現行実装は**全パラメータに単一の OneCycleLR** を適用し、`pct_start = warmup_epochs / max_epochs = 1/30` である。つまり**学習率のピークが epoch 1 の終わり**に来る。ここで backbone を解凍すると、**解凍直後の backbone に最大学習率 3e-4 がかかり、warm-up の意図と正反対**になる。

したがって Stage 2 では次の変更が必要になる（承認後に実装）。

1. optimizer を **param group に分ける**（`head` と backbone の各群）。
2. 群ごとの学習率係数を `LambdaLR` で定義する：
   - 凍結中：係数 0
   - 解凍直後の 1 epoch：0 → 1 に線形に立ち上げ
   - その後：残り epoch にわたって cosine で 0 まで減衰
3. `freeze_backbone_epochs` に加えて、群ごとの解凍 epoch を設定として持つ（`unfreeze_schedule`）。
4. 再開時の設定一致チェックに `unfreeze_schedule` を追加する。

この変更は Stage 2 の条件にのみ影響し、**Stage 1 の run と成果物には一切触れない**（Stage 1 の条件は `unfreeze_schedule` なしのまま再現できる）。

---

## 4. 公平な比較のために固定すること

| 項目 | 内容 |
|---|---|
| 比較対象 | Stage 2 の 2 条件 × 3 seed（**6 run**） vs **Stage 1 の lr3e-4_aug_b の既存 3 run**（再学習しない。データ・split・seed・epoch 上限・early stopping がすべて同一のため再利用できる） |
| 主評価 | 3 seed 平均の Validation AUROC |
| 副次 | Val AUPRC、Val loss、seed 間 SD、学習曲線、best epoch の分布 |
| 改善が 0.005 未満 | **事前に固定した practical tolerance / parsimony margin** として扱い、より単純な条件（Stage 1 の `lr3e-4_aug_b`、warm-up なし）を採用する。「統計的に区別できない」という表現は使わない（研究者指示 2026-09-21、D-054） |
| 改善が 0.005 以上 | mean Validation AUROC が最大の条件を候補とする。近接する条件については、seed 間 SD、Val AUPRC、Val loss、学習曲線、best epoch の分布も確認したうえで決める |
| 禁止事項 | Test の使用、seed の追加・入れ替え、条件の後付け追加、Stage 1 run の再学習 |

**この規則は実行前に固定する。** 結果を見てから規則を変えない。

---

## 5. 成果物と記録

- run ごと：Stage 1 と同じ（config.json / history.csv / best.pt / last.pt / val_predictions.csv / DONE.json）。保存先は `04_Training/AIagent_taskA_runs/<condition>/seed<NN>/`
- Stage 2 の比較表（`stage2_comparison.csv`）：条件 × seed の best epoch、Val AUROC / AUPRC / loss、early stopping、時間
- 最終条件の決定は decision log に記録する（採用条件、平均と SD、同点規則の適用有無、却下した条件とその理由）
- 学習曲線：Stage 1 の best 条件と Stage 2 の 2 条件を重ねた図

---

## 6. 計算量

6 run（2 条件 × 3 seed）。Stage 1 の実測時間（`stage1_report.csv`）から見積もる。Stage 1 が 24 run で N 時間なら、Stage 2 は概ね N/4 時間。early stopping が働けば短くなる。

---

## 7. 検討したが採らなかった案

| 案 | 採らない理由 |
|---|---|
| 2 位の lr1e-4_aug_a にも warm-up を適用する | 学習率・augmentation・解凍方法の 3 つが同時に変わり、どの要素が効いたか分からなくなる。条件数が増えると Val（死亡 17 例）への過適合も増える |
| backbone に低い学習率を割り当てる（discriminative LR） | 解凍方法の検証という Stage 2 の目的から外れ、学習率の再探索になる。Stage 1 で学習率は既に 4 条件検討済み |
| Stage 2 を 1 条件に絞る | warm-up（タイミング）と staged unfreezing（範囲）は別の仮説であり、片方だけでは [FB §7] の「凍結層の範囲」の検討にならない |
| 3 条件以上に増やす | Val の死亡 17 例で比較できる条件数には限りがある。Stage 1 の 8 条件に加えて 2 条件までが妥当と判断 |

---

## 8. 未確定事項

| # | 内容 |
|---|---|
| Q1 | 本設計案の承認（特に §4 の同点規則 0.005 と、§3 の param group 化の実装） |
| Q2 | Stage 1 の実測時間（`stage1_report.csv`）。Stage 2 の所要時間見積もりに必要 |
| Q3 | Stage 1 の各 run の history（best epoch の分布）。epoch 1 が best になる現象が Stage 1 でも起きているかを確認し、Stage 2 の効果判定の材料にする |
