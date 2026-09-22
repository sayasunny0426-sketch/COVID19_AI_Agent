# Grad-CAM 所見メモ（Task A primary model）

- モデル：lr3e-4_aug_b/seed42（checkpoint `best.pt`, sha256 `26d6ed95844e5ff6…`）
- 閾値：0.008508（final_selection.json の Youden。Test では再計算していない）
- 対象層：ResNet18 `layer4[-1]`
- 指標の定義：`padding` は正方形化のためのゼロ埋め領域、`border` は写野の外周 10% の帯（コリメーション縁・L/R マーカー・PORTABLE の文字が入りやすい位置）、`central` は写野中央 50%×50%（正面胸部では肺野・縦隔にあたる）に落ちた CAM 質量の割合。

- 症例数：15 例（TP 14 例中 4、FP 34 例中 4、TN 77 例中 4、FN 3 例中 3）

> **FN が Test set に 3 例しか存在しなかったため、全 3 例を採用し、合計 15 例とした。16 例に合わせるために他群から追加症例は選んでいない。** 4×4 パネルは 1 セルを空欄のままとしている。

## 症例ごとの所見

### TP（4 例 / Test 内 14 例）
- `A701587`（rank 1, p=0.963, true=1, pred=1, T0 との差 None 日）：CAM 質量は padding 0.06 / border 0.16 / central 0.39、重心 (x=0.53, y=0.46)、ピーク (x=0.64, y=0.36)。　**ゼロ埋め領域への注目が 5% 超**
- `A387558`（rank 2, p=0.963, true=1, pred=1, T0 との差 None 日）：CAM 質量は padding 0.11 / border 0.14 / central 0.38、重心 (x=0.58, y=0.46)、ピーク (x=0.50, y=0.64)。　**ゼロ埋め領域への注目が 5% 超**
- `A003797`（rank 3, p=0.913, true=1, pred=1, T0 との差 None 日）：CAM 質量は padding 0.07 / border 0.14 / central 0.47、重心 (x=0.61, y=0.37)、ピーク (x=0.78, y=0.21)。　**ゼロ埋め領域への注目が 5% 超**
- `A164683`（rank 4, p=0.834, true=1, pred=1, T0 との差 None 日）：CAM 質量は padding 0.02 / border 0.14 / central 0.44、重心 (x=0.37, y=0.55)、ピーク (x=0.21, y=0.64)。

### FP（4 例 / Test 内 34 例）
- `A293330`（rank 1, p=0.997, true=0, pred=1, T0 との差 None 日）：CAM 質量は padding 0.04 / border 0.11 / central 0.50、重心 (x=0.51, y=0.42)、ピーク (x=0.50, y=0.50)。
- `A711426`（rank 2, p=0.812, true=0, pred=1, T0 との差 None 日）：CAM 質量は padding 0.09 / border 0.12 / central 0.40、重心 (x=0.51, y=0.35)、ピーク (x=0.36, y=0.35)。　**ゼロ埋め領域への注目が 5% 超**
- `A476289`（rank 3, p=0.626, true=0, pred=1, T0 との差 None 日）：CAM 質量は padding 0.11 / border 0.24 / central 0.23、重心 (x=0.70, y=0.28)、ピーク (x=0.79, y=0.21)。　**ゼロ埋め領域への注目が 5% 超**、中央部への注目が乏しい
- `A071731`（rank 4, p=0.560, true=0, pred=1, T0 との差 None 日）：CAM 質量は padding 0.12 / border 0.21 / central 0.27、重心 (x=0.76, y=0.32)、ピーク (x=0.79, y=0.21)。　**ゼロ埋め領域への注目が 5% 超**、中央部への注目が乏しい

### TN（4 例 / Test 内 77 例）
- `A769617`（rank 1, p=0.000, true=0, pred=0, T0 との差 None 日）：CAM 質量は padding nan / border nan / central nan、重心 (x=nan, y=nan)、ピーク (x=nan, y=nan)。
- `A905325`（rank 2, p=0.000, true=0, pred=0, T0 との差 None 日）：CAM 質量は padding nan / border nan / central nan、重心 (x=nan, y=nan)、ピーク (x=nan, y=nan)。
- `A132584`（rank 3, p=0.000, true=0, pred=0, T0 との差 None 日）：CAM 質量は padding nan / border nan / central nan、重心 (x=nan, y=nan)、ピーク (x=nan, y=nan)。
- `A759584`（rank 4, p=0.000, true=0, pred=0, T0 との差 None 日）：CAM 質量は padding nan / border nan / central nan、重心 (x=nan, y=nan)、ピーク (x=nan, y=nan)。

### FN（3 例 / Test 内 3 例）
FN が Test set に 3 例しか存在しなかったため、全 3 例を採用し、合計 15 例とした。16 例に合わせるために他群から追加症例は選んでいない。
- `A766483`（rank 1, p=0.000, true=1, pred=0, T0 との差 None 日）：CAM 質量は padding nan / border nan / central nan、重心 (x=nan, y=nan)、ピーク (x=nan, y=nan)。
- `A292777`（rank 2, p=0.004, true=1, pred=0, T0 との差 None 日）：CAM 質量は padding 0.53 / border 0.33 / central 0.00、重心 (x=0.36, y=0.37)、ピーク (x=0.00, y=0.21)。　**ゼロ埋め領域への注目が 5% 超**、中央部への注目が乏しい
- `A980322`（rank 3, p=0.005, true=1, pred=0, T0 との差 None 日）：CAM 質量は padding 0.18 / border 0.25 / central 0.08、重心 (x=0.83, y=0.56)、ピーク (x=0.79, y=0.64)。　**ゼロ埋め領域への注目が 5% 超**、中央部への注目が乏しい

## 総括（定量）

- **領域指標が算出できた 10 例の平均**（症例 15 例のうち、予測確率が数値的に 0 に飽和し CAM が全面ゼロとなった 5 例は算出不能）：padding 0.133、border 0.185、central 0.318
- ゼロ埋め領域への注目が 5% を超えた症例：8 / 15（算出不能の 5 例は「超えていない」側に数えられており、判定できない）
- 外周帯への注目が 35% を超えた症例：0 / 15（同上）
- 群別の central 平均（かっこ内は算出できた症例数）：TP 0.420（4/4 例）、FP 0.353（4/4 例）、TN 算出不能（0/4 例）、FN 0.043（2/3 例）

## 総括（読影）

上の数値は「どこに CAM 質量が落ちたか」を機械的に測ったもので、解剖学的な妥当性の判断ではない。
肺野・心陰影・下肺野・末梢陰影のどこを見ているか、焼き込み文字やマーカーに引っ張られていないかは、
`panels/gradcam_panel_4x4.png` と `individual/` の画像を確認したうえで研究責任者が記入する。

### 読影総括（研究者記入）

本所見は、TP 4例・FP 4例・TN 4例・FN 3例の計15例からなる4×4パネル（FNはTest setに3例しか存在しなかったため全3例を採用し、他群からの補充は行っていない。1セルは空欄）に基づく。

事前に規定したTest症例15例（TP 4例、FP 4例、TN 4例、FN 3例）についてGrad-CAMを確認した。

TPでは、肺門周囲から下肺野を含む胸郭内に比較的強いattentionが認められ、モデルが高リスク症例の判定において臨床的に妥当な胸部領域を参照している可能性が示唆された。

FPでも胸郭内に強いattentionを認めたが、その分布は必ずしも死亡リスクに関連する肺病変に特異的とは限らなかった。

TNでは明瞭な局所的attentionは乏しく、全体に弱い反応を示す症例が多かった。

一方、FNの一部では肺野中央よりも画像辺縁付近に強いattentionが認められ、モデルが病変とは直接関係しない周辺構造や高コントラスト領域を利用した可能性が考えられた。

以上より、本モデルは特にTP症例では妥当な胸郭内領域に注意を向ける傾向を示した一方、誤分類症例の一部では非病変領域へのattentionも認められた。

なお、Grad-CAMは定性的な説明手法であり、強調領域が予測の因果的根拠であることを示すものではない。また、本所見は4×4パネルおよび生成されたGrad-CAM画像に基づく定性的評価であり、特定の画像診断所見との厳密な対応を示すものではない。

（追記日時 2026-09-22T15:09:48+09:00／追記前 gradcam_notes.md SHA256 `7c8aa87ffce6a03c4981eb9a78dae35d02ec7aef7acbd3c82dd595bfea805722`／追記のみで既存記載は削除していない）
