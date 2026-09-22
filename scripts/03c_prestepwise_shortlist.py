"""Pre-stepwise shortlist proposal from the 22 tier-A raw variables (draft, not approved).

PRELIMINARY (D-035): made before the CXR-based cohort and split were fixed; kept for the
record only and on hold. Not to be used for final variable selection.

Rules followed (researcher instruction 2026-09-18):
  - no outcome association, no Validation/Test information, no stepwise / imputation /
    scaling / transformation / model fitting;
  - priority: clinical validity for COVID-19 in-hospital mortality > SR/MA and established
    risk scores > availability near T0 > low redundancy > acceptable missingness >
    low dependence on specific lab orders;
  - no new derived variables (comorbidity count, NLR, recoding) -- listed only as alternatives.

Caveat: the missing rates available now are descriptive on all 1,384 patients (the split
manifest is not yet available). Rows where missingness contributed to the decision are
flagged `needs_train_recheck` and must be re-checked on the Training set before approval.

Input : docs/preliminary_pre_cohort_audit/candidate_variables_draft.csv (initial AI draft, preserved; read only)
Output: docs/preliminary_pre_cohort_audit/candidate_shortlist_prestepwise_draft.csv

Usage:
    python scripts/03c_prestepwise_shortlist.py --root .
"""
import argparse
from pathlib import Path

import pandas as pd

# Evidence abbreviations (literature to be verified by the researcher):
#   4C = Knight, BMJ 2020 (ISARIC 4C Mortality Score); CURB-65 = Lim, Thorax 2003;
#   qSOFA = Seymour, JAMA 2016; NEWS2 = Royal College of Physicians 2017;
#   COVID-GRAM = Liang, JAMA Intern Med 2020; Zhou = Zhou, Lancet 2020;
#   OpenSAFELY = Williamson, Nature 2020; Wynants = Wynants, BMJ 2020 (living SR of prediction models).
# variable -> (decision, reason, main evidence, redundancy concern, unresolved issue, needs_train_recheck)
DECISIONS = {
    "age.splits": ("retain", "COVID-19 死亡の最も強い予後因子で、ほぼすべての既存スコアに含まれる",
                   "4C, CURB-65, COVID-GRAM, OpenSAFELY, Wynants", "なし",
                   "3 階級のみで、90 歳超は最上位の階級に含まれる（L-4）。スコアの年齢区分と一致しない", False),
    "gender_concept_name": ("retain", "男性で死亡リスクが高いことが一貫して報告されている",
                            "4C, OpenSAFELY", "なし", "匿名化による NA（32 例）の扱い", False),
    "9279-1_Respiratory rate": ("retain", "呼吸不全の直接の指標。主要スコアの多くに含まれる",
                                "4C, CURB-65, qSOFA, NEWS2", "SpO2・呼吸困難と相関するが、換気の努力と酸素化は別の側面",
                                "測定時点（Q-B3）。95 /min の値（Q-B14）", False),
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": (
        "retain", "低酸素血症は COVID-19 の重症度の中心的な指標", "4C, NEWS2",
        "呼吸数・呼吸困難と相関する", "酸素投与の有無の情報がないため、同じ値でも意味が変わりうる。測定時点（Q-B3）", False),
    "8480-6_Systolic blood pressure": (
        "retain", "循環不全の領域を代表する唯一の変数。低血圧は臓器不全の指標", "CURB-65, qSOFA, NEWS2",
        "心拍数・平均血圧と相関する", "4C の最終モデルには含まれていない（要文献確認）。測定時点（Q-B3）", False),
    "76282-3_Heart rate.beat-to-beat by EKG": (
        "reserve", "単独での根拠は NEWS2 程度。収縮期血圧・呼吸数・体温と同じく全身反応を表し、重なりが大きい",
        "NEWS2", "収縮期血圧、呼吸数、体温", "6 /min の値（Q-B14）", False),
    "8331-1_Oral temperature": (
        "drop", "COVID-19 死亡の予測因子としての支持は弱い（4C の最終モデルに含まれない：要文献確認）。炎症は CRP で代表する",
        "NEWS2", "CRP、心拍数", "–", False),
    "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma": (
        "retain", "腎機能、脱水、異化を反映する。既存スコアで尿素として使われている", "4C（尿素）, CURB-65",
        "クレアチニン（強い）、CKD", "単位は mg/dL と仮定（Q-B12）。ER で欠測が多い（検査オーダーへの依存）", True),
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma": (
        "reserve", "腎の領域は BUN（急性）と CKD（慢性）で代表されている。スコアでの支持は BUN の方が強い",
        "SOFA（腎）", "BUN（強い）、CKD、eGFR", "BUN の代わりに使う案（代替候補）", False),
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": (
        "retain", "炎症の領域を代表する。既存スコアに含まれる", "4C, Wynants",
        "D-dimer、リンパ球数（炎症反応として）", "単位が不明（Q-B12）。検査オーダーへの依存（ER で 75% 欠測）", True),
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": (
        "retain", "リンパ球減少は重症化・死亡との関連が繰り返し報告されている",
        "COVID-GRAM（NLR）, Zhou, Wynants；リンパ球減少のメタ解析（要文献確認）",
        "好中球数（NLR の構成要素）、白血球数", "白血球分画が必要（白血球数より欠測が多い）。好中球数 > 白血球数が 13 例（Q-B14）", True),
    "751-8_Neutrophils [#/volume] in Blood by Automated count": (
        "reserve", "単独での根拠は主に NLR を通したもの。リンパ球数を残し、NLR は別案とする",
        "COVID-GRAM（NLR）", "リンパ球数、白血球数、CRP", "NLR を採用するかどうか（別案。研究設計の変更になる）", False),
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": (
        "retain", "凝固異常を表す。多変量解析で独立した死亡予測因子として報告されている", "Zhou, Wynants",
        "troponin と同じく重症化を疑うときに出される検査。CRP とも関連する",
        "検査オーダーへの依存が強い（ER で 79% 欠測）。検出限界 150 に集中", True),
    "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": (
        "reserve", "心筋障害の根拠は強いが、D-dimer と同じように検査オーダーに依存する。測定値の大部分が検出限界にある",
        "Zhou", "D-dimer、冠動脈疾患、心不全", "0.01 に集中しており、実質的に二値に近い。D-dimer の代わりに使う案（代替候補）", True),
    "dm_v": ("retain", "糖尿病は死亡リスクの上昇と一貫して関連する", "4C（併存症数）, OpenSAFELY",
             "BMI、CKD、冠動脈疾患（代謝・血管）", "カルテ抽出なしの 270 例が NA（Q-B11）", False),
    "ckd_v": ("retain", "慢性腎臓病は強い関連が報告されている。BUN（急性の状態）とは別の、慢性の背景を表す",
              "4C（併存症数）, OpenSAFELY", "BUN、クレアチニン。定義に「検査での GFR 低下」を含む",
              "定義に今回の visit の検査値が使われたか（Q-B3）。Q-B11", False),
    "cad_v": ("retain", "慢性心疾患の代表とする（パラメータ 1 つ、有病率 12%）", "4C（慢性心疾患）, OpenSAFELY",
              "心不全、糖尿病、高血圧", "心不全とまとめて「慢性心疾患」にする案は recoding になる（別案）。Q-B11", False),
    "hf_ef_v": ("reserve", "心疾患は冠動脈疾患で代表する。3 値でパラメータが 2 つ必要で、HFpEF と HFrEF の人数が少ない",
                "4C（慢性心疾患）, OpenSAFELY", "冠動脈疾患", "二値にまとめる（HF あり／なし）のは recoding になる（別案）", False),
    "copd_v": ("reserve", "根拠はあるが、有病率が約 5% と低く、パラメータあたりの情報量が小さい。定義に喫煙歴を含む",
               "4C（慢性肺疾患）, OpenSAFELY", "喫煙（B）、その他の肺疾患（B）", "shortlist を 14 変数にするなら最初に加える候補", False),
    "malignancies_v": ("retain", "悪性腫瘍は死亡リスクの上昇と関連し、複数のスコアに含まれる",
                       "4C（併存症数）, COVID-GRAM, OpenSAFELY", "なし", "定義は ICD-10 または治療中の活動性腫瘍。Q-B11", False),
    "39156-5_Body mass index (BMI) [Ratio]": (
        "reserve", "肥満の根拠は強いが、欠測が多く（ER で 77%）、測定されるかどうかに依存する。糖尿病などの併存症の負荷とも関連する",
        "OpenSAFELY, 4C（肥満）", "糖尿病（高血圧）", "関係が直線でない（U 字型）可能性。欠測は Train で確認し直す", True),
    "dyspnea_admission_v": ("reserve", "呼吸の状態は呼吸数と SpO2 という客観的な指標で表している。自己申告の症状で、カルテ抽出列",
                            "COVID-GRAM", "呼吸数、SpO2", "Q-B11（カルテ抽出なしの 270 例）", False),
}
LABEL = {"retain": "A. 推奨 shortlist", "reserve": "B. 代替候補", "drop": "C. 除外候補"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    root = Path(ap.parse_args().root)
    draft = pd.read_csv(root / "docs/preliminary_pre_cohort_audit/candidate_variables_draft.csv", encoding="utf-8-sig")
    tier_a = draft[draft["provisional_tier"].str.startswith("A")]
    assert set(tier_a["variable"]) == set(DECISIONS), set(tier_a["variable"]) ^ set(DECISIONS)

    rows = []
    for _, r in tier_a.iterrows():
        dec, reason, evid, redund, unresolved, recheck = DECISIONS[r["variable"]]
        miss = (f"全データ {r['missing_rate_all_descriptive']:.0%}（ER {r['missing_rate_ER']:.0%} / "
                f"入院 {r['missing_rate_inpatient']:.0%}）。{r['missingness_flag'] or ''}")
        rows.append(dict(raw_variable=r["variable"], short_name=r["short_name"], clinical_domain=r["domain"],
                         decision=dec, group=LABEL[dec], reason=reason, main_evidence=evid,
                         redundancy_concern=redund, missingness_concern=miss,
                         expected_model_parameters=int(r["n_parameters_if_used"]),
                         unresolved_issue=unresolved, needs_train_recheck=recheck,
                         status="案（研究責任者の承認待ち）"))
    out = pd.DataFrame(rows)
    out["_o"] = out["decision"].map({"retain": 0, "reserve": 1, "drop": 2})
    out = out.sort_values(["_o", "clinical_domain"], kind="stable").drop(columns="_o")
    out.to_csv(root / "docs/preliminary_pre_cohort_audit/candidate_shortlist_prestepwise_draft.csv", index=False, encoding="utf-8-sig")
    print(out.groupby("group")["expected_model_parameters"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
