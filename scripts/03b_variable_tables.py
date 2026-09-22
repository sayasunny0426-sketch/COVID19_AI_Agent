"""Build the clinical variable classification and the candidate raw-variable table.

PRELIMINARY (D-035): run on all 1,384 patients before the CXR-based cohort and split were
fixed. Outputs are kept for the record only and must NOT be used for final variable
selection or preprocessing decisions; redo on the Training set of the fixed cohort.

Inputs
  data/raw/clinical/患者データファイル.csv        clinical data
  data/raw/clinical/患者データまとめ、定義.csv    data dictionary (description / type / chart-abstraction)
  data/interim/clinical_audit/*                   outputs of scripts/03_clinical_audit.py
Outputs
  docs/preliminary_pre_cohort_audit/clinical_variable_classification.csv
  docs/preliminary_pre_cohort_audit/candidate_variables_draft.csv

Judgements below are explicit and traceable:
  - the dictionary (column_type / is_chart_abstracted / description) is used to understand
    meaning, acquisition and timing; its column_counts refer to a different population
    (1,479) and are not used;
  - candidate tiers are an independent judgement (clinical knowledge, prior COVID-19
    mortality studies, existing risk scores, T0 availability, missingness, leakage,
    redundancy);
  - missing rates are descriptive on all 1,384 patients (label-blind); decisions that
    depend on missingness must be recomputed on the Training set.

Usage:
    python scripts/03b_variable_tables.py --root .
"""
import argparse
from pathlib import Path

import pandas as pd

CLINICAL_FILE = "data/raw/clinical/患者データファイル.csv"
DICT_FILE = "data/raw/clinical/患者データまとめ、定義.csv"
AUDIT_DIR = "data/interim/clinical_audit"

# ---------------------------------------------------------------- classification
# column -> (category, domain, t0_availability, t0_basis, note)
T0_UNK = "T0 候補（測定時点は未確認）"
CLASS = {
    "to_patient_id": ("identifier_administrative", "ID", "該当なし", "-", "metadata.csv の Subject ID と 1,384 例で一致"),
    "covid19_statuses": ("identifier_administrative", "選択基準", "該当なし", "-", "全例 positive（定数）"),
    "visit_start_datetime": ("identifier_administrative", "時点", "T0 の定義", "辞書: Date shifted start of selected visit", "日付のみで時刻なし"),
    "visit_concept_name": ("identifier_administrative", "visit 種別", "不明（T0 時点で確定しているか要確認）", "辞書: ED で経過観察入院（複数日）もありうる",
                           "Inpatient 1,025 / ER 357 / Outpatient 2。予測変数にするかは Q-B10"),
    "last.status": ("outcome", "転帰", "転帰", "辞書: For the selected visit the status of the patient", "discharged 1,201 / deceased 183。選ばれた visit の終了時点の状態"),
    "length_of_stay": ("outcome_leakage", "経過", "T0 後", "辞書: Number of calendar days in the facility", "死亡または退院で決まる"),
    "invasive_vent_days": ("outcome_leakage", "治療", "T0 後", "辞書: documented days of invasive ventilation", "最大値 40 が 20 例（打ち切りの可能性）"),
    "is_icu": ("post_T0", "経過", "T0 後", "辞書: ICU admission based on room charges", ""),
    "Acute.Hepatic.Injury..during.hospitalization.": ("post_T0", "合併症", "T0 後", "辞書: ALT/AST が基準上限の 15 倍以上（入院中）", ""),
    "Acute.Kidney.Injury..during.hospitalization.": ("post_T0", "合併症", "T0 後", "辞書: 48 時間以内の Cr 0.3 mg/dL 上昇（入院中）", ""),
    "was_ventilated": ("treatment_intervention", "治療", "T0 後", "辞書: had invasive ventilation", ""),
    "kidney_replacement_therapy": ("treatment_intervention", "治療/既往", "不明（慢性透析か入院中の腎代替療法か、辞書からは判別できない）",
                                   "辞書: documented renal replacement therapy", "値は Yes / NA のみ"),
    "therapeutic.exnox.Boolean": ("treatment_intervention", "治療", "T0 後（投与）", "辞書: enoxaparin administered", ""),
    "therapeutic.heparin.Boolean": ("treatment_intervention", "治療", "T0 後（投与）", "辞書: heparin at a therapeutic dose", ""),
    "Other.anticoagulation.therapy": ("treatment_intervention", "治療", "不明（辞書に時点の記載なし）", "辞書: had other anticoagulation therapy as listed",
                                      "argatroban（入院中に使う薬）を含む。wafarin は原文の綴り"),
    "age.splits": ("T0_candidate", "人口統計", "T0 時点で利用可能", "辞書: Age intervals at time of admission; truncated for patients over 90",
                   "3 階級のみ。90 歳超は (74,90] に含まれる（L-4）"),
    "gender_concept_name": ("T0_candidate", "人口統計", "T0 時点で利用可能", "辞書: gender is dropped in select cases for de-identification",
                            "NA（32 例）は匿名化のために削除されたもの"),
    "htn_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: ICD-10 / 降圧薬 / SBP>140/90（カルテ抽出）", ""),
    "dm_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: ICD-10 / 糖尿病治療薬（カルテ抽出）", ""),
    "cad_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: ICD-10 / stent / cath（カルテ抽出）", ""),
    "hf_ef_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: HFrEF (EF<40%) / HFpEF（カルテ抽出）", "No / HFpEF / HFrEF の 3 値"),
    "ckd_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: ICD-10 / GFR 低下（カルテ抽出）", ""),
    "malignancies_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: ICD-10 / 活動性悪性腫瘍の治療中（カルテ抽出）", ""),
    "copd_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: ICD-10 / PFT + 喫煙歴（カルテ抽出）", ""),
    "other_lung_disease_v": ("T0_candidate", "併存症", "T0 時点で利用可能（既往）", "辞書: 喘息、ILD、肺高血圧、慢性 PE、肺切除（カルテ抽出）", "性質の異なる疾患をまとめた列"),
    "kidney_transplant": ("T0_candidate", "併存症", "T0 時点で利用可能の見込み（既往）", "辞書: had a kidney transplant（時点の記載なし）", "値は Yes / NA のみ（Q-B11）"),
    "acei_v": ("T0_candidate", "常用薬", "T0 時点で利用可能（自宅での常用薬）", "辞書: 入院時の服薬照合で確認した home medication（カルテ抽出）", ""),
    "arb_v": ("T0_candidate", "常用薬", "T0 時点で利用可能（自宅での常用薬）", "辞書: 入院時の服薬照合で確認した home medication（カルテ抽出）", ""),
    "nsaid_use_v": ("T0_candidate", "常用薬", "T0 時点で利用可能（自宅での常用薬）", "辞書: 入院時の服薬照合で確認した home medication（カルテ抽出）", ""),
    "antibiotics_use_v": ("T0_candidate", "薬剤", "T0 時点で利用可能（受診前）", "辞書: on an antibiotic prior to presentation（カルテ抽出）", ""),
    "smoking_status_v": ("T0_candidate", "生活歴", "T0 時点で利用可能", "辞書: 紙巻きたばこ・葉巻のみ（カルテ抽出）", ""),
    "days_prior_sx": ("T0_candidate", "症状", "T0 時点で利用可能", "辞書: 受診の何日前に症状が始まったか（カルテ抽出）", "0〜60"),
    "cough_v": ("T0_candidate", "症状", "T0 時点で利用可能（受診時の申告）", "辞書: Reported cough（カルテ抽出）", ""),
    "dyspnea_admission_v": ("T0_candidate", "症状", "T0 時点で利用可能", "辞書: Shortness of breath on admission（カルテ抽出）", ""),
    "nausea_v": ("T0_candidate", "症状", "T0 時点で利用可能（受診時の申告）", "辞書: Reported nausea（カルテ抽出）", ""),
    "vomiting_v": ("T0_candidate", "症状", "T0 時点で利用可能（受診時の申告）", "辞書: Reported vomiting（カルテ抽出）", ""),
    "diarrhea_v": ("T0_candidate", "症状", "T0 時点で利用可能（受診時の申告）", "辞書: Reported diarrhea（カルテ抽出）", ""),
    "abdominal_pain_v": ("T0_candidate", "症状", "T0 時点で利用可能（受診時の申告）", "辞書: Abdominal pain symptom（カルテ抽出）", ""),
    "fever_v": ("T0_candidate", "症状", "T0 時点で利用可能（自宅での発熱）", "辞書: 自宅での発熱のみ。ED での発熱は含まない（カルテ抽出）", ""),
    "Urine.protein": ("T0_candidate", "尿検査", T0_UNK, "辞書: Protein in urine（時点の記載なし）", "もとの連続値はファイルにない"),
    "Proteinuria.above80": ("T0_candidate", "尿検査", T0_UNK, "辞書: 記載なし", "もとの連続値はファイルにない"),
    "Microscopic_hematuria.above2": ("T0_candidate", "尿検査", T0_UNK, "辞書: Blood in urine（時点の記載なし）", "もとの連続値はファイルにない"),
}
VITAL_NOTES = {
    "8331-1_Oral temperature": ("バイタル", "T0 時点で利用可能の見込み", "辞書: 派生列 temperature.over38 の説明に「at time of admission」とある。連続値の列自体には時点の記載なし", "単位は °C（辞書）"),
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": ("バイタル", T0_UNK, "辞書: 記載なし", "酸素投与下かどうかの情報がない"),
    "9279-1_Respiratory rate": ("バイタル", T0_UNK, "辞書: 記載なし", "最大値 95（暫定範囲の 80 超が 2 例）"),
    "76282-3_Heart rate.beat-to-beat by EKG": ("バイタル", T0_UNK, "辞書: 記載なし", "最小値 6（暫定範囲の 20 未満が 2 例）"),
    "8480-6_Systolic blood pressure": ("バイタル", T0_UNK, "辞書: 記載なし", ""),
    "76536-2_Mean blood pressure by Noninvasive": ("バイタル", T0_UNK, "辞書: 記載なし", "MAP > SBP が 20 例"),
    "39156-5_Body mass index (BMI) [Ratio]": ("身体計測", T0_UNK, "辞書: 記載なし", "最小 11.95、最大 92.8（確認候補）"),
}
LAB_NOTES = {
    "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count": "好中球数 > 白血球数が 13 例",
    "751-8_Neutrophils [#/volume] in Blood by Automated count": "好中球数 > 白血球数が 13 例",
    "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": "0.01 に 766 例が集中（検出限界の可能性）",
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": "150 に 93 例が集中（検出限界の可能性）",
    "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)":
        "120 に 149 例が集中（上限打ち切りの可能性）。クレアチニン、年齢、性別から計算される",
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": "単位が不明（辞書にも記載なし）。0.1 に 22 例",
    "33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma": "最大 267,600（確認候補）",
    "4548-4_Hemoglobin A1c/Hemoglobin.total in Blood": "慢性指標",
}
SODIUM_DUP = "2951-2_Sodium [Moles/volume] in Serum or Plasma.1"
SODIUM = "2951-2_Sodium [Moles/volume] in Serum or Plasma"

# ---------------------------------------------------------------- candidate table
# raw variable -> (short, clinical significance, evidence, tier, tier reason, n_params_if_used)
# Evidence refs (to be verified by the researcher): 4C = Knight BMJ 2020; CURB-65 = Lim Thorax 2003;
# qSOFA = Seymour JAMA 2016; NEWS2 = RCP 2017; COVID-GRAM = Liang JAMA Intern Med 2020;
# Zhou = Zhou Lancet 2020; OpenSAFELY = Williamson Nature 2020.
CAND = {
    "age.splits": ("年齢", "最も強い予後因子", "4C, CURB-65, COVID-GRAM, OpenSAFELY", "A", "全スコアに含まれる。欠測 0", 2),
    "gender_concept_name": ("性別", "男性で死亡リスクが高い", "4C, OpenSAFELY", "A", "欠測 2%（匿名化による）", 1),
    "9279-1_Respiratory rate": ("呼吸数", "呼吸不全", "4C, CURB-65, qSOFA, NEWS2", "A", "欠測 <1%", 1),
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": ("SpO2", "低酸素血症", "4C（room air）, NEWS2", "A", "欠測 <1%。酸素投与の有無がない", 1),
    "8480-6_Systolic blood pressure": ("収縮期血圧", "循環不全", "CURB-65, qSOFA, NEWS2", "A", "欠測 <1%", 1),
    "76282-3_Heart rate.beat-to-beat by EKG": ("心拍数", "循環・全身状態", "NEWS2", "A", "欠測 <1%", 1),
    "8331-1_Oral temperature": ("体温", "全身炎症", "NEWS2", "A", "欠測 3%。入院時の値（辞書）", 1),
    "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma": ("BUN", "腎機能・脱水・異化", "4C（尿素）, CURB-65", "A", "欠測 15%（ER で多い）", 1),
    "1988-5_C reactive protein [Mass/volume] in Serum or Plasma": ("CRP", "炎症", "4C", "A", "欠測 23%（ER で 75%）。単位が不明（Q-B12）", 1),
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": ("リンパ球数", "リンパ球減少は重症化と関連", "COVID-GRAM（NLR）, Zhou", "A", "欠測 29%", 1),
    "751-8_Neutrophils [#/volume] in Blood by Automated count": ("好中球数", "NLR の構成要素", "COVID-GRAM（NLR）", "A", "欠測 29%。好中球数 > 白血球数が 13 例", 1),
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay": ("D-dimer", "凝固異常", "Zhou", "A", "欠測 31%（ER で 79%）。検出限界 150 に集中", 1),
    "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": ("Troponin T", "心筋障害", "Zhou", "A", "欠測 30%。0.01 に 766 例が集中", 1),
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma": ("クレアチニン", "腎機能", "SOFA（腎）", "A", "欠測 15%", 1),
    "dm_v": ("糖尿病", "併存症", "4C（併存症数）, OpenSAFELY", "A", "欠測 20%（カルテ抽出なし 270 例）", 1),
    "ckd_v": ("CKD", "併存症", "4C（併存症数）, OpenSAFELY", "A",
              "欠測 20%（同上）。辞書の定義に「検査での GFR 低下」を含むため、今回の visit の検査値が判定に使われていればクレアチニン・eGFR と重なる可能性がある（要確認）", 1),
    "cad_v": ("冠動脈疾患", "慢性心疾患", "4C（併存症数）, OpenSAFELY", "A", "欠測 20%（同上）", 1),
    "hf_ef_v": ("心不全（EF 別）", "慢性心疾患", "4C（併存症数）, OpenSAFELY", "A", "欠測 21%（同上）。3 値", 2),
    "copd_v": ("COPD", "慢性呼吸器疾患", "4C（併存症数）, OpenSAFELY", "A", "欠測 20%（同上）。辞書の定義に喫煙歴を含むため、喫煙と一部重なる", 1),
    "malignancies_v": ("悪性腫瘍", "併存症", "4C（併存症数）, COVID-GRAM, OpenSAFELY", "A", "欠測 21%（同上）", 1),
    "39156-5_Body mass index (BMI) [Ratio]": ("BMI", "肥満", "4C（肥満）, OpenSAFELY", "A", "欠測 31%（ER で 77%）", 1),
    "dyspnea_admission_v": ("呼吸困難", "呼吸器症状", "COVID-GRAM", "A", "欠測 23%（カルテ抽出なし 270 例）", 1),
    "htn_v": ("高血圧", "併存症", "OpenSAFELY（関連は弱め）", "B",
              "欠測 20%。他の心血管併存症と重なる。辞書の定義に「記録された SBP>140/90」を含み、受診時の血圧が判定に使われた可能性がある（要確認）", 1),
    "76536-2_Mean blood pressure by Noninvasive": ("平均血圧", "循環不全", "SOFA（循環）", "B", "SBP と重なる。欠測 14%（ER で 56%）。MAP > SBP が 20 例", 1),
    "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count": ("白血球数", "炎症", "一般的な指標", "B", "好中球数・リンパ球数と重なる", 1),
    "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)":
        ("eGFR", "腎機能", "SOFA（腎）相当", "B", "クレアチニン、年齢、性別から計算される（重複）。120 で打ち切り", 1),
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay": ("Procalcitonin", "細菌感染の合併・炎症", "Zhou（単変量）", "B", "欠測 24%", 1),
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma": ("Ferritin", "過剰炎症", "Zhou（単変量）", "B", "欠測 35%", 1),
    "2524-7_Lactate [Moles/volume] in Serum or Plasma": ("Lactate", "組織の低灌流", "Sepsis-3（敗血症性ショック）", "B", "欠測 25%。重症例で測定されやすい可能性", 1),
    "33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma": ("NT-proBNP", "心負荷", "COVID-19 の心臓合併症に関する報告（要文献確認）", "B", "欠測 44%", 1),
    "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma": ("AST", "肝・組織障害", "要文献確認", "B", "欠測 19%", 1),
    "1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P": ("ALT", "肝障害", "要文献確認", "B", "欠測 19%。AST と重なる", 1),
    "2345-7_Glucose [Mass/volume] in Serum or Plasma": ("血糖", "高血糖・ストレス反応", "要文献確認", "B", "欠測 15%", 1),
    "2951-2_Sodium [Moles/volume] in Serum or Plasma": ("Na", "電解質", "要文献確認", "B", "欠測 15%", 1),
    "2823-3_Potassium [Moles/volume] in Serum or Plasma": ("K", "電解質", "要文献確認", "B", "欠測 19%", 1),
    "2075-0_Chloride [Moles/volume] in Serum or Plasma": ("Cl", "電解質", "要文献確認", "B", "欠測 15%", 1),
    "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma": ("HCO3", "酸塩基", "要文献確認", "B", "欠測 15%", 1),
    "other_lung_disease_v": ("その他の肺疾患", "慢性呼吸器疾患", "4C（慢性肺疾患。喘息は除く）", "B", "喘息、ILD などを 1 列にまとめており、スコアの定義と一致しない", 1),
    "smoking_status_v": ("喫煙", "生活歴", "報告によって結果が異なる（要文献確認）", "B",
                         "欠測 24%。3 値。辞書の定義には unknown があるが、データでは NA になっている", 2),
    "kidney_transplant": ("腎移植", "免疫抑制", "OpenSAFELY（臓器移植）", "B", "Yes は 22 例（1.6%）。NA の意味が不明", 1),
    "days_prior_sx": ("発症から受診までの日数", "病期", "要文献確認", "B", "欠測 26%", 1),
    "cough_v": ("咳", "症状", "予後との関連の根拠は弱い（要文献確認）", "C", "欠測 24%", 1),
    "fever_v": ("自宅での発熱", "症状", "予後との関連の根拠は弱い（要文献確認）", "C",
                "欠測 23%。辞書では自宅での発熱のみで ED での発熱は含まないため、入院時の体温とは別の概念（重複ではない）", 1),
    "nausea_v": ("悪心", "消化器症状", "根拠は弱い（要文献確認）", "C", "欠測 26%", 1),
    "vomiting_v": ("嘔吐", "消化器症状", "根拠は弱い（要文献確認）", "C", "欠測 26%", 1),
    "diarrhea_v": ("下痢", "消化器症状", "根拠は弱い（要文献確認）", "C", "欠測 25%", 1),
    "abdominal_pain_v": ("腹痛", "消化器症状", "根拠は弱い（要文献確認）", "C", "欠測 26%", 1),
    "acei_v": ("ACE 阻害薬", "常用薬", "重症化との関連は認めないとする報告（要文献確認）", "C", "欠測 21%", 1),
    "arb_v": ("ARB", "常用薬", "同上", "C", "欠測 20%", 1),
    "nsaid_use_v": ("NSAID", "常用薬", "根拠は弱い（要文献確認）", "C", "欠測 24%", 1),
    "antibiotics_use_v": ("受診前の抗菌薬", "受診前の治療", "根拠は弱い（要文献確認）", "C", "欠測 22%", 1),
    "4548-4_Hemoglobin A1c/Hemoglobin.total in Blood": ("HbA1c", "血糖管理", "糖尿病（dm_v）と重なる", "C", "欠測 69%", 1),
    "33254-4_pH of Arterial blood adjusted to patient's actual temperature": ("動脈血 pH", "酸塩基", "SOFA とは別", "C", "欠測 84%。重症例でしか測定されない可能性", 1),
    "2157-6_Creatine kinase [Enzymatic activity/volume] in Serum or Plasma": ("CK", "筋障害", "要文献確認", "C", "欠測 67%", 1),
    "30341-2_Erythrocyte sedimentation rate": ("ESR", "炎症", "CRP と重なる", "C", "欠測 56%", 1),
    "13457-7_Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation": ("LDL", "脂質", "根拠は弱い", "C", "欠測 78%", 1),
    "13458-5_Cholesterol in VLDL [Mass/volume] in Serum or Plasma by calculation": ("VLDL", "脂質", "根拠は弱い", "C", "欠測 78%", 1),
    "2571-8_Triglyceride [Mass/volume] in Serum or Plasma": ("TG", "脂質", "根拠は弱い", "C", "欠測 77%", 1),
    "2085-9_Cholesterol in HDL [Mass/volume] in Serum or Plasma": ("HDL", "脂質", "根拠は弱い", "C", "欠測 78%", 1),
    "Urine.protein": ("尿蛋白", "腎障害", "要文献確認", "C", "欠測 60%。測定時点が不明", 1),
    "Proteinuria.above80": ("蛋白尿 > 80", "腎障害", "要文献確認", "C", "欠測 85%。尿蛋白と重なる", 1),
    "Microscopic_hematuria.above2": ("顕微鏡的血尿", "腎障害", "要文献確認", "C", "欠測 61%", 1),
}
TIER_LABEL = {"A": "一次候補", "B": "二次候補（要検討）", "C": "除外候補"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    root = Path(ap.parse_args().root)

    clin = pd.read_csv(root / CLINICAL_FILE, dtype=str, keep_default_na=False)
    dic = pd.read_csv(root / DICT_FILE, encoding="utf-8-sig", dtype=str, keep_default_na=False).set_index("column_name")
    assert dic.index.tolist() == clin.columns.tolist(), "dictionary and data columns differ"
    prof = pd.read_csv(root / AUDIT_DIR / "clinical_column_profile.csv").set_index("column")
    mv = pd.read_csv(root / AUDIT_DIR / "missing_rate_by_visit_type.csv", index_col=0)
    der = pd.read_csv(root / AUDIT_DIR / "derived_binary_consistency.csv").set_index("binary_column")

    rows = []
    for col in clin.columns:
        if col in CLASS:
            cat, dom, t0, basis, note = CLASS[col]
            derived_from = ""
        elif col in der.index:
            cat, dom, t0, derived_from = "duplicate_derived", "派生", "—", der.loc[col, "source_column"]
            basis = "監査: 連続値から閾値で決定的に作られている"
            note = f"閾値 {der.loc[col, 'rule']}。不一致は strict {der.loc[col, 'mismatch_strict']} / inclusive {der.loc[col, 'mismatch_inclusive']}"
        elif col == SODIUM_DUP:
            cat, dom, t0, derived_from, basis, note = "duplicate_derived", "検査", "—", SODIUM, "監査: 全行一致", "重複列"
        elif col in VITAL_NOTES:
            dom, t0, basis, note = VITAL_NOTES[col]
            cat, derived_from = "T0_candidate", ""
        else:  # remaining continuous labs
            cat, dom, t0, basis, note, derived_from = "T0_candidate", "検査", T0_UNK, "辞書: 記載なし（LOINC 名のみ）", LAB_NOTES.get(col, ""), ""
        rate = prof.loc[col, "missing_rate"]
        er, ip = (mv.loc[col, "Emergency Room Visit"], mv.loc[col, "Inpatient Visit"]) if col in mv.index else (None, None)
        flags = []
        if cat == "T0_candidate":
            if rate >= 0.5:
                flags.append("high(>=50%)")
            elif rate >= 0.2:
                flags.append("moderate(20-50%)")
            if er - ip >= 0.3:
                flags.append("visit種別で大きく異なる")
            if dic.loc[col, "is_chart_abstracted"] == "TRUE":
                flags.append("カルテ抽出列（270 例で一括して NA）")
            if prof.loc[col, "value_set_if_<=10"] == "Yes":
                flags.append("Yes/NA のみ")
        rows.append(dict(column=col, category=cat, domain=dom, t0_availability=t0, t0_basis=basis,
                         derived_from=derived_from, dict_column_type=dic.loc[col, "column_type"],
                         dict_is_chart_abstracted=dic.loc[col, "is_chart_abstracted"],
                         dict_description=dic.loc[col, "description"],
                         missing_rate_all_descriptive=rate, missing_rate_ER=er, missing_rate_inpatient=ip,
                         missingness_flag=";".join(flags), value_set=prof.loc[col, "value_set_if_<=10"],
                         note=note))
    cls = pd.DataFrame(rows)
    cls.to_csv(root / "docs/preliminary_pre_cohort_audit/clinical_variable_classification.csv", index=False, encoding="utf-8-sig")

    t0_cols = cls.loc[cls.category == "T0_candidate", "column"].tolist()
    assert set(t0_cols) == set(CAND), set(t0_cols) ^ set(CAND)
    cand = []
    for col in t0_cols:
        short, sig, evid, tier, reason, npar = CAND[col]
        r = cls.set_index("column").loc[col]
        cand.append(dict(variable=col, short_name=short, domain=r["domain"], clinical_significance=sig,
                         evidence_score_or_study=evid, dict_description=r["dict_description"],
                         dict_is_chart_abstracted=r["dict_is_chart_abstracted"],
                         t0_availability=r["t0_availability"], t0_basis=r["t0_basis"],
                         missing_rate_all_descriptive=r["missing_rate_all_descriptive"],
                         missing_rate_ER=r["missing_rate_ER"], missing_rate_inpatient=r["missing_rate_inpatient"],
                         missingness_flag=r["missingness_flag"], provisional_tier=f"{tier}:{TIER_LABEL[tier]}",
                         tier_reason=reason, n_parameters_if_used=npar,
                         status="案（研究責任者の承認待ち）"))
    cand = pd.DataFrame(cand).sort_values(["provisional_tier", "domain"], kind="stable")
    # The initial AI draft is preserved as a record (D-033, SHA256 6ac5c482...). Never overwrite it.
    draft = root / "docs/preliminary_pre_cohort_audit/candidate_variables_draft.csv"
    if draft.exists():
        print(f"{draft} exists and is preserved (D-033) -> candidate table not rewritten")
    else:
        cand.to_csv(draft, index=False, encoding="utf-8-sig")
    print(cls["category"].value_counts().to_string())
    print(cand.groupby("provisional_tier")["n_parameters_if_used"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
