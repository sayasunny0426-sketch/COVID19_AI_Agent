"""Task B: clinical table data model -- data access and the variable registry.

Everything a reader needs in order to know *why* a column is used or dropped lives in this
module, next to the column name, so the audit table and the modelling code cannot disagree
about the variable set (a defect found in the previous Task B implementation, where the saved
variable-selection documents described a model that was never fitted).

Design rules enforced here:
  * the fixed 1,277-patient split is read, never recreated;
  * every decision carries a rationale and an evidence source;
  * nothing in this module looks at Validation or Test rows.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CLINICAL_CSV = "data/raw/clinical/患者データファイル.csv"
DICTIONARY_CSV = "data/raw/clinical/患者データまとめ、定義.csv"
SPLIT_CSV = "data/splits/COVID19_固定患者split_1277.csv"

EXPECTED = {"cohort": 1277, "train": 1021, "val": 128, "test": 128,
            "deaths": {"train": 135, "val": 17, "test": 17}}

# --------------------------------------------------------------------------------------
# Column roles. T0 = visit_start_datetime (fixed in Task A, D-020). A column is usable only
# if its value is knowable at T0.
# --------------------------------------------------------------------------------------
OUTCOME = "last.status"
ID_COLUMN = "to_patient_id"

POST_T0 = {
    "last.status": "アウトカムそのもの（院内死亡）",
    "is_icu": "入院後の ICU 入室。T0 では未確定で、重症度の結果指標",
    "was_ventilated": "入院後の侵襲的人工呼吸の実施。T0 では未確定",
    "invasive_vent_days": "人工呼吸日数。入院経過の結果",
    "length_of_stay": "在院日数。エピソード終了後にしか確定せず、死亡と強く交絡",
    "Acute.Hepatic.Injury..during.hospitalization.": "定義に during hospitalization と明記＝T0 以降の合併症",
    "Acute.Kidney.Injury..during.hospitalization.": "定義に during hospitalization と明記＝T0 以降の合併症",
    "kidney_replacement_therapy": "腎代替療法。入院後の治療介入である可能性が高い（実施時期の記載なし）",
    "therapeutic.exnox.Boolean": "治療用エノキサパリン投与＝入院後の治療介入",
    "therapeutic.heparin.Boolean": "治療用量ヘパリン投与＝入院後の治療介入",
    "Other.anticoagulation.therapy": "その他の抗凝固療法＝入院後の治療介入の可能性",
}

ADMIN = {
    "to_patient_id": "患者 ID（管理列）",
    "covid19_statuses": "Training set で positive のみの定数列（情報量ゼロ）",
    "visit_start_datetime": "T0 の定義そのもの。予測変数ではない",
    "visit_concept_name": "受診形態（ER / 入院）。T0 で既知だが triage 判断を含み、"
                          "欠測構造の主因でもあるため予測変数には用いず、欠測の記述にのみ使う（Q-B4）",
}

# Boolean band columns that are exactly reproducible from a raw numeric column.
# Verified on the Training set: agreement 1.000 (temperature 0.977, an equality-edge
# difference) and identical missingness patterns, so they carry no extra information.
DERIVED_BANDS = {
    "8331-1_Oral temperature": ["temperature.over38"],
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": ["pulseOx.under90"],
    "9279-1_Respiratory rate": ["Respiration.over24"],
    "76282-3_Heart rate.beat-to-beat by EKG": ["HeartRate.over100"],
    "8480-6_Systolic blood pressure": ["SBP.below120", "SBP.between120and139", "SBP.above139"],
    "76536-2_Mean blood pressure by Noninvasive": ["MAP.below65", "MAP.between65and90", "MAP.above90"],
    "731-0_Lymphocytes [#/volume] in Blood by Automated count": ["Lymphocytes.under1k"],
    "1920-8_Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma": ["Aspartate.over40"],
    "1744-2_Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma by No addition of P-5'-P":
        ["Alanine.over60"],
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma":
        ["Creatinine.above1.2", "Creatinine.between0.5and1.2", "Creatinine.below0.5"],
    "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma":
        ["Blood_Urea_Nitrogen.above20", "Blood_Urea_Nitrogen.between5and20", "Blood_Urea_Nitrogen.below5"],
    "2951-2_Sodium [Moles/volume] in Serum or Plasma":
        ["Sodium.above145", "Sodium.between135and145", "Sodium.below135"],
    "2823-3_Potassium [Moles/volume] in Serum or Plasma":
        ["Potassium.above5.2", "Potassium.between3.5and5.2", "Potassium.below3.5"],
    "2075-0_Chloride [Moles/volume] in Serum or Plasma":
        ["Chloride.above107", "Chloride.between96and107", "Chloride.below96"],
    "1963-8_Bicarbonate [Moles/volume] in Serum or Plasma":
        ["Bicarbonate.above31", "Bicarbonate.between21and31", "Bicarbonate.below21"],
    "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood "
    "by Creatinine-based formula (CKD-EPI)": ["eGFR.above60", "eGFR.between30and60", "eGFR.below30"],
    "33254-4_pH of Arterial blood adjusted to patient's actual temperature":
        ["blood_pH.above7.45", "blood_pH.between7.35and7.45", "blood_pH.below7.35"],
    "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": ["Troponin.above0.01"],
    "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay":
        ["D_dimer.above3000", "D_dimer.between500and3000", "D_dimer.below500"],
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma": ["ferritin.above1k"],
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay":
        ["procalcitonin.below0.25", "procalcitonin.between0.25and0.5", "procalcitonin.above0.5"],
    "30341-2_Erythrocyte sedimentation rate": ["ESR.above30"],
    "4548-4_Hemoglobin A1c/Hemoglobin.total in Blood":
        ["A1C.over6.5", "A1C.under6.5", "A1C.6.6to7.9", "A1C.8to9.9", "A1C.over10"],
    "39156-5_Body mass index (BMI) [Ratio]": ["BMI.over30", "BMI.over35"],
}
EXACT_DUPLICATES = {"2951-2_Sodium [Moles/volume] in Serum or Plasma.1":
                    "2951-2_Sodium [Moles/volume] in Serum or Plasma"}

# Chart-abstracted columns: 178 Training patients have every one of them missing. Their
# absence is itself associated with the outcome, so it is modelled, never imputed away.
CHART_BLOCK_NOTE = ("カルテ抽出列。Training 1,021 例中 178 例で 21 列すべてが欠測し、"
                    "その群の死亡率は 9.6%（抽出あり 14.0%）。欠測を平均・最頻値で埋めず "
                    "Missing カテゴリとして保持する")

# Physiologically impossible values found in the Training set. Set to missing (the patient
# is never dropped); plain statistical outliers are left untouched (faculty feedback §6).
IMPLAUSIBLE_BOUNDS = {
    "8331-1_Oral temperature": (30.0, 43.0),
    "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry": (40.0, 100.0),
    "9279-1_Respiratory rate": (4.0, 60.0),
    "76282-3_Heart rate.beat-to-beat by EKG": (20.0, 250.0),
    "8480-6_Systolic blood pressure": (50.0, 260.0),
    "76536-2_Mean blood pressure by Noninvasive": (30.0, 200.0),
    "39156-5_Body mass index (BMI) [Ratio]": (12.0, 80.0),
    "2951-2_Sodium [Moles/volume] in Serum or Plasma": (100.0, 180.0),
    "2823-3_Potassium [Moles/volume] in Serum or Plasma": (1.5, 9.0),
    "33254-4_pH of Arterial blood adjusted to patient's actual temperature": (6.7, 7.8),
}

# Assay floors: values at or below these are "not detectable" rather than a measurement.
# Decided from the value distribution and the assay, NOT from any outcome association.
ASSAY_FLOOR = {"6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma": 0.01}


@dataclass(frozen=True)
class Variable:
    """One clinical candidate, with the reason it is proposed and where that reason comes from."""
    name: str                 # short model-facing name
    column: str               # raw column in 患者データファイル.csv
    domain: str
    kind: str                 # "continuous" | "binary" | "categorical" | "derived"
    rationale: str            # why this variable could predict in-hospital death
    evidence: str             # prior literature / established risk score
    evidence_source: str      # clinical knowledge | prior literature | established risk score | data
    caveat: str = ""
    params: int = 1           # estimated coefficients once encoded
    members: tuple = field(default_factory=tuple)   # for derived variables


# --------------------------------------------------------------------------------------
# Candidate registry. Selected BEFORE looking at any outcome association, from the 4C
# Mortality Score (Knight et al., BMJ 2020 -- the most widely externally validated COVID-19
# in-hospital mortality score) plus domains repeatedly reported as independent predictors.
# Faculty feedback §2 requires exactly this order: clinical knowledge and literature first,
# data-driven narrowing second.
# --------------------------------------------------------------------------------------
CANDIDATES: tuple[Variable, ...] = (
    Variable("age", "age.splits", "人口統計", "categorical",
             "年齢は COVID-19 院内死亡の最も強い予後因子で、ほぼすべての既存スコアに含まれる",
             "4C Mortality Score, CURB-65, COVID-GRAM, OpenSAFELY", "established risk score",
             caveat="3 階級のみで 90 歳超は最上位階級に統合。既存スコアの年齢区分を再現できない（L-4）",
             params=2),
    Variable("sex", "gender_concept_name", "人口統計", "binary",
             "男性で死亡リスクが高いことが一貫して報告されている",
             "4C Mortality Score, OpenSAFELY", "established risk score",
             caveat="匿名化のため一部で欠測（Train 1.9%）"),
    Variable("spo2", "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry", "呼吸", "continuous",
             "低酸素血症は COVID-19 重症度の中心的指標で、呼吸不全の直接の所見",
             "4C Mortality Score, NEWS2", "established risk score",
             caveat="酸素投与の有無が記録されておらず、同じ値でも臨床的意味が異なりうる（Q-B7）"),
    Variable("rr", "9279-1_Respiratory rate", "呼吸", "continuous",
             "呼吸数は呼吸仕事量を反映し、酸素化とは別の側面を表す",
             "4C Mortality Score, CURB-65, qSOFA, NEWS2", "established risk score"),
    Variable("sbp", "8480-6_Systolic blood pressure", "循環", "continuous",
             "低血圧は循環不全・臓器灌流低下の指標。循環動態を代表する",
             "CURB-65, qSOFA, NEWS2", "established risk score"),
    Variable("bun", "3094-0_Urea nitrogen [Mass/volume] in Serum or Plasma", "腎（急性）", "continuous",
             "尿素窒素の上昇は腎機能低下と脱水・異化亢進を反映し、4C の構成要素",
             "4C Mortality Score, CURB-65", "established risk score",
             caveat="eGFR・クレアチニンと相関（共線性）"),
    Variable("crp", "1988-5_C reactive protein [Mass/volume] in Serum or Plasma", "炎症", "continuous",
             "全身性炎症反応の代表指標で 4C の構成要素",
             "4C Mortality Score, Wynants 系統的レビュー", "established risk score",
             caveat="単位が data dictionary に明記されていない（Q-B12）。ER で 73% 欠測"),
    Variable("lymph", "731-0_Lymphocytes [#/volume] in Blood by Automated count", "血液", "continuous",
             "リンパ球減少は COVID-19 の重症化・死亡と繰り返し関連が報告されている",
             "COVID-GRAM（NLR）, Zhou et al. Lancet 2020", "prior literature",
             caveat="好中球数・白血球数と相関"),
    Variable("ddimer", "48058-2_Fibrin D-dimer DDU [Mass/volume] in Platelet poor plasma by Immunoassay",
             "凝固", "continuous",
             "凝固活性化・血栓症を反映し、多変量解析で独立した死亡予測因子として報告されている",
             "Zhou et al. Lancet 2020, Wynants 系統的レビュー", "prior literature",
             caveat="ER で 78% 欠測（オーダー依存）。検出限界 150 に 9% が集中"),
    Variable("lactate", "2524-7_Lactate [Moles/volume] in Serum or Plasma", "組織灌流", "continuous",
             "乳酸上昇は組織低灌流・嫌気性代謝を示し、敗血症性ショックの診断基準に含まれる",
             "Sepsis-3, Zhou et al. Lancet 2020", "clinical knowledge",
             caveat="ER での測定が多く、入院例では欠測しやすい"),
    Variable("egfr",
             "62238-1_Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, "
             "Plasma or Blood by Creatinine-based formula (CKD-EPI)", "腎（慢性）", "continuous",
             "推算 GFR は慢性腎機能を表し、BUN（急性の変動）とは異なる情報を持つ",
             "OpenSAFELY（CKD）, 4C（併存症）", "prior literature",
             caveat="BUN・クレアチニンと強く相関。120 で上限打ち切り（Train の 12%）"),
    Variable("troponin", "6598-7_Troponin T.cardiac [Mass/volume] in Serum or Plasma", "心筋",
             "binary_threshold",
             "心筋傷害は COVID-19 死亡と強く関連することが報告されている",
             "Zhou et al. Lancet 2020, Shi et al. JAMA Cardiol 2020", "prior literature",
             caveat="Training の 78.2%（568/726）が検出限界 0.01 に集中しており、"
                    "連続値として扱うと測定系が持たない精度を仮定することになる。"
                    "そのため『検出可能（>0.01）』の二値として扱う（Q-B5 の決定）"),
    Variable("comorbidity", "__derived__", "併存症", "derived",
             "併存症の数は 4C の構成要素で、個々の疾患を別々に入れるよりパラメータ効率が高い",
             "4C Mortality Score（number of comorbidities）, OpenSAFELY", "established risk score",
             caveat=CHART_BLOCK_NOTE, params=2,
             members=("htn_v", "dm_v", "cad_v", "hf_ef_v", "ckd_v", "malignancies_v",
                      "copd_v", "other_lung_disease_v")),
)

# Considered but not carried into the candidate set; the reason is recorded so that
# "not used" is never confused with "not thought about".
CONSIDERED_NOT_SELECTED = {
    "39156-5_Body mass index (BMI) [Ratio]":
        "肥満は死亡リスク因子だが Train 欠測 32%。4C には含まれず、パラメータ予算を優先して見送り",
    "8331-1_Oral temperature":
        "発熱は重症度と単調に関連せず（低体温も予後不良）、既存スコアでの寄与も小さいため見送り",
    "76282-3_Heart rate.beat-to-beat by EKG":
        "SBP・呼吸数と重複する循環・代償の指標で、4C には含まれないため見送り",
    "2276-4_Ferritin [Mass/volume] in Serum or Plasma":
        "炎症の指標だが CRP と重複し、Train 欠測 35%",
    "75241-0_Procalcitonin [Mass/volume] in Serum or Plasma by Immunoassay":
        "細菌感染合併の指標で CRP と重複。Train 欠測 23%",
    "33256-9_Leukocytes [#/volume] corrected for nucleated erythrocytes in Blood by Automated count":
        "リンパ球数と重複（NLR の構成要素）。リンパ球減少の方が文献的根拠が強い",
    "751-8_Neutrophils [#/volume] in Blood by Automated count":
        "同上。リンパ球数と強く相関",
    "2160-0_Creatinine [Mass/volume] in Serum or Plasma":
        "BUN・eGFR と強い共線性。腎領域は BUN（急性）と eGFR（慢性）で代表させる",
    "days_prior_sx":
        "発症からの日数は予後と関連しうるが、自己申告で Train 欠測 24%。既存スコアに含まれない",
    "smoking_status_v":
        "COVID-19 死亡との関連は文献間で一貫しない。Train 欠測 22%",
    "33762-6_Natriuretic peptide.B prohormone N-Terminal [Mass/volume] in Serum or Plasma":
        "心不全の指標だが Train 欠測 42%、併存症の hf と重複",
    "symptoms (cough_v / dyspnea_admission_v / fever_v ほか)":
        "自覚症状は主観的で Train 欠測 21-25%、既存スコアでは客観的生理指標に置換されている",
    "A1C / 脂質 / ESR / 尿検査":
        "Train 欠測 55-85% で、かつ急性期の死亡予測における文献的根拠が弱い",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_manifest(project: Path) -> pd.DataFrame:
    """The fixed split, IDs and split labels only. Reading this does not open the Test data."""
    return pd.read_csv(Path(project) / SPLIT_CSV, dtype=str, encoding="utf-8-sig")


def load_cohort(project: Path, splits: tuple[str, ...] = ("train", "val", "test")) -> pd.DataFrame:
    """Join the clinical table onto the fixed split.

    `splits` controls which split's ROWS are materialised. Everything that runs before the
    Test evaluation is approved passes `splits=("train", "val")`, so no Test outcome and no
    Test clinical value is ever read into memory. The split manifest is still checked in full
    -- patient counts and the absence of overlap are properties of the manifest, not of the
    Test data -- but Test rows are dropped before the clinical table is joined.
    """
    project = Path(project)
    clin = pd.read_csv(project / CLINICAL_CSV, dtype=str, encoding="utf-8-sig")
    split = split_manifest(project)
    if clin[ID_COLUMN].duplicated().any():
        raise ValueError("clinical file has duplicated patient ids")

    # manifest-level integrity: sizes and disjointness, using the split column only
    counts = split.split.value_counts().to_dict()
    if len(split) != EXPECTED["cohort"] or any(counts.get(k) != EXPECTED[k]
                                               for k in ("train", "val", "test")):
        raise ValueError(f"cohort/split mismatch in the manifest: n={len(split)} {counts}")
    if split[ID_COLUMN].duplicated().any():
        raise ValueError("the split manifest contains duplicated patient ids")

    wanted = split[split.split.isin(splits)].copy()
    df = wanted.merge(clin, on=ID_COLUMN, how="left", validate="one_to_one",
                      suffixes=("_split", ""))
    if df[OUTCOME].isna().any():
        missing = df.loc[df[OUTCOME].isna(), ID_COLUMN].tolist()
        raise ValueError(f"{len(missing)} split patients absent from the clinical file: {missing[:5]}")

    deaths = df[df.true_label == "1"].split.value_counts().to_dict()
    for s in splits:
        if deaths.get(s, 0) != EXPECTED["deaths"][s]:
            raise ValueError(f"death count for {s} differs from the frozen split: {deaths}")
    inconsistent = ((df[OUTCOME] == "deceased") != (df.true_label == "1")).sum()
    if inconsistent:
        raise ValueError(f"{inconsistent} rows where last.status disagrees with true_label")

    df["y"] = (df.true_label == "1").astype(int)
    return df


def load_development(project: Path) -> pd.DataFrame:
    """Training + Validation only. The default loader for everything before Test approval."""
    return load_cohort(project, splits=("train", "val"))


def training_only(df: pd.DataFrame) -> pd.DataFrame:
    """The only frame that preprocessing parameters and variable selection may look at."""
    return df[df.split == "train"].copy()


def dictionary(project: Path) -> pd.DataFrame:
    return pd.read_csv(Path(project) / DICTIONARY_CSV, encoding="utf-8-sig")


def dropped_derived_columns() -> dict[str, str]:
    """Band columns and exact duplicates, with the reason each one is dropped."""
    out = {}
    for source, bands in DERIVED_BANDS.items():
        for b in bands:
            out[b] = (f"連続変数 `{source[:44]}` から完全に再現可能な区分ダミー。"
                      f"Training set で一致率 1.000、欠測パターンも一致し追加情報がない")
    for dup, keep in EXACT_DUPLICATES.items():
        out[dup] = f"`{keep[:44]}` と完全に同一（値・欠測パターンとも）"
    return out


def comorbidity_count(df: pd.DataFrame) -> pd.Series:
    """4C-style comorbidity count: 0 / 1 / >=2, or 'NotAbstracted' when the chart block is absent.

    A count is only meaningful if the chart was abstracted at all, so patients whose entire
    chart-abstracted block is missing get their own level rather than an imputed number.
    """
    members = [v for v in CANDIDATES if v.name == "comorbidity"][0].members
    present = df[list(members)]
    yes = present.apply(lambda s: s.isin(["Yes", "HFpEF", "HFrEF"])).sum(axis=1)
    all_missing = present.isna().all(axis=1)
    level = pd.Series(pd.NA, index=df.index, dtype="object")
    level[~all_missing & (yes == 0)] = "0"
    level[~all_missing & (yes == 1)] = "1"
    level[~all_missing & (yes >= 2)] = "2+"
    level[all_missing] = "NotAbstracted"
    return level
