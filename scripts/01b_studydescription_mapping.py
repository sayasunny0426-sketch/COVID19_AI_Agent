"""StudyDescription mapping for CR/DX studies (pre-final; index CXR selection NOT applied).

Rules set by the researcher (2026-09-20):
  - CR / DX only; StudyDescription decides chest candidacy; SeriesDescription AP/PA decides
    frontal later; CENTRAL LINE excluded; TRAUMA excluded; CODE T / INFANT / OVER PENETRATED
    are NOT excluded on the name alone (patient-level eligibility is judged separately);
    obvious abdomen/KUB studies are exclude_non_chest; no "contains CHEST" auto-rule.

Input : data/interim/cxr_audit/series_level.csv (from scripts/01_cxr_dicom_audit.py)
Output: docs/cxr_studydescription_mapping_draft.csv  (pre-final mapping)

Counting note: per-classification figures are unique counts over the union of that
classification's series (a previous report mistakenly showed the maximum per-row patient
count instead of the union; see docs/change_log.csv CL-001).

Usage:
    python scripts/01b_studydescription_mapping.py --project .
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

CXR_MODALITIES = ["CR", "DX"]
FRONTAL_SERIES_DESC = ["AP", "PA"]

# StudyDescription -> (classification, reason, notes)
MAPPING = {
    "CHEST AP PORT": ("chest_candidate", "胸部のポータブル正面撮影。COVID-19 入院例の標準的な撮影",
                      "1 Series だけ SeriesDescription が LATERAL で、frontal 判定の段階で除かれる"),
    "CHEST AP VIEWONLY": ("chest_candidate", "胸部正面撮影。VIEWONLY は保存・読影形式の区別で、撮影部位は胸部",
                          "PA が 18 Series 含まれる"),
    "CHEST AP PORTABLE": ("chest_candidate", "CHEST AP PORT と同じ内容の別表記", ""),
    "CHEST AP VIEW ONLY": ("chest_candidate", "CHEST AP VIEWONLY の別表記（スペースあり）", ""),
    "CHEST ROUTINE PA AP AND LATERAL": ("chest_candidate", "胸部の定型撮影。frontal は SeriesDescription の AP/PA で選ぶ",
                                        "LATERAL 8 Series を含むため、frontal 判定が必須"),
    "CHEST ROUTINE PA/AP AND LATERAL": ("chest_candidate", "同上（表記違い）", ""),
    "CHEST OVER PENETRATED PA AP PORTABLE": ("chest_candidate",
                                             "胸部正面撮影。透過条件の指定で、部位や方向は変わらない（名前だけを理由に除外しない）",
                                             "撮影条件が他と異なるため、画像の見え方が違う可能性がある"),
    "CODE T CHEST AP PORTABLE": ("chest_candidate",
                                 "胸部の AP 撮影であり、CODE T という名称だけを理由に除外しない（研究者決定 2026-09-20）。TRAUMA CHEST AP PORTABLE とは別に扱う",
                                 "frontal CXR としての適格性で判断する"),
    "CHEST AP INFANT PORTABLE": ("chest_candidate",
                                 "胸部の AP 撮影であり、INFANT という名称だけを理由に除外しない（研究者決定 2026-09-20）",
                                 "患者側の適格性は StudyDescription ではなく患者選定の段階で評価する"),
    "CHEST AP PORT CENTRAL LINE PL": ("exclude_special_purpose", "中心静脈カテーテル留置の確認目的（方針 4）",
                                      "除外対象のなかで最も多い"),
    "CHEST AP CENTRAL LINE PL PORTABLE": ("exclude_special_purpose", "同上（表記違い）", ""),
    "TRAUMA CHEST AP PORTABLE": ("exclude_special_purpose", "外傷対応での撮影（方針 5）", ""),
    "FLAT PLATE OF ABDOMEN PORTABLE": ("exclude_non_chest", "腹部単純撮影", ""),
    "ABD SERIES FLAT AND ERECT PORTABLE": ("exclude_non_chest", "腹部シリーズ（臥位・立位）", ""),
    "ABDOMEN SUPINE KUB": ("exclude_non_chest", "腹部 KUB", ""),
    "ABDOMEN SERIES FLAT AND ERECT": ("exclude_non_chest", "腹部シリーズ", ""),
    "ABDOMEN SERIES FLAT AND ERECT PORTABLE": ("exclude_non_chest", "腹部シリーズ", ""),
    "ABD UPRT VIEWONLY": ("exclude_non_chest", "腹部立位撮影", ""),
    "ABD SUPINE LATERAL": ("exclude_non_chest", "腹部撮影",
                           "Study 名は LATERAL だが SeriesDescription は AP で、両者が一致しない（1 Series）"),
}
CLASS_ORDER = {"chest_candidate": 0, "exclude_special_purpose": 1, "exclude_non_chest": 2, "needs_review": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    project = Path(ap.parse_args().project)
    s = pd.read_csv(project / "data/interim/cxr_audit/series_level.csv", encoding="utf-8-sig", dtype=str)
    c = s[s.Modality.isin(CXR_MODALITIES)].copy()
    c["StudyDescription"] = c.StudyDescription.fillna("<missing>")
    unknown = set(c.StudyDescription) - set(MAPPING)
    assert not unknown, f"unmapped StudyDescription: {unknown}"
    c["classification"] = c.StudyDescription.map(lambda d: MAPPING[d][0])

    rows = []
    for d, g in c.groupby("StudyDescription"):
        cls, reason, notes = MAPPING[d]
        frontal = g[g.SeriesDescription.isin(FRONTAL_SERIES_DESC)]
        rows.append({
            "StudyDescription": d,
            "classification": cls,
            "n_series": g.SeriesInstanceUID.nunique(),
            "n_studies": g.StudyInstanceUID.nunique(),
            "n_patients": g.PatientID.nunique(),
            "n_frontal_series_AP_PA": frontal.SeriesInstanceUID.nunique(),
            "n_frontal_patients": frontal.PatientID.nunique(),
            "series_description_breakdown": "; ".join(
                f"{k}={v}" for k, v in Counter(g.SeriesDescription.fillna("<missing>")).most_common()),
            "modality_breakdown": "; ".join(f"{k}={v}" for k, v in Counter(g.Modality).most_common()),
            "reason": reason,
            "notes": notes,
            "status": "確定前（研究責任者の承認待ち）",
        })
    df = pd.DataFrame(rows).sort_values(
        by=["classification", "n_series"],
        key=lambda col: col.map(CLASS_ORDER) if col.name == "classification" else -col)
    df.to_csv(project / "docs/cxr_studydescription_mapping_draft.csv", index=False, encoding="utf-8-sig")

    print("== per classification (unique counts over the union of that classification)")
    per = c.groupby("classification").agg(
        study_descriptions=("StudyDescription", "nunique"),
        n_series=("SeriesInstanceUID", "nunique"),
        n_studies=("StudyInstanceUID", "nunique"),
        n_patients=("PatientID", "nunique"))
    frontal_all = c[c.SeriesDescription.isin(FRONTAL_SERIES_DESC)]
    per["n_frontal_series"] = frontal_all.groupby("classification")["SeriesInstanceUID"].nunique()
    per["n_frontal_patients"] = frontal_all.groupby("classification")["PatientID"].nunique()
    print(per.to_string())

    print("\n== consistency checks")
    print(f"  rows in mapping table                : {len(df)} (expected 19)")
    print(f"  sum of per-row n_series              : {df.n_series.sum()}")
    print(f"  unique CR/DX series in audit         : {c.SeriesInstanceUID.nunique()}")
    print(f"  sum of per-row n_studies             : {df.n_studies.sum()}")
    print(f"  unique CR/DX studies                 : {c.StudyInstanceUID.nunique()}")
    print(f"  sum of per-row n_patients (overlaps) : {df.n_patients.sum()}")
    print(f"  unique CR/DX patients (union)        : {c.PatientID.nunique()}")
    print(f"  sum of per-class n_patients (overlaps): {per.n_patients.sum()}")
    multi = c.groupby("StudyInstanceUID").StudyDescription.nunique()
    print(f"  studies whose series carry >1 StudyDescription: {int((multi > 1).sum())}")
    chest = c[c.classification == "chest_candidate"]
    only_excluded = set(c.PatientID) - set(chest.PatientID)
    frontal_chest = chest[chest.SeriesDescription.isin(FRONTAL_SERIES_DESC)]
    print(f"  patients with CR/DX but none in chest_candidate: {len(only_excluded)}")
    print(f"  chest_candidate frontal series       : {frontal_chest.SeriesInstanceUID.nunique()}"
          f"  patients: {frontal_chest.PatientID.nunique()}")
    print(f"  patients lost when limiting chest_candidate to AP/PA: "
          f"{chest.PatientID.nunique() - frontal_chest.PatientID.nunique()}")


if __name__ == "__main__":
    main()
