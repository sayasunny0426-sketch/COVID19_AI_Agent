"""Task B step 4: decide, per candidate variable, whether to add a missing indicator.

The criteria below are fixed BEFORE the per-variable results are read, so that "adopted" and
"not adopted" follow from a rule rather than from whichever variable happened to look best.

Dataset discipline
------------------
Every quantity here is computed on the **Training set only** (n=1,021, 135 deaths).
The Validation set is NOT used for this decision; it is reserved for hyperparameter
selection, model selection and threshold selection. The Test set is not touched at all.

Pre-registered criteria (a variable gets an indicator only if it passes ALL of them)
-----------------------------------------------------------------------------------
C1 coverage      Training missing rate >= 5%. Below that an indicator spends a parameter on
                 too few patients to estimate its coefficient stably (135 events, budget 13).
C2 stratum       The association between missingness and death must persist within the
                 Inpatient stratum. In-hospital death is close to structurally impossible for
                 ER-discharge encounters, so an association that exists only across strata is
                 a property of the care pathway, not of the patient.
C3 panel-held    Within inpatients whose OTHER candidate labs were all measured -- i.e. a full
                 panel was ordered -- the association must survive, at Bonferroni-corrected
                 alpha = 0.05 / (number of variables tested). This removes "a panel was never
                 ordered" as an explanation.
C4 direction     Missingness must be associated with HIGHER mortality. The opposite direction
                 means the test was most likely not ordered because the patient was not sick,
                 which is care-process information that will not transfer to another hospital.
C5 mechanism     A clinical mechanism must be articulable for why the absence of this specific
                 measurement reflects patient state rather than ordering behaviour.

Usage:
    python scripts/24_taskB_missing_indicator_decision.py --project .
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.features import taskB_clinical as tb  # noqa: E402

INPATIENT = "Inpatient Visit"
C1_MIN_MISSING = 0.05
C3_ALPHA = 0.05

# C5: the mechanism has to be stated in advance, per variable, not invented afterwards.
MECHANISM = {
    "bun": ("基礎代謝パネルの一部。単独で省略されることはほとんどなく、"
            "欠測はパネル自体が出ていないことを意味する → 患者状態を反映しない"),
    "egfr": ("クレアチニンからの計算値で、基礎代謝パネルに連動する。"
             "単独欠測は起こりにくい → 患者状態を反映しない"),
    "crp": ("炎症マーカーとして追加オーダーされる。COVID-19 入院例ではほぼ定型的に測定され、"
            "欠測はパネル未オーダーとほぼ同義 → 患者状態を反映しない"),
    "lactate": ("敗血症・低灌流を疑ったときに追加される。"
                "重症を疑わなければ出されない＝欠測は「疑われなかった」ことを意味し、"
                "向きとしては軽症に対応する → 患者状態の代理としては逆向き"),
    "ddimer": ("血栓症を疑ったときに追加される。lactate と同様に、"
               "欠測は「疑われなかった」ことを意味しうる → 逆向き"),
    "troponin": ("心筋傷害を疑ったときに追加される。同上 → 逆向き"),
    "lymph": ("リンパ球数は**白血球分画**を要し、分画は全血球計算とは別に実施・報告される。"
              "基礎パネルが出ていても分画が欠けることがあり、"
              "その欠如が患者状態を反映しうる（下記 C3 で検証）"),
    "sex": ("data dictionary に de-identification の過程で一部削除されると記載がある。"
            "患者の臨床状態ではなく、データ加工に由来する可能性が高い → 指示変数を作らない"),
    "spo2": "バイタルサインでほぼ全例に測定される（欠測 0.3%）",
    "rr": "バイタルサインでほぼ全例に測定される（欠測 0.5%）",
    "sbp": "バイタルサインでほぼ全例に測定される（欠測 0.2%）",
    "comorbidity": ("カルテ抽出ブロックの有無。欠測は NotAbstracted 水準として"
                    "カテゴリ内で明示的に扱うため、別個の指示変数は作らない"),
}


def fisher(missing: np.ndarray, y: np.ndarray) -> tuple[int, int, int, int, float, float]:
    a = int(((missing == 1) & (y == 1)).sum())
    b = int(((missing == 1) & (y == 0)).sum())
    c = int(((missing == 0) & (y == 1)).sum())
    d = int(((missing == 0) & (y == 0)).sum())
    if (a + b) == 0 or (c + d) == 0:
        return a, b, c, d, float("nan"), float("nan")
    or_, p = stats.fisher_exact([[a, b], [c, d]])
    return a, b, c, d, float(or_), float(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--cohort-flow", action="store_true",
                    help="also emit the cohort-level visit-type tables. This READS "
                         "Test metadata (visit type and split counts) and is off by "
                         "default; it was run once, with the researcher's agreement, "
                         "to define the inpatient sensitivity population.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out) if args.out else project / "results/taskB/preprocessing"
    out.mkdir(parents=True, exist_ok=True)

    df = tb.load_development(project)   # Training + Validation only
    train = tb.training_only(df)
    y = train.y.values
    inpat = (train.visit_concept_name == INPATIENT).values
    print(f"Training set only: n={len(train)}, deaths={int(y.sum())}; "
          f"inpatient={int(inpat.sum())} (deaths {int(y[inpat].sum())}), "
          f"ER={int((~inpat).sum())} (deaths {int(y[~inpat].sum())})")
    print("Validation is NOT used for this decision. Test is not touched.\n")

    lab_names = [v.name for v in tb.CANDIDATES
                 if v.domain in ("腎（急性）", "炎症", "血液", "凝固", "組織灌流", "腎（慢性）", "心筋")]
    miss = {}
    for v in tb.CANDIDATES:
        s = tb.comorbidity_count(train) if v.kind == "derived" else train[v.column]
        miss[v.name] = s.isna().values.astype(int)
    n_tested = sum(1 for v in tb.CANDIDATES if miss[v.name].mean() >= C1_MIN_MISSING)
    alpha_c3 = C3_ALPHA / max(n_tested, 1)
    print(f"variables meeting C1 (missing >= {C1_MIN_MISSING:.0%}): {n_tested}"
          f"  ->  Bonferroni alpha for C3 = {alpha_c3:.4f}\n")

    rows = []
    for v in tb.CANDIDATES:
        m = miss[v.name]
        rate = float(m.mean())
        a, b, c, d, or_all, p_all = fisher(m, y)
        c1 = rate >= C1_MIN_MISSING

        # C2 / C4 within the inpatient stratum
        ai, bi, ci, di, or_ip, p_ip = fisher(m[inpat], y[inpat])
        c2 = (ai + bi) >= 10 and p_ip == p_ip and p_ip < 0.05
        c4 = or_ip == or_ip and or_ip > 1.0

        # C3: inpatients whose other candidate labs were all measured
        others = [n for n in lab_names if n != v.name]
        full_panel = inpat & (np.vstack([miss[n] for n in others]).sum(axis=0) == 0) \
            if others else inpat
        af, bf, cf, df_, or_f, p_f = fisher(m[full_panel], y[full_panel])
        c3 = (af + bf) >= 10 and p_f == p_f and p_f < alpha_c3

        adopt = bool(c1 and c2 and c3 and c4)
        fails = [k for k, ok in [("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4)] if not ok]
        rows.append({
            "variable": v.name, "domain": v.domain,
            "missing_rate_train": round(rate, 4),
            "n_missing_train": int(m.sum()),
            "deaths_if_missing": a, "n_missing": a + b,
            "death_rate_if_missing": round(a / (a + b), 4) if a + b else None,
            "deaths_if_observed": c, "n_observed": c + d,
            "death_rate_if_observed": round(c / (c + d), 4) if c + d else None,
            "OR_all": round(or_all, 3) if or_all == or_all else None,
            "p_all": f"{p_all:.3e}" if p_all == p_all else None,
            "C1_coverage>=5%": c1,
            "inpatient_n_missing": ai + bi,
            "inpatient_death_rate_missing": round(ai / (ai + bi), 4) if ai + bi else None,
            "inpatient_death_rate_observed": round(ci / (ci + di), 4) if ci + di else None,
            "inpatient_OR": round(or_ip, 3) if or_ip == or_ip else None,
            "inpatient_p": f"{p_ip:.3e}" if p_ip == p_ip else None,
            "C2_persists_within_inpatient": c2,
            "full_panel_n": int(full_panel.sum()),
            "full_panel_n_missing": af + bf,
            "full_panel_death_rate_missing": round(af / (af + bf), 4) if af + bf else None,
            "full_panel_death_rate_observed": round(cf / (cf + df_), 4) if cf + df_ else None,
            "full_panel_OR": round(or_f, 3) if or_f == or_f else None,
            "full_panel_p": f"{p_f:.4f}" if p_f == p_f else None,
            "C3_survives_panel_held_bonferroni": c3,
            "C4_missing_means_higher_mortality": c4,
            "C5_mechanism": MECHANISM.get(v.name, ""),
            "decision": "adopt missing indicator" if adopt else "do not adopt",
            "criteria_failed": ", ".join(fails) if fails else "none",
            "dataset_used_for_decision": "Training only",
            "status": "fixed（Test 使用前は変更可）",
        })

    dec = pd.DataFrame(rows)
    dec.to_csv(out / "missing_indicator_decision.csv", index=False, encoding="utf-8-sig")

    print("=== per-variable outcome distribution by missingness (Training) ===")
    print(dec[["variable", "missing_rate_train", "death_rate_if_missing",
               "death_rate_if_observed", "OR_all", "p_all"]].to_string(index=False))
    print("\n=== criteria ===")
    print(dec[["variable", "C1_coverage>=5%", "C2_persists_within_inpatient",
               "C3_survives_panel_held_bonferroni", "C4_missing_means_higher_mortality",
               "decision", "criteria_failed"]].to_string(index=False))
    print("\n=== C3 detail: inpatients with a full panel except possibly this variable ===")
    print(dec[["variable", "full_panel_n", "full_panel_n_missing",
               "full_panel_death_rate_missing", "full_panel_death_rate_observed",
               "full_panel_OR", "full_panel_p"]].to_string(index=False))

    adopted = dec.loc[dec.decision == "adopt missing indicator", "variable"].tolist()
    print(f"\nadopted missing indicators: {adopted or 'none'}")

    summary = {
        "generated": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "script": Path(__file__).name,
        "dataset_used_for_decision": "Training only (n=1021, deaths=135)",
        "validation_role": ("この判断には使用しない。Validation はハイパーパラメータ選択・"
                            "モデル選択・閾値決定にのみ用いる"),
        "test_role": "使用しない",
        "criteria": {"C1": f"Training 欠測率 >= {C1_MIN_MISSING:.0%}",
                     "C2": "入院例内でも欠測と死亡の関連が p<0.05 で維持される",
                     "C3": f"フルパネル症例に限定しても Bonferroni alpha={alpha_c3:.4f} で維持される",
                     "C4": "欠測が高い死亡率と関連する（向きの検証）",
                     "C5": "患者状態を反映する臨床的機序が事前に説明できる"},
        "n_variables_tested_for_bonferroni": n_tested,
        "adopted": adopted,
        "not_adopted": dec.loc[dec.decision != "adopt missing indicator", "variable"].tolist(),
    }
    (out / "missing_indicator_decision.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- inpatient sensitivity population flow -------------------------------------
    # Needs the visit type of every split, so it reads Test metadata; off by default.
    if args.cohort_flow:
        clin = pd.read_csv(project / tb.CLINICAL_CSV, dtype=str, encoding="utf-8-sig")
        flow = []
        flow.append({"step": "1. 臨床ファイルの全患者", "n": len(clin), "deaths": None,
                     "note": "患者データファイル.csv（ID 重複なし）"})
        vt = clin.visit_concept_name.value_counts().to_dict()
        for k, n in vt.items():
            flow.append({"step": f"2. うち visit_concept_name = {k}", "n": int(n), "deaths": None,
                         "note": "受診形態の内訳（臨床ファイル全体）"})
        flow.append({"step": "3. CXR 選定後の固定コホート", "n": len(df),
                     "deaths": int(df.y.sum()),
                     "note": "Task A の index CXR 選定規則（T0-2 日〜T0 当日）を満たした患者。"
                             "臨床ファイルの 1,384 例のうち 107 例が CXR 条件を満たさず対象外"})
        ip_all = int((df.visit_concept_name == INPATIENT).sum())
        flow.append({"step": "4. 固定コホート ∩ Inpatient Visit（感度分析の対象）",
                     "n": ip_all, "deaths": int(df[df.visit_concept_name == INPATIENT].y.sum()),
                     "note": "感度分析集団"})
        flow.append({"step": "5. 固定コホートのうち Emergency Room Visit（感度分析で除外）",
                     "n": int((df.visit_concept_name != INPATIENT).sum()),
                     "deaths": int(df[df.visit_concept_name != INPATIENT].y.sum()),
                     "note": "ED で評価され帰宅した患者。Training では在院日数 中央値 1 日"})
        fl = pd.DataFrame(flow)
        fl.to_csv(out / "inpatient_sensitivity_flow.csv", index=False, encoding="utf-8-sig")
        print("\n=== inpatient sensitivity population flow ===")
        print(fl.to_string(index=False))

        ip_clin = int((clin.visit_concept_name == INPATIENT).sum())
        print(f"\n1,384 例中 Inpatient Visit: {ip_clin}")
        print(f"固定コホート 1,277 との intersection: {ip_all} "
              f"(= {ip_clin} - {ip_clin - ip_all} 例が CXR 条件を満たさず固定コホート外)")
    else:
        print("\n[cohort-flow skipped: Test metadata not read. "
              "Re-run with --cohort-flow if the cohort-level flow table is needed.]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
