"""QC for the Task A Dataset / DataLoader. Runs every check required before training.

Checks (researcher's list, 2026-09-20):
  dataset size, per-split counts, per-split deaths, PatientID duplicates, cross-split overlap,
  missing images, label agreement, image read errors, tensor shape, dtype, value range,
  augmentation applied to Train only, Validation / Test deterministic, repeated reads of the
  same image identical, and the tensor handed to ImageNet-pretrained ResNet18 (shape, range,
  normalisation) is as intended.

Exit code 1 if any check fails.

Usage:
    python scripts/07_taskA_dataset_qc.py --project . [--full-read]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covid_mortality.data.taskA_dataset import (  # noqa: E402
    AUG_A, AUG_B, AUG_NONE, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, TaskACXRDataset,
    build_loader, describe_config, load_index)

CACHE = "data/processed/taskA_png512_16bit/images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--full-read", action="store_true",
                    help="decode every image once (slower, verifies all files)")
    args = ap.parse_args()
    project = Path(args.project)
    image_dir = project / CACHE
    results: dict = {"config": describe_config()}
    failures: list[str] = []

    def check(name: str, ok: bool, detail=None):
        results[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            failures.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    index = load_index(project)  # raises if the fixed numbers changed
    sets = {sp: TaskACXRDataset(index, image_dir, sp) for sp in ("train", "val", "test")}

    # --- cohort / split integrity -------------------------------------------------
    check("dataset_size", len(index) == 1277, {"n": len(index)})
    counts = {sp: len(ds) for sp, ds in sets.items()}
    check("split_counts", counts == {"train": 1021, "val": 128, "test": 128}, counts)
    deaths = {sp: int(ds.labels().sum()) for sp, ds in sets.items()}
    check("split_deaths", deaths == {"train": 135, "val": 17, "test": 17}, deaths)
    ids = {sp: set(ds.patient_ids()) for sp, ds in sets.items()}
    check("patient_id_unique", index.PatientID.is_unique and sum(len(v) for v in ids.values()) == 1277,
          {"unique_ids": index.PatientID.nunique()})
    overlaps = {f"{a}&{b}": len(ids[a] & ids[b]) for a, b in
                (("train", "val"), ("train", "test"), ("val", "test"))}
    check("no_cross_split_overlap", all(v == 0 for v in overlaps.values()), overlaps)
    check("one_image_per_patient",
          index.SOPInstanceUID.nunique() == 1277 and index.PatientID.nunique() == 1277,
          {"distinct_SOPInstanceUID": int(index.SOPInstanceUID.nunique())})

    # --- images present and readable ---------------------------------------------
    missing = [p for p in index.PatientID if not (image_dir / f"{p}.png").exists()]
    check("no_missing_images", not missing, {"missing": len(missing)})
    read_errors, dtypes, shapes, vmax = [], set(), set(), []
    to_read = range(len(sets["train"])) if args.full_read else range(0, len(sets["train"]), 25)
    for sp, ds in sets.items():
        rng = range(len(ds)) if (args.full_read or sp != "train") else to_read
        for i in rng:
            try:
                arr = ds.read_raw(i)
                dtypes.add(str(arr.dtype)); shapes.add(arr.shape); vmax.append(int(arr.max()))
            except Exception as e:  # noqa: BLE001
                read_errors.append(f"{sp}:{ds.df.PatientID.iloc[i]}:{type(e).__name__}")
    check("image_read_errors", not read_errors, {"errors": len(read_errors), "examples": read_errors[:3]})
    check("png_dtype_uint16", dtypes == {"uint16"}, sorted(dtypes))
    check("png_shape_512", shapes == {(512, 512)}, [str(s) for s in shapes])
    check("png_scaling_range", 0 < min(vmax) and max(vmax) <= 65535,
          {"min_of_max": int(min(vmax)), "max_of_max": int(max(vmax))})

    # --- labels ------------------------------------------------------------------
    label_map = {"deceased": 1, "discharged": 0}
    bad_labels = index[index.label != index["last.status"].map(label_map)]
    check("label_consistency", len(bad_labels) == 0, {"mismatches": int(len(bad_labels))})

    # --- tensor contract ----------------------------------------------------------
    sample = sets["val"][0]
    img = sample["image"]
    check("tensor_shape", tuple(img.shape) == (3, INPUT_SIZE, INPUT_SIZE), list(img.shape))
    check("tensor_dtype", img.dtype == torch.float32, str(img.dtype))
    check("label_dtype", sample["label"].dtype == torch.float32, str(sample["label"].dtype))
    check("three_channels_identical",
          torch.allclose(img[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0],
                         img[1] * IMAGENET_STD[1] + IMAGENET_MEAN[1], atol=1e-6),
          "channels equal after de-normalisation")
    denorm = img * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    check("denormalised_range_0_1", float(denorm.min()) >= -1e-4 and float(denorm.max()) <= 1 + 1e-4,
          {"min": round(float(denorm.min()), 5), "max": round(float(denorm.max()), 5)})
    expected_min = min((0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD))
    expected_max = max((1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD))
    check("normalised_range_matches_imagenet",
          expected_min - 1e-3 <= float(img.min()) and float(img.max()) <= expected_max + 1e-3,
          {"observed": [round(float(img.min()), 3), round(float(img.max()), 3)],
           "possible": [round(expected_min, 3), round(expected_max, 3)]})

    # --- determinism / augmentation ----------------------------------------------
    same = all(torch.equal(sets[sp][i]["image"], sets[sp][i]["image"])
               for sp in ("val", "test") for i in (0, 5, 17))
    check("val_test_repeat_read_identical", same, "two reads of the same index are bit-identical")
    train_plain = TaskACXRDataset(index, image_dir, "train", AUG_NONE)
    check("train_without_aug_deterministic",
          torch.equal(train_plain[3]["image"], train_plain[3]["image"]), "no randomness when aug=none")
    aug_diff = {}
    for aug in (AUG_A, AUG_B):
        ds_aug = TaskACXRDataset(index, image_dir, "train", aug)
        torch.manual_seed(0); a = ds_aug[3]["image"]
        torch.manual_seed(1); b = ds_aug[3]["image"]
        aug_diff[aug.name] = float((a - b).abs().max())
    check("train_augmentation_active", all(v > 0 for v in aug_diff.values()), aug_diff)
    for sp in ("val", "test"):
        try:
            TaskACXRDataset(index, image_dir, sp, AUG_A)
            check(f"augmentation_blocked_for_{sp}", False, "constructor did NOT raise")
        except ValueError:
            check(f"augmentation_blocked_for_{sp}", True, "constructor raises ValueError")

    # --- loaders ------------------------------------------------------------------
    val_loader = build_loader(sets["val"], batch_size=16, seed=42)
    order1 = [pid for b in val_loader for pid in b["patient_id"]]
    order2 = [pid for b in build_loader(sets["val"], batch_size=16, seed=7) for pid in b["patient_id"]]
    check("val_loader_order_deterministic", order1 == order2 == sorted(sets["val"].patient_ids()),
          {"n": len(order1)})
    t1 = [pid for b in build_loader(sets["train"], batch_size=32, seed=42) for pid in b["patient_id"]]
    t2 = [pid for b in build_loader(sets["train"], batch_size=32, seed=42) for pid in b["patient_id"]]
    t3 = [pid for b in build_loader(sets["train"], batch_size=32, seed=43) for pid in b["patient_id"]]
    check("train_loader_shuffles_reproducibly", t1 == t2 and t1 != t3 and sorted(t1) == sorted(t3),
          {"same_seed_identical": t1 == t2, "different_seed_differs": t1 != t3})
    batch = next(iter(val_loader))
    check("batch_shape", tuple(batch["image"].shape) == (16, 3, INPUT_SIZE, INPUT_SIZE),
          list(batch["image"].shape))
    check("loader_covers_all_patients",
          set(order1) == set(sets["val"].patient_ids()), {"n_unique": len(set(order1))})

    # --- class imbalance (Train only) --------------------------------------------
    pw = sets["train"].pos_weight()
    check("pos_weight_from_train_only", abs(pw - (1021 - 135) / 135) < 1e-9, {"pos_weight": round(pw, 4)})
    try:
        sets["val"].pos_weight()
        check("pos_weight_blocked_outside_train", False, "did NOT raise")
    except ValueError:
        check("pos_weight_blocked_outside_train", True, "raises ValueError")

    results["summary"] = {"checks": len(results) - 1, "failures": failures}
    out = project / "data/interim/taskA_dataset_qc.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n{len(failures)} failure(s). report -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
