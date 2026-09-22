"""Build the Colab code bundle as a ZIP with POSIX entry names, and verify it.

Why this script exists: PowerShell's Compress-Archive writes Windows path separators into
the archive ("src\\covid_mortality\\..."). Python's zipfile hides that on Windows (it maps
os.sep to "/" when reading) but on Linux the backslash stays inside the file NAME, so Colab
extracted one flat file called "src\\covid_mortality\\training\\taskA_trainer.py".
Here every arcname is written with Path.as_posix(), and the result is verified by reading the
raw central-directory names (orig_filename), which is what a Linux extractor sees.

The notebook is deliberately NOT included: it carries the expected SHA256 of this zip, so
including it would make the hash self-referential.

Usage:
    python scripts/11_build_code_bundle.py --project . --out outputs/bundles/AIagent_code_YYYYMMDD.zip
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

INCLUDE_DIRS = ["src", "scripts", "tests", "docs", "notebooks"]
INCLUDE_FILES = ["data/splits/COVID19_固定患者split_1277.csv",
                 "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv"]
REQUIRED = ["src/covid_mortality/training/taskA_trainer.py",
            "src/covid_mortality/evaluation/metrics.py",
            "src/covid_mortality/evaluation/gradcam.py",
            "src/covid_mortality/data/taskA_dataset.py",
            "src/covid_mortality/models/taskA_resnet18.py",
            "scripts/09_taskA_run_training.py",
            "scripts/18_taskA_gradcam_primary_test.py",
            "notebooks/TaskA_gradcam_colab.ipynb",
            "data/splits/COVID19_固定患者split_1277.csv",
            "data/interim/cxr_audit/index_cxr_manifest_window_T0m2_T0.csv"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(project: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for d in INCLUDE_DIRS:
        for p in sorted((project / d).rglob("*")):
            if (p.is_file() and not set(p.parts) & SKIP_PARTS and p.suffix not in SKIP_SUFFIX):
                items.append((p, p.relative_to(project).as_posix()))  # POSIX arcname
    for f in INCLUDE_FILES:
        p = project / f
        if not p.exists():
            raise FileNotFoundError(p)
        items.append((p, Path(f).as_posix()))
    return items


def verify(zip_path: Path) -> dict:
    """Verify using the RAW names stored in the archive (what Linux sees)."""
    report: dict = {"zip": str(zip_path), "sha256": sha256(zip_path)}
    with zipfile.ZipFile(zip_path) as z:
        raw = [i.orig_filename for i in z.infolist()]
        names = [i.filename for i in z.infolist()]
        non_ascii = [i for i in z.infolist() if any(ord(c) > 127 for c in i.filename)]
        report["n_entries"] = len(raw)
        report["entries_with_backslash"] = [n for n in raw if "\\" in n]
        report["all_posix"] = not report["entries_with_backslash"]
        report["non_ascii_entries"] = [{"name": i.filename, "utf8_flag": bool(i.flag_bits & 0x800)}
                                       for i in non_ascii]
        report["utf8_flag_on_all_non_ascii"] = all(bool(i.flag_bits & 0x800) for i in non_ascii)
        report["required_present"] = {r: (r in names and r in raw) for r in REQUIRED}
        report["missing_required"] = [r for r, ok in report["required_present"].items() if not ok]
        # extract into a temp dir and check the tree exists as real directories
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "code"
            z.extractall(dest)
            report["extracted_required_exist"] = {r: (dest / r).exists() for r in REQUIRED}
            report["extracted_missing"] = [r for r, ok in report["extracted_required_exist"].items()
                                           if not ok]
            # a Linux extractor splits only on "/": emulate that from the raw names
            report["linux_paths"] = [n for n in raw][:5]
            report["flat_files_with_backslash_after_extract"] = [
                p.name for p in dest.rglob("*") if "\\" in p.name]
    report["passed"] = (report["all_posix"] and not report["missing_required"]
                        and not report["extracted_missing"]
                        and report["utf8_flag_on_all_non_ascii"]
                        and not report["flat_files_with_backslash_after_extract"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    project = Path(args.project).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    items = collect(project)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arcname in items:
            assert "\\" not in arcname, arcname
            z.write(src, arcname=arcname)
    print(f"wrote {out} with {len(items)} files ({out.stat().st_size / 1024:.0f} KB)")

    report = verify(out)
    manifest = {"bundle": out.name, "sha256": report["sha256"], "n_entries": report["n_entries"],
                "required": REQUIRED, "notebook_included": False,
                "verification": {k: report[k] for k in
                                 ["all_posix", "utf8_flag_on_all_non_ascii", "missing_required",
                                  "extracted_missing", "flat_files_with_backslash_after_extract",
                                  "passed"]}}
    (out.parent / f"{out.stem}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not report["passed"]:
        print("VERIFICATION FAILED")
        return 1
    print(f"\nSHA256: {report['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
