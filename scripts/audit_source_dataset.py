"""Count source identifiers and flag unresolved patient-ID assumptions."""
import argparse
import glob
import json
import os
from pathlib import Path

import pydicom


def audit(base_dir: Path, output: Path):
    tal_dir = base_dir / "ILD_DB_talismanTestSuite"
    source_dirs = {
        "volume_rois": base_dir / "ILD_DB_volumeROIs",
        "text_rois": base_dir / "ILD_DB_txtROIs",
        "lung_masks": base_dir / "ILD_DB_lungMasks",
    }
    tal_ids, sentinels = set(), 0
    for path in tal_dir.glob("*.tif"):
        token = path.stem.split("_patient", 1)[-1]
        if token.startswith("-1_"):
            sentinels += 1
        else:
            try:
                tal_ids.add(int(token))
            except ValueError:
                raise ValueError(f"Unparseable Talisman patient token: {path.name}")
    regular_ids = {}
    pilot_ids = {}
    for name, root in source_dirs.items():
        regular_ids[name] = {int(p.name) for p in root.iterdir()
                             if p.is_dir() and p.name.isdigit()}
        pilot_root = root / "HRCT_pilot"
        pilot_ids[name] = ({int(p.name) for p in pilot_root.iterdir()
                            if p.is_dir() and p.name.isdigit()}
                           if pilot_root.is_dir() else set())
    volume_ids = regular_ids["volume_rois"]
    pilot_volume_ids = pilot_ids["volume_rois"]
    all_volume_ids = volume_ids | pilot_volume_ids
    overlap = tal_ids & all_volume_ids

    # DICOM PatientID is anonymized in the supplied files; do not use names or
    # UIDs as a person mapping. Report this limitation instead of asserting ID
    # namespaces are identical based solely on their numeric spelling.
    sample_dicom = next(iter(source_dirs["volume_rois"].glob("*/*.dcm")), None)
    patient_id_value = None
    if sample_dicom:
        ds = pydicom.dcmread(sample_dicom, stop_before_pixels=True, force=True)
        patient_id_value = str(getattr(ds, "PatientID", ""))
    result = {
        "talisman_real_numeric_ids": sorted(tal_ids),
        "talisman_real_patient_id_count": len(tal_ids),
        "talisman_training_sentinel_patch_count": sentinels,
        "volume_numeric_folder_ids": sorted(volume_ids),
        "volume_folder_id_count": len(volume_ids),
        "hrct_pilot_numeric_ids": sorted(pilot_volume_ids),
        "hrct_pilot_case_count": len(pilot_volume_ids),
        "volume_and_pilot_identifier_count": len(volume_ids | pilot_volume_ids),
        "regular_folder_counts_by_source": {name: len(ids) for name, ids in regular_ids.items()},
        "regular_ids_match_across_three_sources": (
            regular_ids["volume_rois"] == regular_ids["text_rois"] == regular_ids["lung_masks"]
        ),
        "pilot_folder_counts_by_source": {name: len(ids) for name, ids in pilot_ids.items()},
        "pilot_ids_match_across_three_sources": (
            pilot_ids["volume_rois"] == pilot_ids["text_rois"] == pilot_ids["lung_masks"]
        ),
        "numeric_id_overlap": sorted(overlap),
        "numeric_id_overlap_count": len(overlap),
        "numeric_id_union_count_if_shared_namespace": len(tal_ids | all_volume_ids),
        "numeric_namespaces_proven_same_patient": False,
        "clinical_workbook_status": "password-protected; not opened by this audit script",
        "sample_dicom_patient_id_is_anonymized": patient_id_value in ("Anonymous-ID", ""),
        "status": "SHARED_NUMERIC_ID_GROUPING_ENABLED",
        "note": "The pipeline groups matching numeric IDs across Talisman, volume and HRCT pilot sources in one 75/25 patient split; sentinel -1 rows are training-only. Counts describe folder identifiers, not extracted patches.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(Path(args.base_dir), Path(args.output))


if __name__ == "__main__":
    main()
