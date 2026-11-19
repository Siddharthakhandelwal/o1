r"""
Step 2: Unified Patch Extraction, HU Standardization & Dataset Packaging
=========================================================================
This script:
  1. Loads all Talisman pre-extracted 32x32 patches
  2. Extracts additional patches from volume ROIs using DICOM + masks
  3. Standardizes all patches to HU (clip to [-1024, 600], normalize to [0, 1])
  4. Computes radiomics features per patch (mean, std, skewness, kurtosis, entropy)
  5. Packages everything into a single .npz file for downstream use

Runs entirely on LOCAL CPU — no GPU required.
Output: E:\D\ild-folder\preprocessing\outputs\ild_dataset_final.npz
"""

import os
import sys
import glob
import json
import numpy as np
import pydicom
import tifffile
from collections import Counter, defaultdict
from pathlib import Path
from scipy import stats as scipy_stats
from skimage.draw import polygon as ski_polygon
from skimage.feature import graycomatrix, graycoprops
import warnings
import argparse
import re
import hashlib
import tempfile
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"./ILD_DB"
TALISMAN_DIR = os.path.join(BASE_DIR, "ILD_DB_talismanTestSuite")
VOLUME_ROIS_DIR = os.path.join(BASE_DIR, "ILD_DB_volumeROIs")
TXT_ROIS_DIR = os.path.join(BASE_DIR, "ILD_DB_txtROIs")
LUNG_MASKS_DIR = os.path.join(BASE_DIR, "ILD_DB_lungMasks")
OUTPUT_DIR = r"./outputs"

# HU Standardization parameters (lung window)
HU_MIN = -1024  # Air
HU_MAX = 600    # Bone boundary
PATCH_SIZE = 32
PATCH_STRIDE = 32

# Label encoding for the 5 core Talisman classes
CORE_LABELS = {
    "healthy": 0,
    "emphysema": 1,
    "ground_glass": 2,
    "fibrosis": 3,
    "micronodules": 4,
}

# Extended labels for volume ROI masks (17 classes)
EXTENDED_LABELS = {
    1: "healthy", 2: "emphysema", 3: "ground_glass", 4: "fibrosis",
    5: "micronodules", 6: "consolidation", 7: "bronchial_wall_thickening",
    8: "reticulation", 9: "macronodules", 10: "cysts",
    11: "peripheral_micronodules", 12: "bronchiectasis", 13: "air_trapping",
    14: "early_fibrosis", 15: "increased_attenuation", 16: "tuberculosis",
    17: "pcp"
}

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def hu_normalize(patch, hu_min=HU_MIN, hu_max=HU_MAX):
    """Clip to HU range and normalize to [0, 1]."""
    patch = np.clip(patch.astype(np.float32), hu_min, hu_max)
    patch = (patch - hu_min) / (hu_max - hu_min)
    return patch


def compute_radiomics(patch):
    """Compute basic radiomics features from a normalized patch.
    
    Returns a feature vector of length 8:
      [mean, std, skewness, kurtosis, energy, entropy,
       glcm_contrast, glcm_homogeneity]
    """
    flat = patch.flatten()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    skew_val = float(scipy_stats.skew(flat))
    kurt_val = float(scipy_stats.kurtosis(flat))
    energy = float(np.sum(flat ** 2))
    
    # Entropy (histogram-based)
    hist, _ = np.histogram(flat, bins=64, range=(0, 1))
    hist = hist / hist.sum()
    hist = hist[hist > 0]  # Remove zeros for log
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    # GLCM features (on quantized 8-bit patch)
    patch_uint8 = (patch * 255).astype(np.uint8)
    try:
        glcm = graycomatrix(patch_uint8, distances=[1], angles=[0],
                           levels=256, symmetric=True, normed=True)
        contrast = float(graycoprops(glcm, 'contrast')[0, 0])
        homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    except Exception:
        contrast = 0.0
        homogeneity = 0.0
    
    return np.array([mean_val, std_val, skew_val, kurt_val,
                     energy, entropy, contrast, homogeneity], dtype=np.float32)


RADIOMICS_NAMES = [
    "mean", "std", "skewness", "kurtosis",
    "energy", "entropy", "glcm_contrast", "glcm_homogeneity"
]

# ============================================================
# 1. LOAD TALISMAN PATCHES
# ============================================================
def load_talisman_patches(limit=0):
    """Load all 32x32 pre-extracted Talisman patches."""
    print("=" * 60)
    print("1. LOADING TALISMAN PATCHES")
    print("=" * 60)
    
    tif_files = sorted(glob.glob(os.path.join(TALISMAN_DIR, "*.tif")))
    
    patches = []
    labels = []
    patient_ids = []
    sources = []  # 'talisman' or 'volume'
    extended_labels = []
    source_paths = []
    slice_ids = []
    crop_rows = []
    crop_cols = []
    
    skipped = 0
    if limit > 0:
        tif_files = tif_files[:limit]
    for f in tif_files:
        basename = os.path.basename(f)
        parts = basename.replace(".tif", "").split("_patch")
        class_name = parts[0]
        rest = parts[1]
        patient_part = rest.split("_patient")[1]
        
        # Skip classes not in core 5
        if class_name not in CORE_LABELS:
            skipped += 1
            continue
        
        # Parse patient ID
        if patient_part.startswith("-1"):
            pid = -1  # Training-only patches
        else:
            pid = int(patient_part)
        
        # Load patch
        patch = tifffile.imread(f).astype(np.float32)
        
        # Verify shape
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            skipped += 1
            continue
        
        patches.append(patch)
        labels.append(CORE_LABELS[class_name])
        patient_ids.append(pid)
        sources.append("talisman")
        extended_labels.append(class_name)
        source_paths.append(os.path.relpath(f, BASE_DIR))
        slice_ids.append(-1)
        crop_rows.append(-1)
        crop_cols.append(-1)
    
    print(f"Loaded {len(patches)} Talisman patches (skipped {skipped})")
    print(f"Class distribution: {Counter(labels)}")
    
    return patches, labels, patient_ids, sources, extended_labels, source_paths, slice_ids, crop_rows, crop_cols


def _slice_number(path):
    """Return final numeric token used by this dataset's CT/mask filenames."""
    match = re.search(r"(\d+)(?=\.dcm$)", os.path.basename(path), re.IGNORECASE)
    return int(match.group(1)) if match else None


# ============================================================
# 2. EXTRACT PATCHES FROM VOLUME ROIs
# ============================================================
def extract_volume_patches(patient_limit=0):
    """Extract stride-32 patches from each numeric patient and HRCT pilot case."""
    print("\n" + "=" * 60)
    print("2. EXTRACTING PATCHES FROM VOLUME ROIs")
    print("=" * 60)
    patient_entries = []
    for child in sorted(Path(VOLUME_ROIS_DIR).iterdir()):
        if child.is_dir() and child.name.isdigit():
            patient_entries.append((child, int(child.name)))
        elif child.is_dir() and child.name.lower() == "hrct_pilot":
            for pilot_case in sorted(child.iterdir()):
                if pilot_case.is_dir() and pilot_case.name.isdigit():
                    patient_entries.append((pilot_case, int(pilot_case.name)))
    if patient_limit > 0:
        patient_entries = patient_entries[:patient_limit]

    patches, labels, patient_ids, sources = [], [], [], []
    extended, source_paths, slice_ids, crop_rows, crop_cols = [], [], [], [], []
    read_errors = 0
    for patient_path, patient_id in patient_entries:
        mask_dirs = sorted(patient_path.rglob("roi_mask"))
        for mask_dir in mask_dirs:
            series_dir = mask_dir.parent
            dcm_files = sorted(series_dir.glob("*.dcm"))
            mask_files = sorted(mask_dir.glob("*.dcm"))
            dcm_by_slice = {_slice_number(f): f for f in dcm_files if _slice_number(f) is not None}
            mask_by_slice = {_slice_number(f): f for f in mask_files if _slice_number(f) is not None}
            common_slices = sorted(set(dcm_by_slice) & set(mask_by_slice))
            if len(common_slices) != len(dcm_files) or len(common_slices) != len(mask_files):
                print(f"  [WARN] {patient_path.name}/{series_dir.name}: matched "
                      f"{len(common_slices)} image/mask pairs ({len(dcm_files)} CT, {len(mask_files)} masks)")
            for slice_id in common_slices:
                dcm_file, mask_file = dcm_by_slice[slice_id], mask_by_slice[slice_id]
                try:
                    ds = pydicom.dcmread(str(dcm_file), force=True)
                    mask_ds = pydicom.dcmread(str(mask_file), force=True)
                    pixel_array = ds.pixel_array.astype(np.float32)
                    hu_image = pixel_array * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
                    mask = mask_ds.pixel_array
                    if mask.shape != hu_image.shape:
                        read_errors += 1
                        continue
                    for label_val in np.unique(mask):
                        if label_val == 0:
                            continue
                        label_name = EXTENDED_LABELS.get(int(label_val))
                        if label_name is None:
                            continue
                        core_label = CORE_LABELS.get(label_name, -1)
                        roi_mask = mask == label_val
                        rows_any, cols_any = np.any(roi_mask, axis=1), np.any(roi_mask, axis=0)
                        if not rows_any.any() or not cols_any.any():
                            continue
                        rmin, rmax = np.where(rows_any)[0][[0, -1]]
                        cmin, cmax = np.where(cols_any)[0][[0, -1]]
                        for r in range(rmin, rmax - PATCH_SIZE + 2, PATCH_STRIDE):
                            for c in range(cmin, cmax - PATCH_SIZE + 2, PATCH_STRIDE):
                                patch = hu_image[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                                mask_patch = roi_mask[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
                                if patch.shape != (PATCH_SIZE, PATCH_SIZE):
                                    continue
                                if np.mean(mask_patch) < 0.5:
                                    continue
                                patches.append(patch)
                                labels.append(core_label)
                                patient_ids.append(patient_id)
                                sources.append("volume")
                                extended.append(label_name)
                                source_paths.append(os.path.relpath(str(dcm_file), BASE_DIR))
                                slice_ids.append(int(slice_id))
                                crop_rows.append(int(r))
                                crop_cols.append(int(c))
                except Exception:
                    read_errors += 1
                    continue
    print(f"Extracted {len(patches)} patches across {len(patient_entries)} patient cases")
    if read_errors:
        print(f"[WARN] Skipped {read_errors} DICOM/mask pairs due to decode or shape errors")
    print(f"Extended label distribution: {Counter(extended)}")
    return (patches, labels, patient_ids, sources, extended,
            source_paths, slice_ids, crop_rows, crop_cols)


# ============================================================
# 3. STANDARDIZE AND PACKAGE
# ============================================================
def _package_subset(patches, labels, patient_ids, sources, extended_labels,
                   source_paths, slice_ids, crop_rows, crop_cols):
    if not patches:
        raise ValueError("Cannot package an empty dataset subset")
    raw = np.asarray(patches, dtype=np.float32)
    normalized = np.clip(raw, HU_MIN, HU_MAX)
    normalized = (normalized - HU_MIN) / (HU_MAX - HU_MIN)
    radiomics = np.stack([compute_radiomics(patch) for patch in normalized]).astype(np.float32)
    radiomics = np.nan_to_num(radiomics, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "patches": normalized.astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int32),
        "patient_ids": np.asarray(patient_ids, dtype=np.int32),
        "sources": np.asarray(sources),
        "extended_labels": np.asarray(extended_labels),
        "source_paths": np.asarray(source_paths),
        "slice_ids": np.asarray(slice_ids, dtype=np.int32),
        "crop_rows": np.asarray(crop_rows, dtype=np.int32),
        "crop_cols": np.asarray(crop_cols, dtype=np.int32),
        "radiomics": radiomics,
        "radiomics_names": np.asarray(RADIOMICS_NAMES),
        "core_label_names": np.asarray(list(CORE_LABELS.keys())),
        "extended_label_names": np.asarray(list(EXTENDED_LABELS.values())),
    }


def _atomic_save_npz(path, arrays):
    temp_path = path + ".tmp.npz"
    np.savez_compressed(temp_path, **arrays)
    os.replace(temp_path, path)


def standardize_and_package(include_talisman_sentinel_train=False,
                            patient_limit=0, talisman_limit=0):
    """Build separate train (volume/pilot) and held-out Talisman NPZ files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tal = load_talisman_patches(limit=talisman_limit)
    vol = extract_volume_patches(patient_limit=patient_limit)
    (tal_patches, tal_labels, tal_pids, tal_sources, tal_ext,
     tal_paths, tal_slices, tal_rows, tal_cols) = tal
    (vol_patches, vol_labels, vol_pids, vol_sources, vol_ext,
     vol_paths, vol_slices, vol_rows, vol_cols) = vol

    # Volume/pilot cases are the training cohort. Talisman is validation-only,
    # but Talisman patients whose IDs also occur in training must be removed
    # from validation to preserve patient-disjoint evaluation.
    volume_patient_ids = set(int(x) for x in vol_pids)
    tal_real_mask = np.asarray(tal_pids) >= 0
    tal_real_ids = set(int(x) for x in np.asarray(tal_pids)[tal_real_mask])
    overlap_ids = tal_real_ids & volume_patient_ids
    val_mask = np.asarray([
        bool(is_real) and int(pid) not in volume_patient_ids
        for is_real, pid in zip(tal_real_mask, tal_pids)
    ], dtype=bool)

    train_rows = [list(x) for x in zip(vol_patches, vol_labels, vol_pids, vol_sources,
                                       vol_ext, vol_paths, vol_slices, vol_rows, vol_cols)]
    sentinel_count = sum(int(pid) == -1 for pid in tal_pids)
    if include_talisman_sentinel_train:
        train_rows.extend([list(row) for row, pid in zip(zip(tal_patches, tal_labels, tal_pids,
            tal_sources, tal_ext, tal_paths, tal_slices, tal_rows, tal_cols), tal_pids) if int(pid) == -1])

    def unzip_rows(rows):
        return tuple([row[i] for row in rows] for i in range(9))
    train_tuple = unzip_rows(train_rows)
    tal_rows_all = list(zip(tal_patches, tal_labels, tal_pids, tal_sources, tal_ext,
                            tal_paths, tal_slices, tal_rows, tal_cols))
    val_tuple = tuple([row[field] for index, row in enumerate(tal_rows_all) if val_mask[index]]
                      for field in range(9))
    # The comprehension above selects one field at a time to avoid a second
    # full-size copy of the TIFF patch list.
    train_arrays = _package_subset(*train_tuple)
    validation_arrays = _package_subset(*val_tuple)

    train_path = os.path.join(OUTPUT_DIR, "ild_train_dataset.npz")
    validation_path = os.path.join(OUTPUT_DIR, "ild_talisman_validation.npz")
    _atomic_save_npz(train_path, train_arrays)
    _atomic_save_npz(validation_path, validation_arrays)

    train_core = train_arrays["labels"]
    val_core = validation_arrays["labels"]
    summary = {
        "train_file": train_path,
        "validation_file": validation_path,
        "train_patches": int(len(train_core)),
        "validation_talisman_patches": int(len(val_core)),
        "volume_patches_excluded_for_talisman_patient_overlap": 0,
        "talisman_patient_ids_excluded_from_validation_for_training_overlap": int(len(overlap_ids)),
        "talisman_validation_patient_ids_retained": int(len(set(validation_arrays["patient_ids"].tolist()))),
        "candidate_volume_numeric_patient_ids": int(len(volume_patient_ids)),
        "numeric_ids_overlapping_validation_and_volume": int(len(overlap_ids)),
        "train_numeric_patient_ids": int(len(set(train_arrays["patient_ids"].tolist()) - {-1})),
        "talisman_training_only_sentinel_patches_excluded": int(sentinel_count if not include_talisman_sentinel_train else 0),
        "talisman_training_only_sentinel_patches_included": int(sentinel_count if include_talisman_sentinel_train else 0),
        "include_talisman_sentinel_train": bool(include_talisman_sentinel_train),
        "patch_shape": [PATCH_SIZE, PATCH_SIZE],
        "patch_stride": PATCH_STRIDE,
        "hu_window": [HU_MIN, HU_MAX],
        "train_extended_label_counts": {name: int(np.sum(train_arrays["extended_labels"] == name)) for name in EXTENDED_LABELS.values()},
        "validation_class_counts": {name: int(np.sum(val_core == idx)) for name, idx in CORE_LABELS.items()},
        "patient_overlap_policy": "keep all volume and HRCT pilot cases for training; exclude matching numeric IDs from Talisman validation",
        "patient_count_note": "Counts are source identifiers; confirm the stated cohort count separately.",
    }
    summary_path = os.path.join(OUTPUT_DIR, "dataset_summary.json")
    with open(summary_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    os.replace(summary_path + ".tmp", summary_path)
    print(f"Training NPZ: {train_path} ({summary['train_patches']} patches)")
    print(f"Talisman validation NPZ: {validation_path} ({summary['validation_talisman_patches']} patches)")
    print(f"Excluded {len(overlap_ids)} matching patient IDs from Talisman validation")
    return summary


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and package ILD patch dataset")
    parser.add_argument("--base_dir", default=BASE_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--include_talisman_sentinel_train", action="store_true",
                        help="Include patient -1 sentinel patches in training (off by default)")
    parser.add_argument("--patient_limit", type=int, default=0,
                        help="Process only the first N volume cases; intended for smoke checks")
    parser.add_argument("--talisman_limit", type=int, default=0,
                        help="Read only the first N Talisman TIFFs; intended for smoke checks")
    args = parser.parse_args()
    if args.stride != PATCH_SIZE:
        raise ValueError("Paper-aligned non-overlapping patches require --stride 32")
    BASE_DIR = args.base_dir
    TALISMAN_DIR = os.path.join(BASE_DIR, "ILD_DB_talismanTestSuite")
    VOLUME_ROIS_DIR = os.path.join(BASE_DIR, "ILD_DB_volumeROIs")
    OUTPUT_DIR = args.output_dir
    PATCH_STRIDE = args.stride
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("ILD Dataset — Step 2: Patch Extraction & Packaging")
    print("=" * 60)
    summary = standardize_and_package(args.include_talisman_sentinel_train,
                                      args.patient_limit, args.talisman_limit)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Training patches: {summary['train_patches']}")
    print(f"Validation patches: {summary['validation_talisman_patches']}")
