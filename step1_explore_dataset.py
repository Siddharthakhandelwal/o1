r"""
Step 1: ILD Dataset Exploration & Statistics
=============================================
This script explores the complete ILD dataset structure, loads sample data
from each modality, and generates comprehensive statistics for the research.

Runs entirely on LOCAL CPU — no GPU required.
"""

import os
import sys
import glob
import re
import json
import numpy as np
import pandas as pd
import tifffile
from collections import Counter, defaultdict
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r"./ILD_DB"
TALISMAN_DIR = os.path.join(BASE_DIR, "ILD_DB_talismanTestSuite")
VOLUME_ROIS_DIR = os.path.join(BASE_DIR, "ILD_DB_volumeROIs")
TXT_ROIS_DIR = os.path.join(BASE_DIR, "ILD_DB_txtROIs")
LUNG_MASKS_DIR = os.path.join(BASE_DIR, "ILD_DB_lungMasks")
CLINICAL_FILE = os.path.join(BASE_DIR, "clinicalParameters.xlsx")
OUTPUT_DIR = r"./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label mapping from README (volume ROI mask values)
LABEL_MAP = {
    1: "healthy", 2: "emphysema", 3: "ground_glass", 4: "fibrosis",
    5: "micronodules", 6: "consolidation", 7: "bronchial_wall_thickening",
    8: "reticulation", 9: "macronodules", 10: "cysts",
    11: "peripheral_micronodules", 12: "bronchiectasis", 13: "air_trapping",
    14: "early_fibrosis", 15: "increased_attenuation", 16: "tuberculosis",
    17: "pcp"
}

# ============================================================
# 1. TALISMAN TEST SUITE ANALYSIS
# ============================================================
def analyze_talisman():
    """Analyze the pre-extracted 32x32 Talisman Test Suite patches."""
    print("=" * 60)
    print("1. TALISMAN TEST SUITE ANALYSIS")
    print("=" * 60)

    tif_files = glob.glob(os.path.join(TALISMAN_DIR, "*.tif"))
    print(f"Total .tif files: {len(tif_files)}")

    # Parse filename pattern: {class}_patch{id}_patient{patient_id}.tif
    class_counts = Counter()
    patient_classes = defaultdict(set)
    test_patches = []  # patient IDs without -1 prefix
    train_only_patches = []  # patient IDs with -1 prefix (training only)

    for f in tif_files:
        basename = os.path.basename(f)
        # Pattern: classname_patchNNN_patientXXX.tif or classname_patchNNN_patient-1_YYY.tif
        parts = basename.replace(".tif", "").split("_patch")
        class_name = parts[0]
        rest = parts[1]  # e.g., "100_patient50" or "408_patient-1_3"
        
        patient_part = rest.split("_patient")[1]
        
        if patient_part.startswith("-1"):
            train_only_patches.append((class_name, basename))
        else:
            test_patches.append((class_name, basename))
            patient_classes[class_name].add(patient_part)
        
        class_counts[class_name] += 1

    # Summary
    print(f"\nTotal patches: {len(tif_files)}")
    print(f"  Test patches (real patients): {len(test_patches)}")
    print(f"  Train-only patches (patient -1_*): {len(train_only_patches)}")
    
    print(f"\nClass distribution:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        n_patients = len(patient_classes.get(cls, set()))
        print(f"  {cls:25s}: {count:6d} patches ({n_patients} patients)")
    
    # Load a sample patch to verify shape and dtype
    sample = tifffile.imread(tif_files[0])
    print(f"\nSample patch shape: {sample.shape}")
    print(f"Sample patch dtype: {sample.dtype}")
    print(f"Sample patch min/max: {sample.min():.2f} / {sample.max():.2f}")
    print(f"  (Values should be in Hounsfield Units)")
    
    # Save class distribution
    stats = {
        "total_patches": len(tif_files),
        "test_patches": len(test_patches),
        "train_only_patches": len(train_only_patches),
        "class_distribution": dict(class_counts),
        "sample_shape": list(sample.shape),
        "sample_dtype": str(sample.dtype),
        "hu_range": [float(sample.min()), float(sample.max())]
    }
    
    with open(os.path.join(OUTPUT_DIR, "talisman_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    
    return stats

# ============================================================
# 2. TEXT ROI ANNOTATION ANALYSIS
# ============================================================
def parse_roi_txt(filepath):
    """Parse a text ROI annotation file (Python port of loadROIfiles.m)."""
    rois = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        spacing_x = spacing_y = label = slice_number = None
        for line in f:
            line = line.strip()
            if line.startswith("SpacingX:"):
                spacing_x = float(line.split(": ")[1])
            elif line.startswith("SpacingY:"):
                spacing_y = float(line.split(": ")[1])
            elif line.startswith("label:"):
                label = line.split(": ")[1].strip()
            elif line.startswith("slice_number:"):
                slice_number = int(line.split(": ")[1])
            elif line.startswith("nb_points_on_contour:"):
                nb_points = int(line.split(": ")[1])
                x_vals, y_vals = [], []
                for _ in range(nb_points):
                    coords = f.readline().strip().split()
                    x_vals.append(float(coords[0]))
                    y_vals.append(float(coords[1]))
                rois.append({
                    "spacing_x": spacing_x,
                    "spacing_y": spacing_y,
                    "label": label,
                    "slice_number": slice_number,
                    "nb_points": nb_points,
                    "x_values": x_vals,
                    "y_values": y_vals,
                })
    return rois


def analyze_txt_rois():
    """Analyze all text ROI annotations across all patients."""
    print("\n" + "=" * 60)
    print("2. TEXT ROI ANNOTATION ANALYSIS")
    print("=" * 60)

    patient_dirs = sorted([
        d for d in os.listdir(TXT_ROIS_DIR)
        if os.path.isdir(os.path.join(TXT_ROIS_DIR, d))
    ])
    
    total_rois = 0
    label_counts = Counter()
    patient_labels = defaultdict(set)
    rois_per_patient = {}

    for patient_id in patient_dirs:
        patient_path = os.path.join(TXT_ROIS_DIR, patient_id)
        txt_files = glob.glob(os.path.join(patient_path, "*.txt"))
        
        patient_roi_count = 0
        for txt_file in txt_files:
            rois = parse_roi_txt(txt_file)
            for roi in rois:
                label_counts[roi["label"]] += 1
                patient_labels[patient_id].add(roi["label"])
                patient_roi_count += 1
        
        rois_per_patient[patient_id] = patient_roi_count
        total_rois += patient_roi_count

    print(f"Total patients with txt ROIs: {len(patient_dirs)}")
    print(f"Total ROI annotations: {total_rois}")
    
    print(f"\nLabel distribution across all patients:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        n_patients = sum(1 for p in patient_labels if label in patient_labels[p])
        print(f"  {label:35s}: {count:5d} ROIs ({n_patients} patients)")
    
    avg_rois = np.mean(list(rois_per_patient.values()))
    print(f"\nAvg ROIs per patient: {avg_rois:.1f}")
    print(f"Min ROIs: {min(rois_per_patient.values())}, Max: {max(rois_per_patient.values())}")
    
    stats = {
        "total_patients": len(patient_dirs),
        "total_rois": total_rois,
        "label_distribution": dict(label_counts),
        "avg_rois_per_patient": float(avg_rois)
    }
    
    with open(os.path.join(OUTPUT_DIR, "txt_roi_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    
    return stats

# ============================================================
# 3. VOLUME ROI ANALYSIS (DICOM + masks)
# ============================================================
def analyze_volume_rois():
    """Analyze volume ROIs: count DICOM slices and unique mask labels per patient."""
    print("\n" + "=" * 60)
    print("3. VOLUME ROI ANALYSIS")
    print("=" * 60)
    
    try:
        import pydicom
    except ImportError:
        print("pydicom not installed. Skipping DICOM analysis.")
        return {}
    
    patient_dirs = sorted([
        d for d in os.listdir(VOLUME_ROIS_DIR)
        if os.path.isdir(os.path.join(VOLUME_ROIS_DIR, d))
    ])
    
    print(f"Total patient folders: {len(patient_dirs)}")
    
    # Sample one patient to understand the data
    sample_patient = patient_dirs[0]
    sample_path = os.path.join(VOLUME_ROIS_DIR, sample_patient)
    
    dcm_files = sorted(glob.glob(os.path.join(sample_path, "*.dcm")))
    mask_dir = os.path.join(sample_path, "roi_mask")
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.dcm"))) if os.path.exists(mask_dir) else []
    
    print(f"\nSample patient ({sample_patient}):")
    print(f"  DICOM slices: {len(dcm_files)}")
    print(f"  Mask files: {len(mask_files)}")
    
    # Load sample DICOM
    ds = pydicom.dcmread(dcm_files[0])
    pixel_array = ds.pixel_array
    print(f"  Image shape: {pixel_array.shape}")
    print(f"  Pixel dtype: {pixel_array.dtype}")
    print(f"  Rows x Cols: {ds.Rows} x {ds.Columns}")
    
    # Convert to HU
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    slope = float(getattr(ds, "RescaleSlope", 1))
    hu_image = pixel_array * slope + intercept
    print(f"  HU range: [{hu_image.min():.0f}, {hu_image.max():.0f}]")
    print(f"  Rescale Intercept: {intercept}, Slope: {slope}")
    
    # Load sample mask
    if mask_files:
        mask_ds = pydicom.dcmread(mask_files[0])
        mask_array = mask_ds.pixel_array
        unique_labels = np.unique(mask_array)
        print(f"  Mask unique values: {unique_labels}")
        label_names = [LABEL_MAP.get(v, f"unknown_{v}") for v in unique_labels if v != 0]
        print(f"  Corresponding labels: {label_names}")
    
    # Scan all patients for label statistics
    all_labels = Counter()
    slices_per_patient = {}
    
    for pid in patient_dirs:
        p_path = os.path.join(VOLUME_ROIS_DIR, pid)
        p_dcm = glob.glob(os.path.join(p_path, "*.dcm"))
        slices_per_patient[pid] = len(p_dcm)
        
        p_mask_dir = os.path.join(p_path, "roi_mask")
        if os.path.exists(p_mask_dir):
            p_masks = glob.glob(os.path.join(p_mask_dir, "*.dcm"))
            for mf in p_masks:
                try:
                    mds = pydicom.dcmread(mf, force=True)
                    if hasattr(mds, 'pixel_array'):
                        for val in np.unique(mds.pixel_array):
                            if val != 0:
                                all_labels[int(val)] += 1
                except Exception:
                    pass
    
    print(f"\nLabel occurrence across ALL patients (# of slices containing the label):")
    for label_val, count in sorted(all_labels.items()):
        label_name = LABEL_MAP.get(label_val, f"unknown_{label_val}")
        print(f"  {label_val:2d} ({label_name:30s}): {count:5d} slices")
    
    avg_slices = np.mean(list(slices_per_patient.values()))
    print(f"\nAvg slices per patient: {avg_slices:.1f}")
    
    stats = {
        "total_patients": len(patient_dirs),
        "sample_image_shape": list(pixel_array.shape),
        "hu_range_sample": [float(hu_image.min()), float(hu_image.max())],
        "avg_slices_per_patient": float(avg_slices),
        "label_occurrences": {LABEL_MAP.get(k, f"unknown_{k}"): v for k, v in all_labels.items()}
    }
    
    with open(os.path.join(OUTPUT_DIR, "volume_roi_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    
    return stats

# ============================================================
# 4. CLINICAL PARAMETERS ANALYSIS
# ============================================================
def analyze_clinical():
    """Parse and analyze the clinical parameters Excel file."""
    print("\n" + "=" * 60)
    print("4. CLINICAL PARAMETERS ANALYSIS")
    print("=" * 60)
    
    # The file is an encrypted OLE2 document. Try decryption first.
    df = None
    
    # Method 1: Try msoffcrypto decryption with common passwords
    try:
        import msoffcrypto, io
        passwords_to_try = ['z3Sgl', '', 'VelvetSweatshop', 'password', '1234']
        with open(CLINICAL_FILE, 'rb') as f:
            ms_file = msoffcrypto.OfficeFile(f)
            for pwd in passwords_to_try:
                try:
                    ms_file.load_key(password=pwd)
                    decrypted = io.BytesIO()
                    ms_file.decrypt(decrypted)
                    decrypted.seek(0)
                    df = pd.read_excel(decrypted)
                    print(f"  Decrypted with password: '{pwd}'")
                    break
                except Exception:
                    continue
    except ImportError:
        pass
    
    # Method 2: Try xlrd directly (for unencrypted .xls)
    if df is None:
        try:
            df = pd.read_excel(CLINICAL_FILE, engine="xlrd")
        except Exception:
            pass
    
    # Method 3: Try openpyxl (for standard .xlsx)
    if df is None:
        try:
            df = pd.read_excel(CLINICAL_FILE, engine="openpyxl")
        except Exception:
            pass
    
    if df is None:
        print("  WARNING: Could not read clinicalParameters.xlsx")
        print("  The file appears to be PASSWORD-ENCRYPTED (OLE2 with EncryptedPackage).")
        print("  Please decrypt manually or provide the password.")
        print("  Skipping clinical parameters analysis.")
        return {"error": "encrypted_file"}
    
    print(f"Shape: {df.shape}")
    print(f"Rows (patients): {df.shape[0]}")
    print(f"Columns (parameters): {df.shape[1]}")
    
    print(f"\nColumn names ({df.shape[1]} total):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        print(f"  [{i:3d}] {col:40s} | dtype: {str(dtype):10s} | missing: {n_missing}/{df.shape[0]}")
    
    print(f"\nFirst 5 rows (transposed):")
    print(df.head().T.to_string())
    
    # Save summary
    summary = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns}
    }
    
    with open(os.path.join(OUTPUT_DIR, "clinical_stats.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    # Save cleaned CSV for easier reuse
    df.to_csv(os.path.join(OUTPUT_DIR, "clinical_parameters_cleaned.csv"), index=False)
    print(f"\nSaved cleaned CSV to {OUTPUT_DIR}/clinical_parameters_cleaned.csv")
    
    return summary

# ============================================================
# 5. LOPO CROSS-VALIDATION SPLIT PREPARATION
# ============================================================
def prepare_lopo_splits():
    """Create Leave-One-Patient-Out cross-validation splits for the Talisman suite."""
    print("\n" + "=" * 60)
    print("5. LOPO CROSS-VALIDATION SPLITS")
    print("=" * 60)
    
    tif_files = glob.glob(os.path.join(TALISMAN_DIR, "*.tif"))
    
    # Parse all test patches (exclude train-only with patient -1_*)
    patches = []
    for f in tif_files:
        basename = os.path.basename(f)
        parts = basename.replace(".tif", "").split("_patch")
        class_name = parts[0]
        rest = parts[1]
        patient_part = rest.split("_patient")[1]
        
        if not patient_part.startswith("-1"):
            patches.append({
                "file": basename,
                "class": class_name,
                "patient_id": int(patient_part),
                "path": f
            })
    
    df = pd.DataFrame(patches)
    unique_patients = sorted(df["patient_id"].unique())
    
    print(f"Total test patches: {len(df)}")
    print(f"Unique patients (for LOPO): {len(unique_patients)}")
    print(f"Patient IDs: {unique_patients}")
    
    # Create LOPO splits info
    splits = []
    for pid in unique_patients:
        test_mask = df["patient_id"] == pid
        n_test = test_mask.sum()
        n_train = (~test_mask).sum()
        test_classes = dict(df[test_mask]["class"].value_counts())
        splits.append({
            "patient_id": int(pid),
            "n_test": int(n_test),
            "n_train": int(n_train),
            "test_class_distribution": test_classes
        })
    
    print(f"\nSample LOPO split (patient {unique_patients[0]}):")
    print(f"  Test: {splits[0]['n_test']} patches")
    print(f"  Train: {splits[0]['n_train']} patches")
    print(f"  Test classes: {splits[0]['test_class_distribution']}")
    
    with open(os.path.join(OUTPUT_DIR, "lopo_splits.json"), "w") as f:
        json.dump(splits, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nSaved LOPO splits to {OUTPUT_DIR}/lopo_splits.json")
    
    return splits

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("ILD Dataset Exploration Script")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print()
    
    # 1. Analyze Talisman patches
    talisman_stats = analyze_talisman()
    
    # 2. Analyze text ROI annotations
    txt_roi_stats = analyze_txt_rois()
    
    # 3. Analyze volume ROIs (DICOM + masks)
    volume_stats = analyze_volume_rois()
    
    # 4. Analyze clinical parameters
    clinical_stats = analyze_clinical()
    
    # 5. Prepare LOPO splits
    lopo_splits = prepare_lopo_splits()
    
    print("\n" + "=" * 60)
    print("EXPLORATION COMPLETE")
    print("=" * 60)
    print(f"All statistics saved to: {OUTPUT_DIR}")
    print(f"Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")
