"""Create a deterministic patient-disjoint split manifest from RadioSSL NPZ."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def build_split(data_path: Path, output_path: Path, test_size: float, seed: int, trials: int,
                patient_ids_same_namespace: bool = False):
    if not patient_ids_same_namespace:
        raise ValueError(
            "Source audit cannot prove Talisman and volume IDs share a namespace. "
            "Verify from original records, then pass --patient_ids_same_namespace."
        )
    data = np.load(data_path, allow_pickle=False)
    required = {"labels", "patient_ids"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Dataset missing arrays: {sorted(missing)}")
    labels = np.asarray(data["labels"], dtype=np.int64)
    pids = np.asarray(data["patient_ids"], dtype=np.int64)
    if len(labels) != len(pids):
        raise ValueError("labels and patient_ids have different row counts")

    # -1 means no known patient (Talisman training-only rows); it can never be
    # used as a validation/test group. All real IDs are kept intact by GSS.
    sentinel = pids == -1
    real = ~sentinel
    if not real.any():
        raise ValueError("No rows have a real patient ID")
    best_score = 0.0
    if test_size == 0:
        # Use every eligible volume row for SSL when the separate Talisman
        # archive is the locked patient-disjoint evaluation set.
        train_idx = np.arange(len(labels), dtype=np.int64)
        test_idx = np.empty(0, dtype=np.int64)
    else:
        classes = np.unique(labels[(labels >= 0) & real])
        if len(classes) < 2:
            raise ValueError("Need at least two labelled classes among real patients")

        best = None
        best_score = float("inf")
        for offset in range(trials):
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed + offset)
            train_local, test_local = next(splitter.split(
                np.zeros(real.sum()), labels[real], groups=pids[real]))
            real_rows = np.flatnonzero(real)
            tr_real, te = real_rows[train_local], real_rows[test_local]
            tr = np.concatenate([np.flatnonzero(sentinel), tr_real])
            # Select split by class representation only, never by model results.
            tr_dist = np.array([(labels[tr] == c).sum() for c in classes], dtype=float)
            te_dist = np.array([(labels[te] == c).sum() for c in classes], dtype=float)
            if (tr_dist == 0).any() or (te_dist == 0).any():
                continue
            tr_dist /= tr_dist.sum()
            te_dist /= te_dist.sum()
            score = float(np.abs(tr_dist - te_dist).sum())
            if score < best_score:
                best_score = score
                best = (np.sort(tr), np.sort(te))
        if best is None:
            raise ValueError("Could not find a split containing every class on both sides")
        train_idx, test_idx = best
    train_groups = set(map(int, np.unique(pids[train_idx][pids[train_idx] >= 0])))
    test_groups = set(map(int, np.unique(pids[test_idx])))
    if train_groups & test_groups:
        raise AssertionError("Patient leakage: train/test patient IDs overlap")
    if (pids[test_idx] == -1).any():
        raise AssertionError("Sentinel patient -1 must remain training-only")

    digest = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "data_path_name": Path(data_path).name,
        "data_sha256": digest,
        "seed": seed,
        "test_size_requested": test_size,
        "evaluation_mode": "external_validation_only" if test_size == 0 else "internal_patient_holdout",
        "split_search_trials": trials,
        "class_balance_l1": best_score,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_patient_ids": sorted(train_groups),
        "test_patient_ids": sorted(test_groups),
        "sentinel_train_rows": int(sentinel.sum()),
        "patient_id_mapping_assumption": "same numeric patient_id across Talisman and volume sources; explicitly confirmed by caller",
        "class_counts_train": {str(int(c)): int((labels[train_idx] == c).sum()) for c in np.unique(labels[train_idx])},
        "class_counts_test": {str(int(c)): int((labels[test_idx] == c).sum()) for c in np.unique(labels[test_idx])},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test_size", type=float, default=0.25,
                        help="Use 0 when a separate patient-disjoint validation NPZ is held out")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--patient_ids_same_namespace", action="store_true",
                        help="Confirm verified source mapping uses shared numeric patient IDs")
    args = parser.parse_args()
    if not 0 <= args.test_size < 1:
        parser.error("--test_size must be in [0, 1)")
    result = build_split(Path(args.data_path), Path(args.output), args.test_size, args.seed,
                         args.trials, args.patient_ids_same_namespace)
    print(f"train rows={len(result['train_indices'])} test rows={len(result['test_indices'])}")
    print(f"train patients={len(result['train_patient_ids'])} test patients={len(result['test_patient_ids'])}")
    print(f"manifest={args.output}")


if __name__ == "__main__":
    main()
