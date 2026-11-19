"""Fit a linear probe on train patients and evaluate once on locked test patients."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_radiossl import RadioSSLEncoder


@torch.no_grad()
def embed(model, patches, device, batch_size=128):
    result = []
    model.eval()
    for start in range(0, len(patches), batch_size):
        x = torch.from_numpy(patches[start:start+batch_size]).unsqueeze(1).to(device)
        result.append(model(x).cpu().numpy())
    return np.concatenate(result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--validation_data_path", default="",
                   help="Optional held-out Talisman NPZ; otherwise use the manifest's internal test rows")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    args = p.parse_args()
    with open(args.data_path, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    if split.get("data_sha256") != data_hash:
        raise ValueError("Split manifest checksum does not match the NPZ")
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    test_idx = np.asarray(split["test_indices"], dtype=np.int64)
    if set(split["train_patient_ids"]) & set(split["test_patient_ids"]):
        raise ValueError("Patient leakage in split manifest")
    d = np.load(args.data_path, allow_pickle=False)
    pids = d["patient_ids"]
    if set(np.unique(pids[train_idx][pids[train_idx] >= 0])) & set(np.unique(pids[test_idx])):
        raise ValueError("Train/test patient overlap")
    tr = train_idx[(d["labels"][train_idx] >= 0) & (d["labels"][train_idx] <= 4)]
    y_train = d["labels"][tr].astype(int)
    if args.validation_data_path:
        vd = np.load(args.validation_data_path, allow_pickle=False)
        val_pids = vd["patient_ids"]
        if set(np.unique(pids[tr][pids[tr] >= 0])) & set(np.unique(val_pids[val_pids >= 0])):
            raise ValueError("Talisman validation patients overlap the linear-probe training patients")
        te = np.flatnonzero((vd["labels"] >= 0) & (vd["labels"] <= 4))
        val_patches = vd["patches"][te].astype(np.float32)
        y_test = vd["labels"][te].astype(int)
        test_patient_ids = sorted(set(map(int, val_pids[te])))
    else:
        te = test_idx[(d["labels"][test_idx] >= 0) & (d["labels"][test_idx] <= 4)]
        val_patches = d["patches"][te].astype(np.float32)
        y_test = d["labels"][te].astype(int)
        test_patient_ids = split["test_patient_ids"]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("Both splits need at least two labeled classes for the probe")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadioSSLEncoder(args.embed_dim, args.depth, args.num_heads).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    tr_feat = embed(model, d["patches"][tr].astype(np.float32), device, args.batch_size)
    te_feat = embed(model, val_patches, device, args.batch_size)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)
    clf.fit(tr_feat, y_train)
    pred = clf.predict(te_feat)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "train_patients": split["train_patient_ids"],
        "test_patients": test_patient_ids,
        "test_rows": int(len(te)),
        "data_sha256": data_hash,
        "checkpoint": str(args.checkpoint),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("train_patients", "test_patients")}, indent=2))


if __name__ == "__main__":
    main()
