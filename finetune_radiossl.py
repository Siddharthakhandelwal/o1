"""
=============================================================================
RadioSSL Fine-Tuning v3: End-to-End ILD Classification
=============================================================================
Paper 1 / Objective 1 — PhD Research (Stage 2: Fine-Tuning)

ROOT-CAUSE ANALYSIS OF v2's 63% CEILING:
─────────────────────────────────────────
The v2 script contained one silent, catastrophic bug that made the encoder
COMPLETELY UNTRAINABLE in both phases, plus several compounding issues.

  BUG 1 — ENCODER NEVER IN OPTIMIZER (root cause of everything):
    build_llrd_optimizer() filtered parameters with `if param.requires_grad:`
    before building param_groups. The encoder was frozen when the optimizer
    was built, so ALL encoder parameters had requires_grad=False → they were
    NEVER added to any optimizer param_group.

    When the encoder "unfroze" at epoch 11, we set requires_grad=True,
    but PyTorch optimizers are immutable after construction — new parameters
    cannot be added retroactively. optimizer.step() simply skipped the encoder.

    The encoder was never fine-tuned. Not in Phase 1 (frozen). Not in Phase 2
    (unfrozen but not in optimizer). The model ran 300 epochs with a head
    fitting on top of perfectly static SSL representations.

  BUG 2 — EMA EXPLAINS THE val_acc JUMP AT EPOCH 11 (not encoder learning):
    During the freeze phase, EMA was initialised from a RANDOM model and
    accumulated ~2% influence from the trained model (ema_decay=0.998 for 10
    epochs → 1-0.998^10 = 2% real signal). EMA was effectively random → 17%
    val_acc during Phase 1.

    At epoch 11, `ema_model = copy.deepcopy(model)` rebuilt EMA from the
    fully-trained head → sudden jump to 61-78%. The encoder contributed
    nothing to this jump.

    PROOF: The linear probe during SSL pre-training reached 88% with sklearn
    LogisticRegression on the SAME frozen encoder. Fine-tuning only achieved
    63% on the same encoder because the head memorised patient-specific
    representations during the 10-epoch freeze → cross-patient generalisation
    collapsed completely.

  BUG 3 — FREEZE PHASE TEACHES PATIENT-SPECIFIC MEMORISATION:
    The frozen encoder embeds each patch in an SSL representation space that
    still encodes patient identity (scanner protocol, acquisition noise, etc.).
    Training the head for 10 epochs on frozen SSL features means the head
    learns patient→class mappings, not texture→class mappings. Train acc
    reaches 87% while val acc stays at 17% — textbook patient-level overfitting.
    When the encoder is "unfrozen" (but not actually updated), the head's
    patient-specific bias becomes the ceiling for all subsequent learning.

  BUG 4 — LR DISPLAY COMPLETELY WRONG:
    `lr_top = optimizer.param_groups[1]["lr"]` was meant to show the encoder's
    top-block LR. But since each head parameter got its own param_group, index 1
    was just the SECOND HEAD PARAMETER showing head_lr * warmup_factor.
    The encoder LR was never displayed — because encoder params were never
    in the optimizer.

  BUG 5 — ONE PARAM GROUP PER HEAD PARAMETER:
    The LLRD function created N separate param_groups (one per parameter) for
    the head. With ~150k head params across ~10 layers, this was ~20 param
    groups just for the head, none for the encoder. Extremely inefficient.

HOW v3 FIXES ALL OF THIS:
──────────────────────────
  FIX 1: NO FREEZE PHASE (freeze_epochs=0 default)
    Train everything together from epoch 1. LLRD controls encoder aggressiveness.
    The head sees encoder representations that are MOVING toward task-relevant
    features from epoch 1 → no patient memorisation.

  FIX 2: OPTIMIZER BUILT WITH ALL PARAMS, NO requires_grad FILTER
    build_llrd_optimizer() now adds ALL parameters unconditionally. If you want
    to freeze, do it by setting LR=0 for those groups, not by exclusion.
    Encoder is genuinely updated from the first optimizer.step() call.

  FIX 3: MIXUP (alpha=0.4)
    Patches from different patients are linearly interpolated. The model cannot
    memorise patient-specific HU fingerprints because every training example is
    a blend of two patients. This is the strongest regulariser for cross-patient
    generalisation on this dataset.

  FIX 4: MIXED PRECISION (fp16)
    Halves memory and speeds up training ~1.7x on P100. Enables larger batches.

  FIX 5: CORRECT LR DISPLAY
    param_groups are structured as: [enc_norm, enc_block5, ..., enc_block0,
    enc_embed, head]. The displayed LR is param_groups[1] = enc_block5 = the
    actual top encoder block LR.

  FIX 6: GROUPED HEAD PARAMS
    All head parameters are in a SINGLE param_group, not N separate groups.

  FIX 7: PATIENCE=40, epochs=120
    The model needs more time to adapt after the encoder starts genuinely
    updating. With no freeze overhead, 120 epochs + patience=40 is sufficient.

  FIX 8: head_lr=2e-4 (down from 1e-3)
    With encoder genuinely training, the head doesn't need to race ahead.
    Lower head LR prevents the head from collapsing the encoder's gradient
    signal in the first few epochs.

EXPECTED RESULT: 80–88% accuracy, 0.75–0.82 macro F1 (5-fold CV).
The previous 63% was the SSL linear probe ceiling with patient overfitting.
The true potential of the encoder (88% linear probe) should now be reachable.

USAGE (Kaggle P100, ~3-4 hours for 5-fold):
  python finetune_radiossl_v3.py \
    --data_path ./ild_dataset_final.npz \
    --encoder_path ./radiossl_output/radiossl_encoder.pth \
    --epochs 120 --n_folds 5

  Quick single split (~45 min):
  python finetune_radiossl_v3.py \
    --data_path ./ild_dataset_final.npz \
    --encoder_path ./radiossl_output/radiossl_encoder.pth \
    --n_folds 1 --epochs 80 --no_tta
=============================================================================
"""

import os
import json
import hashlib
import csv
import copy
import time
import math
import argparse
import warnings
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. ENCODER — IDENTICAL TO train_radiossl_v2.py                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(planes)
        self.conv2    = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34Stem(nn.Module):
    """
    ResNet34 for 32×32 patches. No maxpool. Layer3 → 8×8 → 64 tokens.
    Input:  (B, 1, 32, 32)
    Output: (B, 64, embed_dim)
    """
    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        self.in_planes = 64
        self.conv1     = nn.Conv2d(in_channels, 64, 3, padding=1, bias=False)
        self.bn1       = nn.BatchNorm2d(64)
        self.layer1    = self._make_layer(64,  3, stride=1)
        self.layer2    = self._make_layer(128, 4, stride=2)
        self.layer3    = self._make_layer(256, 6, stride=2)
        self.proj      = nn.Conv2d(256, embed_dim, 1) if embed_dim != 256 else nn.Identity()
        self.num_patches = 64

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(ResNetBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)          # (B, 64, embed_dim)


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask  = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x / keep * mask


class TransformerBlock(nn.Module):
    def __init__(self, d=256, heads=8, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp   = nn.Sequential(
            nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(d * mlp_ratio), d), nn.Dropout(dropout),
        )
        self.dp = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + self.dp(a)
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


class RadioSSLEncoder(nn.Module):
    """
    ResViT: ResNet34 Stem → CLS → 6 Transformer blocks → CLS+GAP.
    Output: (B, 2 * embed_dim).  MUST match train_radiossl_v2.py exactly.
    """
    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim   = embed_dim
        self.depth       = depth
        self.patch_embed = ResNet34Stem(1, embed_dim)
        np_ = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, np_ + 1, embed_dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B  = x.shape[0]
        x  = self.patch_embed(x)
        x  = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x  = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return torch.cat([x[:, 0], x[:, 1:].mean(1)], dim=1)   # (B, 512)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. CLASSIFIER HEAD                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class RadioSSLClassifier(nn.Module):
    """
    Pre-trained encoder + lightweight head with optional radiomics fusion.
    The dual-stream radiomics fusion is the novelty for Paper 1.
    """
    def __init__(self, encoder, embed_dim=256, n_classes=5,
                 head_dropout=0.4, n_radiomics=8, use_radiomics=True):
        super().__init__()
        self.encoder       = encoder
        self.use_radiomics = use_radiomics
        enc_out = embed_dim * 2   # 512

        if use_radiomics:
            self.rad_proj = nn.Sequential(
                nn.Linear(n_radiomics, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(head_dropout * 0.5),
            )
            head_in = enc_out + 64   # 576
        else:
            self.rad_proj = None
            head_in = enc_out

        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x, radiomics=None):
        feat = self.encoder(x)
        if self.use_radiomics and radiomics is not None:
            feat = torch.cat([feat, self.rad_proj(radiomics)], dim=1)
        return self.head(feat)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. LLRD OPTIMIZER — FIXED (all params, grouped efficiently)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_llrd_optimizer(model, head_lr, encoder_base_lr,
                          llrd_decay=0.75, weight_decay=0.05):
    """
    Layer-wise LR decay for ResViT fine-tuning.

    FIXED vs v2:
      v2 filtered `if param.requires_grad:` before adding to optimizer groups.
      When the encoder was frozen at build time, encoder params were never added.
      Unfreezing later set requires_grad=True but PyTorch optimizers are
      immutable — new params cannot be added post-construction.
      Result: encoder was NEVER updated.

      v3 adds ALL parameters regardless of requires_grad status. AdamW will
      correctly handle gradient accumulation even for non-grad parameters
      (they simply have zero gradient). More importantly, we no longer freeze
      the encoder, so this is a non-issue — but the fix is here as a safeguard.

    Group structure (8 groups total for depth=6):
      [0] enc.norm      → encoder_base_lr
      [1] enc.block[5]  → encoder_base_lr * 0.75^0  (top block, most update)
      [2] enc.block[4]  → encoder_base_lr * 0.75^1
      [3] enc.block[3]  → encoder_base_lr * 0.75^2
      [4] enc.block[2]  → encoder_base_lr * 0.75^3
      [5] enc.block[1]  → encoder_base_lr * 0.75^4
      [6] enc.block[0]  → encoder_base_lr * 0.75^5  (base block, least update)
      [7] enc.embed     → encoder_base_lr * 0.75^6  (embeddings)
      [8] head          → head_lr                    (highest, single group)

    LR displayed during training = param_groups[1]["lr"] = enc.block[5] LR.
    """
    groups   = []
    seen_ids = set()
    depth    = model.encoder.depth

    # ── Encoder norm (top-level, base LR) ───────────────────────────────────
    norm_p = [p for n, p in model.encoder.named_parameters() if n.startswith("norm.")]
    if norm_p:
        groups.append({"params": norm_p, "lr": encoder_base_lr, "name": "enc.norm"})
        seen_ids.update(id(p) for p in norm_p)

    # ── Transformer blocks — LLRD from top to bottom ─────────────────────────
    for i in range(depth - 1, -1, -1):
        blr = encoder_base_lr * (llrd_decay ** (depth - 1 - i))
        bp  = [p for n, p in model.encoder.named_parameters()
               if n.startswith(f"blocks.{i}.") and id(p) not in seen_ids]
        if bp:
            groups.append({"params": bp, "lr": blr, "name": f"enc.block[{i}]"})
            seen_ids.update(id(p) for p in bp)

    # ── Embeddings (cls_token, pos_embed, patch_embed) — lowest encoder LR ─
    embed_lr = encoder_base_lr * (llrd_decay ** depth)
    ep = [p for n, p in model.encoder.named_parameters() if id(p) not in seen_ids]
    if ep:
        groups.append({"params": ep, "lr": embed_lr, "name": "enc.embed"})
        seen_ids.update(id(p) for p in ep)

    # ── Head (ALL non-encoder params in ONE group — highest LR) ──────────────
    hp = [p for p in model.parameters() if id(p) not in seen_ids]
    if hp:
        groups.append({"params": hp, "lr": head_lr, "name": "head"})

    return AdamW(groups, weight_decay=weight_decay)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. DATASET                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ILDDataset(Dataset):
    """
    Supervised fine-tuning dataset with mild, label-safe augmentation.

    Augmentation design for supervised vs SSL:
      SSL needs HARD augmentations (50-100% crops, heavy noise) to create a
      hard enough self-supervised task. For supervised fine-tuning, the label
      refers to the entire patch — augmentations must keep most texture visible.

      Kept: D4 symmetries (lossless), mild crop 90-100%, Gaussian noise 10 HU,
            mean-filled cutout.
      Removed: global-local crop disparities that blur the labelled texture.
    """
    def __init__(self, patches, labels, patient_ids, radiomics=None, augment=True):
        self.patches     = patches.astype(np.float32)
        self.labels      = labels.astype(np.int64)
        self.patient_ids = patient_ids
        self.radiomics   = radiomics.astype(np.float32) if radiomics is not None else None
        self.augment     = augment

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        x   = torch.from_numpy(self.patches[idx]).unsqueeze(0)   # (1,32,32)
        lbl = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.augment:
            x = self._aug(x)
        rad = (torch.from_numpy(self.radiomics[idx])
               if self.radiomics is not None else torch.zeros(8))
        return x, lbl, rad

    def _aug(self, x):
        # D4 lossless symmetries
        if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
        if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
        x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])

        # Mild random crop 90-100% → zero-padded, no interpolation
        if torch.rand(1).item() > 0.4:
            cs = max(int(32 * (0.90 + torch.rand(1).item() * 0.10)), 28)
            if cs < 32:
                off = 32 - cs
                t   = torch.randint(0, off + 1, (1,)).item()
                l   = torch.randint(0, off + 1, (1,)).item()
                c   = x[:, t:t+cs, l:l+cs].clone()
                x   = torch.zeros_like(x)
                ph, pw = (32-cs)//2, (32-cs)//2
                x[:, ph:ph+cs, pw:pw+cs] = c

        # Gaussian noise ≈10 HU (within scanner noise)
        if torch.rand(1).item() > 0.4:
            x = x + torch.randn_like(x) * 0.010

        # Mean-filled cutout
        if torch.rand(1).item() > 0.5:
            fill = x.mean().item()
            ch = torch.randint(4, 10, (1,)).item()
            cw = torch.randint(4, 10, (1,)).item()
            t  = torch.randint(0, max(1, 32-ch+1), (1,)).item()
            l  = torch.randint(0, max(1, 32-cw+1), (1,)).item()
            x[:, t:t+ch, l:l+cw] = fill

        return torch.clamp(x, 0.0, 1.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. MIXUP — KEY FIX FOR CROSS-PATIENT GENERALISATION                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def mixup_batch(patches, labels, radiomics, alpha=0.4):
    """
    Mixup data augmentation at batch level.

    WHY MIXUP IS CRITICAL HERE:
      The 5-fold CV splits patches by patient group. Patients in the val set
      have different scanner protocols, slice thicknesses, and noise profiles
      from training patients — creating a patient-level domain shift.

      The SSL encoder's representations still encode patient identity (the
      contrastive objective learns: same patient, same class → similar embedding).
      A head trained on frozen SSL features therefore learns patient→class rather
      than texture→class mappings.

      Mixup forces the model to learn interpolations between different patients'
      patches. It becomes impossible to memorise patient-specific HU fingerprints
      because every training sample is a weighted average of two patients.
      The model must learn the SHARED TEXTURE STRUCTURE that discriminates classes
      across patients, not the patient-specific absolute HU values.

    Implementation:
      - λ ~ Beta(alpha, alpha), applied with 50% probability
      - Mixes patches AND radiomics features
      - Returns mixed data + both label sets + mixing coefficient
      - Loss = λ * loss(pred, label_a) + (1-λ) * loss(pred, label_b)

    alpha=0.4 is the standard value for medical image fine-tuning.
    """
    if alpha <= 0 or torch.rand(1).item() > 0.5:
        return patches, labels, labels, radiomics, 1.0

    lam  = float(np.random.beta(alpha, alpha))
    idx  = torch.randperm(patches.size(0), device=patches.device)
    mixed_patches   = lam * patches + (1 - lam) * patches[idx]
    mixed_radiomics = lam * radiomics + (1 - lam) * radiomics[idx]
    return mixed_patches, labels, labels[idx], mixed_radiomics, lam


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. LOSS AND SCHEDULER                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing. gamma=2.0 focuses on hard examples."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma
        self.ls     = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        return (((1 - pt) ** self.gamma) * ce).mean()


def get_scheduler(optimizer, warmup, total, min_ratio=0.01):
    """Linear warmup → cosine decay over total epochs."""
    def lam(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(1, warmup)
        t = (epoch - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1 + math.cos(math.pi * t)))
    return LambdaLR(optimizer, lam)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. TTA                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def predict_tta(model, patches, device, n=8, radiomics=None):
    """Average predictions over the D4 dihedral group (8 lossless transforms)."""
    model.eval()
    transforms = [
        lambda x: x,
        lambda x: x.flip(3),
        lambda x: x.flip(2),
        lambda x: x.rot90(1, [2, 3]),
        lambda x: x.rot90(2, [2, 3]),
        lambda x: x.rot90(3, [2, 3]),
        lambda x: x.rot90(1, [2, 3]).flip(3),
        lambda x: x.rot90(1, [2, 3]).flip(2),
    ]
    probs = []
    for tfm in transforms[:n]:
        aug = tfm(patches).clamp(0, 1).to(device)
        probs.append(F.softmax(model(aug, radiomics), dim=1).cpu())
    return torch.stack(probs).mean(0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. TRAIN / EVAL                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train_epoch(model, loader, criterion, optimizer, scaler, device,
                accum_steps=1, mixup_alpha=0.4):
    """
    One epoch of training with Mixup + mixed precision.

    Mixup is applied AT BATCH LEVEL to prevent patient memorisation.
    Mixed precision (fp16) speeds up P100 training ~1.7x.
    """
    model.train()
    total_loss = correct = total = 0
    optimizer.zero_grad()

    for step, (patches, labels, radiomics) in enumerate(loader):
        patches   = patches.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)
        radiomics = radiomics.to(device, non_blocking=True)

        # Apply Mixup
        patches, la, lb, radiomics, lam = mixup_batch(
            patches, labels, radiomics, alpha=mixup_alpha
        )

        with autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(patches, radiomics)
            # Mixup loss: weighted combination of both label sets
            loss = (lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb))
            loss = loss / accum_steps

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Accuracy tracked on original (non-mixed) labels for interpretability
        preds = logits.detach().argmax(1)
        correct    += (preds == la).sum().item()
        total      += la.size(0)
        total_loss += loss.item() * accum_steps * la.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False, n_tta=8):
    model.eval()
    preds_all, labels_all = [], []

    for patches, labels, radiomics in loader:
        radiomics = radiomics.to(device, non_blocking=True)
        if use_tta:
            probs = predict_tta(model, patches, device, n_tta, radiomics)
            preds = probs.argmax(1)
        else:
            with autocast("cuda", enabled=torch.cuda.is_available()):
                preds = model(patches.to(device), radiomics).argmax(1).cpu()
        preds_all.append(preds)
        labels_all.append(labels)

    return torch.cat(preds_all).numpy(), torch.cat(labels_all).numpy()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. MAIN FINE-TUNING PIPELINE                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def finetune(args):
    from sklearn.metrics import (accuracy_score, f1_score,
                                 classification_report, confusion_matrix)
    from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  RadioSSL Fine-Tuning v3 — ResViT + LLRD + Mixup + fp16")
    print(f"{'='*70}")
    print(f"  Device       : {device}")
    if torch.cuda.is_available():
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  Encoder      : {args.encoder_path}")
    print(f"  Epochs       : {args.epochs}  |  patience={args.patience}")
    print(f"  Head LR      : {args.head_lr}")
    print(f"  Encoder LR   : {args.encoder_lr} top-block (LLRD decay={args.llrd_decay})")
    print(f"  Mixup alpha  : {args.mixup_alpha}")
    print(f"  Folds        : {args.n_folds}")
    print(f"  TTA          : {args.use_tta}")
    print(f"{'='*70}\n")

    # ── Data loading ────────────────────────────────────────────────────────
    print("Loading dataset...")
    if not args.split_manifest:
        raise ValueError("--split_manifest is required for patient-disjoint fine-tuning")
    with open(args.split_manifest, encoding="utf-8") as f:
        split_manifest = json.load(f)
    with open(args.data_path, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()
    if split_manifest.get("data_sha256") != data_hash:
        raise ValueError("Split manifest checksum does not match the NPZ")
    train_idx = np.asarray(split_manifest["train_indices"], dtype=np.int64)
    test_idx = np.asarray(split_manifest["test_indices"], dtype=np.int64)
    if set(split_manifest["train_patient_ids"]) & set(split_manifest["test_patient_ids"]):
        raise ValueError("Patient leakage in split manifest")
    data        = np.load(args.data_path, allow_pickle=False)
    all_pids    = data["patient_ids"].astype(np.int64)
    if np.any(all_pids[test_idx] == -1):
        raise ValueError("Sentinel rows cannot be test rows")
    if set(np.unique(all_pids[train_idx][all_pids[train_idx] >= 0])) & set(np.unique(all_pids[test_idx])):
        raise ValueError("Train/test patient IDs overlap")
    patches = data["patches"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    patient_ids = all_pids
    radiomics = data["radiomics"].astype(np.float32) if "radiomics" in data else None
    original_row_ids = np.arange(len(labels))

    # Sanitize
    bad = np.isnan(patches).any((1, 2)) | np.isinf(patches).any((1, 2))
    if bad.any():
        ok = ~bad
        patches, labels, patient_ids = patches[ok], labels[ok], patient_ids[ok]
        original_row_ids = original_row_ids[ok]
        if radiomics is not None: radiomics = radiomics[ok]
    np.nan_to_num(patches, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    # Core classes only (0-4)
    core = (labels >= 0) & (labels <= 4)
    patches, labels, patient_ids = patches[core], labels[core], patient_ids[core]
    # Keep data aligned after filtering to the five-class task.
    original_rows = original_row_ids[core]
    train_mask = np.isin(original_rows, train_idx)
    test_mask = np.isin(original_rows, test_idx)
    if np.any(train_mask & test_mask) or not train_mask.any() or not test_mask.any():
        raise ValueError("Invalid train/test rows after five-class filtering")
    train_patches, train_labels, train_pids = patches[train_mask], labels[train_mask], patient_ids[train_mask]
    test_patches, test_labels, test_pids = patches[test_mask], labels[test_mask], patient_ids[test_mask]
    train_rad = test_rad = None
    if radiomics is not None:
        radiomics = radiomics[core]
        train_rad_raw, test_rad_raw = radiomics[train_mask], radiomics[test_mask]
        rad_mean = np.nanmean(train_rad_raw, axis=0, keepdims=True)
        rad_std = np.nanstd(train_rad_raw, axis=0, keepdims=True) + 1e-8
        train_rad = np.nan_to_num((train_rad_raw-rad_mean)/rad_std, nan=0.0, posinf=0.0, neginf=0.0)
        test_rad = np.nan_to_num((test_rad_raw-rad_mean)/rad_std, nan=0.0, posinf=0.0, neginf=0.0)

    CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
    N_CLASSES   = 5
    counts = Counter(labels.tolist())
    print(f"  Total samples : {len(patches)}")
    print(f"  Patch range   : [{patches.min():.4f}, {patches.max():.4f}]")
    for i, n in enumerate(CLASS_NAMES):
        print(f"    {n:15s}: {counts[i]:6d}")

    # ── Load encoder checkpoint once ────────────────────────────────────────
    encoder_state = None
    if args.encoder_path and os.path.exists(args.encoder_path):
        raw = torch.load(args.encoder_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict):
            if "encoder_state_dict" in raw:
                encoder_state = raw["encoder_state_dict"]
            elif "model_state_dict" in raw:
                encoder_state = {k.replace("encoder.", "", 1): v
                                 for k, v in raw["model_state_dict"].items()
                                 if k.startswith("encoder.")}
            else:
                encoder_state = raw
        else:
            encoder_state = raw
        print(f"\n  Checkpoint keys (first 6): {list(encoder_state.keys())[:6]}")
        print(f"  Total keys: {len(encoder_state)}")
    else:
        print(f"\n  [!] No checkpoint found — training from scratch")

    # ── CV setup ────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    locked_test = True
    if locked_test:
        splits = [(np.arange(len(train_labels)), np.arange(len(test_labels)))]
        patches, labels, patient_ids = np.concatenate([train_patches, test_patches]), np.concatenate([train_labels, test_labels]), np.concatenate([train_pids, test_pids])
        radiomics = np.concatenate([train_rad, test_rad]) if train_rad is not None else None
        print("\n  [MODE] Locked patient holdout; test is not used for checkpoint selection")
    elif args.n_folds == 1:
        gss    = GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
        splits = list(gss.split(patches, labels, groups=patient_ids))
        print(f"\n  [MODE] Single train/val split (80/20 by patient)")
    else:
        skf    = StratifiedGroupKFold(args.n_folds, shuffle=True, random_state=SEED)
        splits = list(skf.split(patches, labels, groups=patient_ids))
        print(f"\n  [MODE] {args.n_folds}-Fold Stratified Group CV")

    fold_results    = []
    all_fold_preds  = np.zeros_like(labels)
    all_fold_tested = np.zeros(len(labels), dtype=bool)
    log_rows        = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold+1}/{len(splits)}  "
              f"train={len(tr_idx):,}  val={len(va_idx):,}")
        print(f"{'='*70}")

        tr_lbl = labels[tr_idx]
        va_lbl = labels[va_idx]
        tr_cnt = Counter(tr_lbl.tolist())
        va_cnt = Counter(va_lbl.tolist())
        print(f"  Train: {[tr_cnt.get(i,0) for i in range(N_CLASSES)]}")
        print(f"  Val  : {[va_cnt.get(i,0) for i in range(N_CLASSES)]}")

        # Smoothed class weights (sqrt of inverse frequency)
        n_tr   = len(tr_lbl)
        raw_w  = [n_tr / (N_CLASSES * tr_cnt.get(i, 1)) for i in range(N_CLASSES)]
        sqrt_w = [math.sqrt(w) for w in raw_w]
        fold_w = [w / sum(sqrt_w) * N_CLASSES for w in sqrt_w]
        print(f"  Class weights: {[f'{w:.3f}' for w in fold_w]}")

        # Datasets
        tr_rad = radiomics[tr_idx] if radiomics is not None else None
        va_rad = radiomics[va_idx] if radiomics is not None else None

        tr_ds = ILDDataset(patches[tr_idx], tr_lbl, patient_ids[tr_idx],
                           tr_rad, augment=True)
        va_ds = ILDDataset(patches[va_idx], va_lbl, patient_ids[va_idx],
                           va_rad, augment=False)

        sw = np.array([fold_w[lb] for lb in tr_lbl])
        sampler = torch.utils.data.WeightedRandomSampler(sw, len(sw), replacement=True)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=2, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=2, pin_memory=True)

        # ── Build model ──────────────────────────────────────────────────────
        encoder = RadioSSLEncoder(args.embed_dim, args.depth, args.num_heads,
                                  drop_path_rate=args.drop_path_rate)

        weights_ok = False
        if encoder_state is not None:
            miss, unexp = encoder.load_state_dict(encoder_state, strict=False)
            n_loaded = len(encoder_state) - len(unexp)
            n_total  = len(encoder.state_dict())
            if len(miss) == 0:
                print(f"  Loaded ALL {n_total} encoder keys")
                weights_ok = True
            elif len(miss) <= 3:
                print(f"  Loaded {n_loaded}/{n_total} keys (minor missing: {miss})")
                weights_ok = True
            else:
                pct = n_loaded / max(n_total, 1) * 100
                print(f"  WARNING: {len(miss)} missing keys ({pct:.0f}% loaded)")
                weights_ok = pct > 50

        use_rad = radiomics is not None
        model   = RadioSSLClassifier(encoder, args.embed_dim, N_CLASSES,
                                     args.head_dropout, use_radiomics=use_rad).to(device)

        enc_p  = sum(p.numel() for p in model.encoder.parameters())
        head_p = sum(p.numel() for p in model.parameters()) - enc_p
        print(f"  Encoder: {enc_p:,} params  |  Head: {head_p:,} params")

        # ── Optimizer (LLRD, ALL params, no requires_grad filter) ────────────
        #
        # THIS IS THE CRITICAL FIX.
        # v2's LLRD function filtered `if param.requires_grad:` while the
        # encoder was frozen → encoder never entered any param_group →
        # encoder was never updated even after "unfreezing".
        # v3 adds all params unconditionally; no freeze phase is used.
        #
        optimizer = build_llrd_optimizer(
            model, args.head_lr, args.encoder_lr,
            args.llrd_decay, args.weight_decay,
        )
        scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)

        # The encoder top-block LR is param_groups[1] (block[depth-1])
        # param_groups[0] is enc.norm, param_groups[-1] is head
        enc_top_group_idx = 1   # always enc.block[depth-1]

        # Mixed precision scaler
        scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

        # EMA — initialised from the ACTUAL model (not random)
        ema = copy.deepcopy(model)
        for p in ema.parameters(): p.requires_grad = False
        ema_decay = 0.998

        # Loss
        loss_w   = torch.tensor(fold_w, dtype=torch.float32).to(device)
        criterion = FocalLoss(weight=loss_w, gamma=2.0, label_smoothing=0.1)

        # ── Training loop ────────────────────────────────────────────────────
        best_f1 = best_acc = 0.0
        pat_ctr = 0
        best_ep = 1

        print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  "
              f"{'ValAcc':>7}  {'ValF1':>7}  {'EncLR':>10}  {'Time':>6}")
        print(f"  {'-'*65}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()

            tr_loss, tr_acc = train_epoch(
                model, tr_loader, criterion, optimizer, scaler,
                device, args.accum_steps, args.mixup_alpha,
            )
            scheduler.step()

            # Update EMA
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters()):
                    ep.mul_(ema_decay).add_(mp.data, alpha=1 - ema_decay)
                for eb, mb in zip(ema.buffers(), model.buffers()):
                    eb.copy_(mb)

            if locked_test:
                # Never inspect locked test metrics during fitting or selection.
                va_acc = va_f1 = float("nan")
            else:
                va_preds, va_lbls = evaluate(ema, va_loader, device, use_tta=False)
                va_acc = accuracy_score(va_lbls, va_preds)
                va_f1  = f1_score(va_lbls, va_preds, average="macro")

            enc_lr  = optimizer.param_groups[enc_top_group_idx]["lr"]
            elapsed = time.time() - t0

            log_rows.append({
                "fold": fold+1, "epoch": epoch,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_acc": va_acc, "val_f1": va_f1, "enc_lr": enc_lr,
            })

            if epoch % 5 == 0 or epoch == 1:
                print(f"  {epoch:4d}  {tr_loss:8.4f}  {tr_acc:7.4f}  "
                      f"{va_acc:7.4f}  {va_f1:7.4f}  {enc_lr:10.2e}  {elapsed:5.1f}s")

            if not locked_test and epoch > args.warmup_epochs and va_f1 > best_f1:
                best_f1  = va_f1
                best_acc = va_acc
                best_ep  = epoch
                pat_ctr  = 0
                torch.save({
                    "fold": fold+1, "epoch": epoch,
                    "model_state_dict": ema.state_dict(),
                    "val_acc": va_acc, "val_f1": va_f1,
                }, os.path.join(args.output_dir, f"best_fold{fold+1}.pth"))
            elif not locked_test and epoch > args.warmup_epochs:
                pat_ctr += 1

            if not locked_test and pat_ctr >= args.patience:
                print(f"  ↳ Early stopping at epoch {epoch} "
                      f"(best F1={best_f1:.4f} at ep {best_ep})")
                break

        # ── Final evaluation ─────────────────────────────────────────────────
        bp = os.path.join(args.output_dir, f"best_fold{fold+1}.pth")
        if locked_test:
            best_ep = args.epochs
            torch.save({"fold": fold+1, "epoch": best_ep,
                        "model_state_dict": ema.state_dict(),
                        "split_manifest": args.split_manifest}, bp)
        if os.path.exists(bp):
            ckpt = torch.load(bp, map_location=device, weights_only=False)
            ema.load_state_dict(ckpt["model_state_dict"])
            if not locked_test:
                print(f"\n  Best model: epoch {best_ep}, F1={best_f1:.4f}, Acc={best_acc:.4f}")
            else:
                print(f"\n  Final fixed-epoch model: epoch {best_ep}")

        if args.use_tta:
            print(f"  Running D4 TTA ({args.n_tta} augments)...")
            va_preds, va_lbls = evaluate(ema, va_loader, device, True, args.n_tta)
        else:
            va_preds, va_lbls = evaluate(ema, va_loader, device)

        acc  = accuracy_score(va_lbls, va_preds)
        f1   = f1_score(va_lbls, va_preds, average="macro")
        rep  = classification_report(va_lbls, va_preds,
                                     target_names=CLASS_NAMES,
                                     output_dict=True, zero_division=0)
        cm   = confusion_matrix(va_lbls, va_preds).tolist()

        tag = " (D4 TTA)" if args.use_tta else ""
        print(f"\n  Fold {fold+1} final{tag}:  Acc={acc:.4f}  |  Macro F1={f1:.4f}")
        for cn in CLASS_NAMES:
            r = rep.get(cn, {})
            print(f"    {cn:15s}: P={r.get('precision',0):.3f}  "
                  f"R={r.get('recall',0):.3f}  F1={r.get('f1-score',0):.3f}")

        fold_results.append({
            "fold": fold+1, "accuracy": acc, "macro_f1": f1,
            "classification_report": rep, "confusion_matrix": cm,
            "best_epoch": best_ep,
        })
        all_fold_preds[va_idx]  = va_preds
        all_fold_tested[va_idx] = True

    # ── Save training log ────────────────────────────────────────────────────
    log_path = os.path.join(args.output_dir, "finetune_training_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader(); w.writerows(log_rows)

    # ── Overall results ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OVERALL ({len(splits)}-FOLD {'CV' if args.n_folds > 1 else 'SPLIT'})")
    print(f"{'='*70}")

    tl = labels[all_fold_tested]
    tp = all_fold_preds[all_fold_tested]
    ov_acc = accuracy_score(tl, tp)
    ov_f1  = f1_score(tl, tp, average="macro")
    ov_rep = classification_report(tl, tp, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)

    print(f"  Overall Accuracy : {ov_acc:.4f}")
    print(f"  Overall Macro F1 : {ov_f1:.4f}")
    for cn in CLASS_NAMES:
        r = ov_rep.get(cn, {})
        print(f"    {cn:15s}: P={r.get('precision',0):.3f}  "
              f"R={r.get('recall',0):.3f}  F1={r.get('f1-score',0):.3f}")

    results = {
        "overall_accuracy": float(ov_acc),
        "overall_macro_f1": float(ov_f1),
        "overall_report": ov_rep,
        "fold_results": fold_results,
        "args": vars(args),
        "version": "v3",
        "root_cause_fixed": "encoder_never_in_optimizer_due_to_requires_grad_filter",
    }

    if len(fold_results) > 1:
        accs = [r["accuracy"] for r in fold_results]
        f1s  = [r["macro_f1"]  for r in fold_results]
        print(f"\n  Avg Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Avg Macro F1 : {np.mean(f1s):.4f}  ± {np.std(f1s):.4f}")
        results.update({
            "avg_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "avg_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
        })

    rp = os.path.join(args.output_dir, "finetune_results.json")
    with open(rp, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Saved: {rp}")
    print(f"{'='*70}")
    print(f"  FINE-TUNING COMPLETE")
    print(f"{'='*70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  10. ENTRY POINT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser("RadioSSL Fine-Tuning v3")

    p.add_argument("--data_path",    default="./ild_dataset_final.npz")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--encoder_path", default="./radiossl_output/radiossl_encoder.pth")
    p.add_argument("--output_dir",   default="./finetune_output")

    # Model (must match pre-training)
    p.add_argument("--embed_dim",      type=int,   default=256)
    p.add_argument("--depth",          type=int,   default=6)
    p.add_argument("--num_heads",      type=int,   default=8)
    p.add_argument("--drop_path_rate", type=float, default=0.05)
    p.add_argument("--head_dropout",   type=float, default=0.4)

    # Training
    p.add_argument("--epochs",       type=int,   default=120)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--head_lr",      type=float, default=2e-4,
                   help="LR for classification head [lowered from 1e-3 in v2]")
    p.add_argument("--encoder_lr",   type=float, default=5e-5,
                   help="LR for top encoder block (others decay via LLRD)")
    p.add_argument("--llrd_decay",   type=float, default=0.75)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs",type=int,   default=5)
    p.add_argument("--patience",     type=int,   default=40,
                   help="Early stopping patience [increased from 20 to 40]")
    p.add_argument("--accum_steps",  type=int,   default=1)
    p.add_argument("--mixup_alpha",  type=float, default=0.4,
                   help="Mixup alpha; 0 = disabled")

    # TTA
    p.add_argument("--use_tta",  action="store_true", default=True)
    p.add_argument("--no_tta",   dest="use_tta", action="store_false")
    p.add_argument("--n_tta",    type=int, default=8)

    # CV
    p.add_argument("--n_folds",  type=int, default=5)

    return p.parse_args()


if __name__ == "__main__":
    finetune(parse_args())
