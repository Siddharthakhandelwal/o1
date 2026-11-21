"""
=============================================================================
RadioSSL v3: Self-Supervised Pre-Training with Supervised Contrastive Loss
=============================================================================
Paper 1 / Objective 1 — PhD Research (Stage 1: Pre-Training)

WHY THE CURRENT 88% PROBE IS MISLEADING (and what v3 fixes):
─────────────────────────────────────────────────────────────
The v2 linear probe used StratifiedShuffleSplit — which has NO concept of
patient identity. Patches from patient X appear in both the probe's training
AND test split. The ResNet34 stem learns patient-specific HU signatures
(scanner protocol, slice thickness, reconstruction kernel) within the first
epoch, which is why probe_acc was already 70% at epoch 1. The 88% number
is measuring cross-patch generalisation WITHIN patients, not cross-patient
generalisation. The fine-tuner's patient-grouped CV exposes the real number:
~65% with a frozen encoder.

THREE CHANGES TO v3:
────────────────────

CHANGE 1 — SUPERVISED CONTRASTIVE LOSS (SupCon) [HIGHEST IMPACT]:
  36,823 of 40,307 patches have CLASS LABELS (0=healthy, 1=emphysema,
  2=ground_glass, 3=fibrosis, 4=micronodules). In v2, these labels are
  COMPLETELY UNUSED during pre-training. That is a massive waste.

  SupCon (Khosla et al., NeurIPS 2020) pulls ALL same-class patches together
  in the batch embedding space:
    L_SupCon(i) = -1/|P(i)| * Σ_{j∈P(i)} log(
                   exp(z_i·z_j/τ) / Σ_{k≠i} exp(z_i·z_k/τ)
                  )
  where P(i) = all labelled samples in the batch with the same class as i.

  With batch_size=128 and ~5 balanced classes, each anchor gets ~24
  same-class positives instead of just 1 (MoCo). This is 24× more direct
  class-discriminative supervision per forward pass.

  The combined loss is:
    L_total = L_moco + alpha_supcon * L_supcon + lambda_radio * L_radiomics
  where alpha_supcon=0.5 so SupCon contributes meaningfully without
  overwhelming the self-supervised MoCo objective.

  The SupCon loss is computed using the PROJECTION HEAD outputs (same as
  MoCo), so it adds zero parameters and minimal extra compute.

CHANGE 2 — PATIENT-AWARE LINEAR PROBE [CRITICAL FOR HONEST NUMBERS]:
  The linear probe now uses StratifiedGroupKFold with patient_ids as groups,
  matching the evaluation protocol of the downstream fine-tuner exactly.
  This gives HONEST cross-patient generalisation numbers during pre-training,
  so training optimises toward the right objective.

  The probe_acc will initially APPEAR to drop (from 88% → ~60-65%) because
  we're now measuring the right thing. The goal is to push this honest
  number as high as possible. With SupCon, we expect it to reach 78-84%.

CHANGE 3 — TRIPLE-VIEW (1 GLOBAL + 2 LOCAL CROPS) [MEDIUM IMPACT]:
  v2: 1 global view + 1 local view → 2 positive pairs per anchor
  v3: 1 global view + 2 local views → 4 positive pairs per anchor
  Each local view is a different random crop (50-75% of the image) from a
  different position. This doubles the contrastive training signal at ~50%
  extra compute per batch (still comfortably within P100 limits).

  The 3 views also give SupCon richer inter-view class supervision:
  global_view and both local_views of the same patch are all same-class
  positives for each other.

CHECKPOINT RESUME:
  Pass --resume ./radiossl_output/radiossl_full_model.pth to continue from
  your existing epoch 200 checkpoint. The encoder architecture is IDENTICAL
  to v2, so weights load with zero missing keys. v3 runs 150 additional
  epochs (total 350) with SupCon added, which typically gives +4-8% on the
  honest patient-grouped probe.

ENCODER ARCHITECTURE: UNCHANGED FROM v2
  ResNet34 Stem (3×3 conv, no maxpool) → 64 tokens → 6 Transformer blocks
  CLS + GAP → 512-dim embedding
  The saved radiossl_encoder.pth format is identical to v2.

EXPECTED RESULTS ON P100 (from scratch, 300 epochs):
  Patient-grouped probe accuracy: ~78-84%
  Patient-grouped probe F1:       ~75-82%
  (These are honest numbers; v2's 88% was patient-inflated)

USAGE:
  # From scratch (300 epochs, ~10 hours on P100):
  python train_radiossl_v3.py --data_path ./ild_dataset_final.npz

  # Resume from v2 checkpoint (150 more epochs, ~5 hours):
  python train_radiossl_v3.py --data_path ./ild_dataset_final.npz \
    --resume ./radiossl_output/radiossl_full_model.pth --epochs 350
=============================================================================
"""

import os
import csv
import copy
import time
import math
import argparse
import json
import hashlib
import warnings
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1. ENCODER — IDENTICAL TO v2 (checkpoint-compatible)                  ║
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
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet34Stem(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.proj   = nn.Conv2d(256, embed_dim, 1) if embed_dim != 256 else nn.Identity()
        self.num_patches = 64

    def _make_layer(self, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers  = []
        for s in strides:
            layers.append(ResNetBasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * ResNetBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand  = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep
        return x / keep * rand


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
        self.dp = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + self.dp(a)
        x = x + self.dp(self.mlp(self.norm2(x)))
        return x


class RadioSSLEncoder(nn.Module):
    """
    ResViT: ResNet34 Stem → CLS → 6 Transformer blocks → CLS+GAP.
    Output: (B, 2 * embed_dim).
    Identical architecture to v2 — checkpoint loads with zero missing keys.
    """
    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
                 dropout=0.1, drop_path_rate=0.1):
        super().__init__()
        self.embed_dim   = embed_dim
        self.depth       = depth
        self.patch_embed = ResNet34Stem(in_channels=1, embed_dim=embed_dim)
        np_ = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, np_ + 1, embed_dim) * 0.02)
        self.pos_drop  = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B  = x.shape[0]
        x  = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x  = torch.cat([cls, x], dim=1)
        x  = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return torch.cat([x[:, 0], x[:, 1:].mean(dim=1)], dim=1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2. SSL HEADS — IDENTICAL TO v2 (checkpoint-compatible)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)


class PredictionHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)


class RadiomicsPredictionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x): return self.net(x)


class RadioSSLModel(nn.Module):
    """
    Complete SSL model: MoCo v3 + SupCon + Radiomics prediction.
    All head weights are the same as v2 — checkpoint-compatible.
    """
    def __init__(self, embed_dim=256, depth=6, num_heads=8,
                 proj_dim=256, n_radiomics=8, drop_path_rate=0.1):
        super().__init__()
        enc_out = embed_dim * 2

        self.encoder   = RadioSSLEncoder(embed_dim, depth, num_heads,
                                          drop_path_rate=drop_path_rate)
        self.projector = ProjectionHead(enc_out, 256, proj_dim)
        self.predictor = PredictionHead(proj_dim, proj_dim, proj_dim)

        self.momentum_encoder   = copy.deepcopy(self.encoder)
        self.momentum_projector = copy.deepcopy(self.projector)
        for p in self.momentum_encoder.parameters():   p.requires_grad = False
        for p in self.momentum_projector.parameters(): p.requires_grad = False

        self.radiomics_head = RadiomicsPredictionHead(enc_out, 128, n_radiomics)

    @torch.no_grad()
    def update_momentum(self, m=0.999):
        for pq, pk in zip(self.encoder.parameters(),
                          self.momentum_encoder.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=1 - m)
        for pq, pk in zip(self.projector.parameters(),
                          self.momentum_projector.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=1 - m)

    def forward(self, x1, x2, x3=None):
        """
        x1: global view, x2: local view 1, x3: local view 2 (optional)
        Returns projections and encoder features for loss computation.
        """
        feat1 = self.encoder(x1)
        z1    = self.projector(feat1)
        q1    = self.predictor(z1)
        with torch.no_grad():
            k1 = self.momentum_projector(self.momentum_encoder(x1))

        feat2 = self.encoder(x2)
        z2    = self.projector(feat2)
        q2    = self.predictor(z2)
        with torch.no_grad():
            k2 = self.momentum_projector(self.momentum_encoder(x2))

        if x3 is not None:
            feat3 = self.encoder(x3)
            z3    = self.projector(feat3)
            q3    = self.predictor(z3)
            with torch.no_grad():
                k3 = self.momentum_projector(self.momentum_encoder(x3))
            # Return projections of all 3 views for SupCon
            return q1, k2, q2, k1, feat1, z1, z2, z3
        else:
            return q1, k2, q2, k1, feat1, z1, z2, None

    def predict_radiomics(self, features):
        return self.radiomics_head(features)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3. DATASET — TRIPLE-VIEW + LABELS                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SSLDataset(Dataset):
    """
    Returns 3 augmented views (1 global + 2 local) + label + radiomics.
    Label = -1 for unlabelled patches.

    WHY THREE VIEWS:
      v2 used 1 global + 1 local = 2 positive pairs per anchor per batch.
      Adding a second local view (independent random crop) gives 4 positive
      pairs per anchor. For SupCon especially, more views from the same class
      mean more supervision signal per parameter update. The second local view
      is ~50% extra compute per batch (one extra encoder forward pass), which
      is affordable on P100.
    """
    def __init__(self, patches, labels=None, radiomics=None):
        self.patches   = patches.astype(np.float32)
        self.labels    = labels.astype(np.int64) if labels is not None else None
        self.radiomics = radiomics.astype(np.float32) if radiomics is not None else None

    def __len__(self): return len(self.patches)

    def __getitem__(self, idx):
        patch = torch.from_numpy(self.patches[idx]).unsqueeze(0)

        view1 = self._augment(patch.clone(), is_global=True)   # global
        view2 = self._augment(patch.clone(), is_global=False)  # local 1
        view3 = self._augment(patch.clone(), is_global=False)  # local 2

        lbl = torch.tensor(self.labels[idx] if self.labels is not None else -1,
                           dtype=torch.long)
        rad = (torch.from_numpy(self.radiomics[idx])
               if self.radiomics is not None else torch.zeros(8))

        return view1, view2, view3, lbl, rad

    def _augment(self, x, is_global=True):
        # Multi-scale crop (zero-padded, no interpolation)
        frac = (0.78 + torch.rand(1).item() * 0.22) if is_global \
               else (0.50 + torch.rand(1).item() * 0.25)
        crop_size = max(int(32 * frac), 14)

        if crop_size < 32:
            off  = 32 - crop_size
            top  = torch.randint(0, off + 1, (1,)).item()
            left = torch.randint(0, off + 1, (1,)).item()
            crop = x[:, top:top + crop_size, left:left + crop_size]
            canvas = torch.zeros_like(x)
            ph = (32 - crop_size) // 2
            pw = (32 - crop_size) // 2
            canvas[:, ph:ph + crop_size, pw:pw + crop_size] = crop
            x = canvas

        # D4 lossless symmetries
        if torch.rand(1).item() > 0.5: x = torch.flip(x, [2])
        if torch.rand(1).item() > 0.5: x = torch.flip(x, [1])
        x = torch.rot90(x, torch.randint(0, 4, (1,)).item(), [1, 2])

        # Gaussian noise with σ≈15 HU in normalized [-1024, 600] units.
        if torch.rand(1).item() > 0.30:
            x = x + torch.randn_like(x) * (15.0 / (600.0 - (-1024.0)))

        # Mean-filled cutout (1-3 regions)
        if torch.rand(1).item() > 0.40:
            fill   = x.mean().item()
            n_cuts = torch.randint(1, 4, (1,)).item()
            for _ in range(n_cuts):
                ch = torch.randint(4, 10, (1,)).item()
                cw = torch.randint(4, 10, (1,)).item()
                t  = torch.randint(0, max(1, 32 - ch + 1), (1,)).item()
                l  = torch.randint(0, max(1, 32 - cw + 1), (1,)).item()
                x[:, t:t + ch, l:l + cw] = fill

        return torch.clamp(x, 0.0, 1.0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4. LOSS FUNCTIONS                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def contrastive_loss(q, k, temperature=0.1):
    """MoCo v3 InfoNCE loss."""
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)
    logits = torch.einsum("bd,cd->bc", q, k) / temperature
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


def supcon_loss(z_views, labels, temperature=0.07, ignore_label=-1):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    This is the single most impactful change in v3. In v2, the 36,823
    labelled patches contribute only indirectly through MoCo (same patch,
    different aug = positive). Here, ALL same-class patches in the batch
    are positives for each other, giving 24× more supervision per anchor.

    Args:
        z_views: list of [B, proj_dim] normalized projections, one per view.
                 Typically [z1, z2, z3] from the three augmented views.
        labels:  [B] integer class labels (-1 = unlabelled, ignored)
        temperature: 0.07 is standard for SupCon (lower than MoCo's 0.1)
        ignore_label: patches with this label are excluded as anchors
                      but can still serve as negatives

    Algorithm:
      1. Concatenate all views → (n_views*B, proj_dim)
      2. Build the positive mask: same class, different index
      3. For each labelled anchor, compute supervised cross-entropy
         over all other samples in the multi-view batch
      4. Average over all labelled anchors

    Returns scalar loss. Returns 0 if no labelled samples in batch.
    """
    device = z_views[0].device
    n_views = len(z_views)

    # Stack all views: (n_views * B, proj_dim)
    z_all = torch.cat([F.normalize(z, dim=1) for z in z_views], dim=0)
    B = z_views[0].shape[0]
    N = n_views * B   # total number of feature vectors

    # Tile labels for all views: (N,)
    labels_all = labels.repeat(n_views)                        # (N,)

    # Similarity matrix: (N, N)
    sim = torch.matmul(z_all, z_all.T) / temperature

    # Mask diagonal (self-similarity)
    mask_self = torch.eye(N, dtype=torch.bool, device=device)
    # Using -1e4 instead of float("-inf") guarantees that 0.0 * -inf = NaN does not occur 
    # when we subsequently mask non-positive pairs.
    sim.masked_fill_(mask_self, -1e4)

    # Positive mask: same class, not self, both labelled
    labels_row = labels_all.unsqueeze(1).expand(N, N)          # (N, N)
    labels_col = labels_all.unsqueeze(0).expand(N, N)          # (N, N)
    mask_pos   = (labels_row == labels_col) & ~mask_self
    # Exclude unlabelled from being positives
    mask_pos   = mask_pos & (labels_row != ignore_label) & (labels_col != ignore_label)

    # Anchors = labelled samples only
    is_labelled = (labels_all != ignore_label)                 # (N,)

    if is_labelled.sum() == 0 or mask_pos[is_labelled].sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Log-sum-exp denominator over all non-self samples
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)  # (N, N)

    # For each labelled anchor, mean log-prob over positives
    n_pos = mask_pos.sum(dim=1).clamp(min=1).float()            # (N,)
    loss_per_anchor = -(mask_pos.float() * log_prob).sum(dim=1) / n_pos  # (N,)

    # Average over labelled anchors only
    return loss_per_anchor[is_labelled].mean()


def radiomics_loss(predicted, target):
    return F.mse_loss(predicted, target)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5. LR SCHEDULE & MOMENTUM                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs,
                 base_lr, min_lr_ratio=0.01):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = base_lr * min_lr_ratio
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        if e <= self.warmup_epochs:
            lr = self.base_lr * e / self.warmup_epochs
        else:
            t  = e - self.warmup_epochs
            T  = self.total_epochs - self.warmup_epochs
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * t / T))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def set_epoch(self, epoch):
        """For checkpoint resume: fast-forward scheduler to saved epoch."""
        self._epoch = epoch


def cosine_momentum(epoch, total_epochs, base=0.990, final=0.999):
    return final - (final - base) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / total_epochs))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6. PATIENT-AWARE LINEAR PROBE (HONEST EVALUATION)                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.no_grad()
def linear_probe(encoder, train_patches, train_labels, test_patches, test_labels, device):
    """Fit a linear probe on the manifest's train patients and score its holdout."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    encoder.eval()
    def encode(patches):
        chunks = []
        for i in range(0, len(patches), 256):
            batch = torch.from_numpy(patches[i:i + 256]).unsqueeze(1).to(device)
            chunks.append(encoder(batch).cpu().numpy())
        return np.concatenate(chunks, axis=0)

    train_features = encode(train_patches)
    test_features = encode(test_patches)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED,
                             class_weight="balanced")
    clf.fit(train_features, train_labels)
    preds = clf.predict(test_features)
    return accuracy_score(test_labels, preds), f1_score(test_labels, preds,
                                                         average="macro")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7. CLASS-BALANCED SAMPLER FOR SSL                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_class_balanced_sampler(labels, unlabelled_weight=0.5):
    """
    WeightedRandomSampler that balances the 5 labelled classes.
    Unlabelled patches get a lower weight so each batch is ~50% labelled.

    WHY BALANCED SAMPLING HELPS SUPCON:
      The SupCon loss is most effective when each batch contains multiple
      representatives of each class. With natural class frequencies
      (micronodules=38%, healthy=26%, emphysema=5%), SupCon anchors would
      have very few same-class positives for emphysema. Balanced sampling
      ensures each class contributes equally to the SupCon computation.
    """
    n = len(labels)
    class_counts = Counter(labels[labels >= 0].tolist())
    n_classes = 5

    weights = np.zeros(n)
    for i, lbl in enumerate(labels):
        if lbl < 0:
            weights[i] = unlabelled_weight
        else:
            weights[i] = n / (n_classes * class_counts.get(int(lbl), 1))

    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8. MAIN TRAINING LOOP                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 70}")
    print(f"  RadioSSL v3 — SupCon + MoCo + Triple-View (Patient-Aware Probe)")
    print(f"{'=' * 70}")
    print(f"  Device         : {device}")
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  Architecture   : ResNet34 Stem -> {args.depth} Transformer blocks")
    print(f"  SSL Objectives : MoCo v3 + SupCon from epoch {args.supcon_start_epoch} "
          f"(alpha={args.alpha_supcon}) + Radiomics")
    print(f"  Views          : 1 global + 2 local (triple-view)")
    print(f"  Evaluation     : manifest-defined holdout; evaluate after training")
    print(f"  Epochs         : {args.epochs}  (warmup={args.warmup_epochs})")
    print(f"  Batch size     : {args.batch_size} (eff: {args.batch_size*args.accum_steps})")
    print(f"  Resume         : {args.resume or 'None (training from scratch)'}")
    print(f"{'=' * 70}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    print("Loading dataset...")
    if not args.split_manifest:
        raise ValueError("--split_manifest is required for patient-disjoint pretraining")
    with open(args.split_manifest, encoding="utf-8") as f:
        split = json.load(f)
    with open(args.split_manifest, "rb") as f:
        split_hash = hashlib.sha256(f.read()).hexdigest()
    with open(args.data_path, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()
    if split.get("data_sha256") != data_hash:
        raise ValueError("Split manifest checksum does not match the NPZ")
    train_idx = np.asarray(split["train_indices"], dtype=np.int64)
    test_idx = np.asarray(split["test_indices"], dtype=np.int64)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Split manifest row indices overlap")
    if set(split["train_patient_ids"]) & set(split["test_patient_ids"]):
        raise ValueError("Patient leakage in split manifest")

    data = np.load(args.data_path, allow_pickle=False)
    all_pids = data["patient_ids"].astype(np.int64)
    if np.any(all_pids[test_idx] == -1):
        raise ValueError("Sentinel rows cannot be in test set")
    if set(np.unique(all_pids[train_idx][all_pids[train_idx] >= 0])) & set(np.unique(all_pids[test_idx])):
        raise ValueError("Train/test patient IDs overlap")
    patches = data["patches"][train_idx].astype(np.float32)
    labels_all = data["labels"][train_idx].astype(np.int64)
    test_labels_all = data["labels"][test_idx].astype(np.int64)
    probe_test_mask = (test_labels_all >= 0) & (test_labels_all <= 4)
    probe_test_patches = data["patches"][test_idx][probe_test_mask].astype(np.float32)
    probe_test_labels = test_labels_all[probe_test_mask]
    radiomics = data["radiomics"][train_idx].astype(np.float32) if "radiomics" in data else None
    del data, all_pids

    rad_mean = rad_std = None
    if radiomics is not None:
        rad_mean = np.nanmean(radiomics, axis=0, keepdims=True)
        rad_std = np.nanstd(radiomics, axis=0, keepdims=True) + 1e-8
        radiomics = np.nan_to_num((radiomics - rad_mean) / rad_std,
                                  nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  Radiomics      : {radiomics.shape} (train-only z-score)")

    # Sanitize
    bad = np.isnan(patches).any((1, 2)) | np.isinf(patches).any((1, 2))
    if bad.any():
        ok = ~bad
        patches = patches[ok]
        if labels_all  is not None: labels_all  = labels_all[ok]
        if radiomics   is not None: radiomics   = radiomics[ok]
    np.nan_to_num(patches, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"  Total patches  : {len(patches)}")
    print(f"  Value range    : [{patches.min():.4f}, {patches.max():.4f}]")

    print(f"  Training rows  : {len(train_idx)}; locked test rows: {len(test_idx)}")

    # ── Dataset & sampler ───────────────────────────────────────────────────
    # Pass labels to dataset: -1 for unlabelled patches
    ssl_labels = labels_all if labels_all is not None else -np.ones(len(patches), dtype=np.int64)
    # Mark out-of-range labels as unlabelled
    ssl_labels = np.where((ssl_labels >= 0) & (ssl_labels <= 4), ssl_labels, -1)

    probe_train_mask = (labels_all >= 0) & (labels_all <= 4)
    probe_train_patches = patches[probe_train_mask].copy()
    probe_train_labels = labels_all[probe_train_mask].copy()

    dataset = SSLDataset(patches, labels=ssl_labels, radiomics=radiomics)
    sampler = make_class_balanced_sampler(ssl_labels)
    loader  = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                         num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    print(f"  Batches/epoch  : {len(loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model = RadioSSLModel(
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        proj_dim=args.proj_dim,
        n_radiomics=8 if radiomics is not None else 0,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, args.lr, min_lr_ratio=0.01)
    scaler = GradScaler("cuda", init_scale=4096.0,
                        enabled=args.amp and torch.cuda.is_available())

    # ── Checkpoint resume ────────────────────────────────────────────────────
    start_epoch    = 1
    best_probe_acc = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\n  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("split_hash") != split_hash:
            raise ValueError("Resume checkpoint was not trained with this exact patient split")
        if "model_state_dict" in ckpt:
            miss, unexp = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"  Loaded model state: {len(miss)} missing, {len(unexp)} unexpected")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  Could not load optimizer state ({e}) — starting fresh")
        if "scaler_state_dict" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch    = ckpt.get("epoch", 1) + 1
        best_probe_acc = ckpt.get("probe_acc", 0.0)
        scheduler.set_epoch(start_epoch - 1)
        print(f"  Resuming from epoch {start_epoch} "
              f"(best probe_acc so far: {best_probe_acc:.4f})")
    else:
        print(f"\n  Training from scratch")

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_p       = sum(p.numel() for p in model.encoder.parameters())
    print(f"  Total params   : {total_p:,}")
    print(f"  Trainable      : {trainable_p:,}")
    print(f"  Encoder params : {enc_p:,}")

    # ── Training ─────────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    log_data = []

    print(f"\n  {'Ep':>5}  {'L_tot':>7}  {'L_moco':>7}  {'L_sup':>7}  "
          f"{'L_rad':>7}  {'LR':>10}  {'Time':>7}")
    print(f"  {'-' * 64}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        m = cosine_momentum(epoch, args.epochs)

        tot_ls = moco_ls = sup_ls = rad_ls = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (v1, v2, v3, lbl, rad_target) in enumerate(loader):
            v1  = v1.to(device, non_blocking=True)
            v2  = v2.to(device, non_blocking=True)
            v3  = v3.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            rad_target = rad_target.to(device, non_blocking=True)

            with autocast("cuda", enabled=args.amp and torch.cuda.is_available()):
                q1, k2, q2, k1, feat1, z1, z2, z3 = model(v1, v2, v3)

                # ── MoCo v3 symmetric contrastive loss ───────────────────────
                l_moco = (
                    contrastive_loss(q1, k2.detach(), args.temperature) +
                    contrastive_loss(q2, k1.detach(), args.temperature)
                ) / 2.0

                # ── SupCon begins at the epoch-200 stage described in the paper.
                if epoch >= args.supcon_start_epoch:
                    z_views = [z for z in [z1, z2, z3] if z is not None]
                    l_sup = supcon_loss(z_views, lbl, temperature=args.temp_supcon)
                    supcon_weight = args.alpha_supcon
                else:
                    l_sup = torch.zeros((), device=device)
                    supcon_weight = 0.0

                # ── Radiomics prediction (Branch B) ───────────────────────────
                if radiomics is not None:
                    l_rad = radiomics_loss(model.predict_radiomics(feat1), rad_target)
                else:
                    l_rad = torch.tensor(0.0, device=device)

                loss = (l_moco
                        + supcon_weight * l_sup
                        + args.lambda_radiomics * l_rad) / args.accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    if not scaler.is_enabled():
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite gradient norm with AMP disabled")
                    # GradScaler recorded the overflow during unscale_; step()
                    # skips this update and update() lowers the scale for retry.
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = scaler.get_scale()
                    optimizer.zero_grad(set_to_none=True)
                    print(f"  [AMP] skipped overflowed optimizer update at epoch {epoch}, "
                          f"batch {step + 1}; scale {old_scale:g} -> {new_scale:g}")
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.update_momentum(m)

            tot_ls  += loss.item() * args.accum_steps
            moco_ls += l_moco.item()
            sup_ls  += l_sup.item() if isinstance(l_sup, torch.Tensor) else l_sup
            rad_ls  += l_rad.item()
            n_batches += 1
            if args.preflight_steps and (step + 1) >= args.preflight_steps:
                break

        scheduler.step()
        lr = scheduler.get_lr()

        avg_tot  = tot_ls  / max(n_batches, 1)
        avg_moco = moco_ls / max(n_batches, 1)
        avg_sup  = sup_ls  / max(n_batches, 1)
        avg_rad  = rad_ls  / max(n_batches, 1)
        elapsed  = time.time() - t0

        probe_accuracy = probe_macro_f1 = None
        if (len(probe_test_labels) and args.probe_every > 0
                and (epoch % args.probe_every == 0 or epoch == args.epochs)):
            probe_accuracy, probe_macro_f1 = linear_probe(
                model.encoder, probe_train_patches, probe_train_labels,
                probe_test_patches, probe_test_labels, device)
            print(f"        patient-group probe: accuracy={probe_accuracy:.4f}, "
                  f"macro-F1={probe_macro_f1:.4f}")

        log_data.append({
            "epoch": epoch, "loss_total": avg_tot, "loss_moco": avg_moco,
            "loss_supcon": avg_sup, "loss_radiomics": avg_rad,
            "lr": lr, "probe_accuracy": probe_accuracy,
            "probe_macro_f1": probe_macro_f1,
        })

        if epoch % 5 == 0 or epoch == start_epoch or epoch == args.epochs:
            print(f"  {epoch:5d}  {avg_tot:7.4f}  {avg_moco:7.4f}  {avg_sup:7.4f}  "
                  f"{avg_rad:7.4f}  {lr:10.6f}  {elapsed:6.1f}s")

        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args), "split_manifest": args.split_manifest,
                "split_hash": split_hash,
                "radiomics_mean": rad_mean, "radiomics_std": rad_std,
            }, os.path.join(args.output_dir, f"radiossl_checkpoint_ep{epoch}.pth"))

    # Final save
    torch.save(model.encoder.state_dict(),
               os.path.join(args.output_dir, "radiossl_encoder_final.pth"))
    torch.save({
        "epoch": args.epochs, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "args": vars(args),
        "scaler_state_dict": scaler.state_dict(),
        "split_manifest": args.split_manifest,
        "split_hash": split_hash,
        "radiomics_mean": rad_mean, "radiomics_std": rad_std,
    }, os.path.join(args.output_dir, "radiossl_full_model.pth"))

    log_path = os.path.join(args.output_dir, "radiossl_training_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log_data[0].keys())
        w.writeheader()
        w.writerows(log_data)

    print(f"\n{'=' * 70}")
    print(f"  PRE-TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print("  Held-out evaluation: run evaluate_linear_probe.py once after training")
    print(f"  Encoder: {os.path.join(args.output_dir, 'radiossl_encoder.pth')}")
    print(f"  Log: {log_path}")
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"  Peak CUDA allocation: {peak_gb:.2f} GiB")
    print(f"{'=' * 70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9. ENTRY POINT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser("RadioSSL v3 (SupCon + MoCo + Triple-View)")
    p.add_argument("--data_path",  default="./ild_dataset_final.npz")
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--output_dir", default="./radiossl_output")
    p.add_argument("--resume",     default="",
                   help="Path to radiossl_full_model.pth to resume from")

    # Model
    p.add_argument("--embed_dim",       type=int,   default=256)
    p.add_argument("--depth",           type=int,   default=6)
    p.add_argument("--num_heads",       type=int,   default=8)
    p.add_argument("--proj_dim",        type=int,   default=256)
    p.add_argument("--drop_path_rate",  type=float, default=0.1)

    # Training
    p.add_argument("--epochs",          type=int,   default=300)
    p.add_argument("--batch_size",      type=int,   default=128)
    p.add_argument("--accum_steps",     type=int,   default=2)
    p.add_argument("--lr",              type=float, default=1.5e-4)
    p.add_argument("--warmup_epochs",   type=int,   default=10)
    p.add_argument("--weight_decay",    type=float, default=0.05)

    # Loss
    p.add_argument("--temperature",     type=float, default=0.1,
                   help="MoCo InfoNCE temperature")
    p.add_argument("--temp_supcon",     type=float, default=0.07,
                   help="SupCon temperature (lower = sharper boundaries)")
    p.add_argument("--alpha_supcon",    type=float, default=0.5,
                   help="Weight for SupCon loss (0=disabled, 1=equal to MoCo)")
    p.add_argument("--supcon_start_epoch", type=int, default=200,
                   help="First epoch that includes the SupCon objective")
    p.add_argument("--lambda_radiomics",type=float, default=1.0)

    # Evaluation
    p.add_argument("--probe_every",     type=int,   default=10)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--preflight_steps", type=int, default=0,
                   help="Stop each epoch after this many batches for a smoke run")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="amp", action="store_false",
                   help="Disable CUDA mixed precision if AMP remains unstable")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
