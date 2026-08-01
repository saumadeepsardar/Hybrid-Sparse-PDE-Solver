#!/usr/bin/env python3
"""
scripts/train_gnn.py  —  Train GNN configuration advisor for HAMLSS

Reads JSON Lines training data produced by:
    bin/run_dataset --output-jsonl data/training.jsonl --repeat 5

Outputs:
    models/gnn_v1.pt      (TorchScript for C++ inference)
    models/gnn_v1_acc.json (accuracy metrics)

Usage:
    python3 scripts/train_gnn.py \\
        --input  data/training.jsonl \\
        --output models/gnn_v1.pt    \\
        --epochs 200                 \\
        --lr 1e-3                    \\
        --hidden_dim 128             \\
        --layers 3

Dependencies:
    pip install torch torchvision numpy pandas scikit-learn tqdm
    pip install torch_geometric  # optional: for full graph GNN variant
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Imports with graceful fallback ─────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    print("ERROR: PyTorch not installed. Run:  pip install torch")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not found. Using manual normalisation.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kw: x  # no-op wrapper

FEATURE_DIM = 24   # matches MatrixFeatures::feature_dim()
N_STATES    = 3    # EASY=0, MODERATE=1, HARD=2
STATE_NAMES = ["EASY", "MODERATE", "HARD"]

# ── Dataset ────────────────────────────────────────────────────────────────

class SolverDataset(Dataset):
    """Load (features, labels, energy) triples from a JSON Lines file."""

    def __init__(self, jsonl_path: str, scaler=None):
        self.samples = []
        bad = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    feats  = obj["features"][:FEATURE_DIM]
                    # Pad if fewer than FEATURE_DIM fields
                    feats += [0.0] * (FEATURE_DIM - len(feats))
                    state  = int(obj.get("label_state",   0))
                    restart= float(obj.get("label_restart", 50))
                    dt     = float(obj.get("label_drop_tol", 1e-4))
                    amg    = float(obj.get("label_amg_str",  0.25))
                    energy = float(obj.get("energy_j",       1.0))
                    conv   = bool(obj.get("converged",       True))
                    # Skip non-converged samples (label is meaningless)
                    if not conv:
                        continue
                    self.samples.append({
                        "features" : feats,
                        "state"    : state,
                        "restart"  : restart,
                        "log_dt"   : math.log(max(dt, 1e-8)),
                        "amg_str"  : amg,
                        "energy"   : energy,
                    })
                except (KeyError, ValueError, json.JSONDecodeError):
                    bad += 1
        if bad:
            print(f"  Skipped {bad} malformed lines")

        # Normalise features
        X = np.array([s["features"] for s in self.samples], dtype=np.float32)
        # Replace NaN/Inf
        X = np.where(np.isfinite(X), X, 0.0)

        if scaler is None:
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            else:
                mean = X.mean(axis=0)
                std  = X.std(axis=0) + 1e-8
                X    = (X - mean) / std
                scaler = (mean, std)
        else:
            if SKLEARN_AVAILABLE:
                X = scaler.transform(X)
            else:
                mean, std = scaler
                X = (X - mean) / std

        self.scaler = scaler
        for i, s in enumerate(self.samples):
            s["features_norm"] = X[i].tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "x"      : torch.tensor(s["features_norm"], dtype=torch.float32),
            "state"  : torch.tensor(s["state"],         dtype=torch.long),
            "restart": torch.tensor([s["restart"] / 300.0], dtype=torch.float32),
            "log_dt" : torch.tensor([s["log_dt"] / 6.0],    dtype=torch.float32),
            "amg_str": torch.tensor([s["amg_str"]],          dtype=torch.float32),
            "energy" : torch.tensor([s["energy"]],           dtype=torch.float32),
        }


# ── Model ──────────────────────────────────────────────────────────────────

class HAMLSSNet(nn.Module):
    """
    Feature-vector MLP with multi-task output heads.
    Used when graph structure is not available (feature-only mode).
    For full GNN replace the encoder with torch_geometric.nn.SAGEConv layers.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 dropout: float = 0.2):
        super().__init__()
        # Shared encoder
        layers = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                  nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim),
                       nn.LayerNorm(hidden_dim),
                       nn.GELU(),
                       nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.head_state   = nn.Linear(hidden_dim, N_STATES)   # softmax
        self.head_restart = nn.Linear(hidden_dim, 1)           # in [0,1] → ×300
        self.head_log_dt  = nn.Linear(hidden_dim, 1)           # log-scale
        self.head_amg_str = nn.Linear(hidden_dim, 1)           # in [0,1]

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return (self.head_state(z),
                torch.sigmoid(self.head_restart(z)),
                self.head_log_dt(z),
                torch.sigmoid(self.head_amg_str(z)))


# ── Training ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device, lambda_energy=0.1):
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    mse     = nn.MSELoss()

    total_loss = 0.0
    n_correct  = 0
    n_total    = 0

    for batch in loader:
        x       = batch["x"].to(device)
        y_state = batch["state"].to(device)
        y_rest  = batch["restart"].to(device)
        y_dt    = batch["log_dt"].to(device)
        y_amg   = batch["amg_str"].to(device)
        y_eng   = batch["energy"].to(device)

        s_logits, r_pred, dt_pred, amg_pred = model(x)

        # State classification loss
        l_state   = ce_loss(s_logits, y_state)
        # Regression losses
        l_restart = mse(r_pred,   y_rest)
        l_dt      = mse(dt_pred,  y_dt)
        l_amg     = mse(amg_pred, y_amg)
        # Energy-weighted penalty: higher weight for high-energy configurations
        #   normalise energy by its mean to avoid scale dominance
        e_norm    = y_eng / (y_eng.mean() + 1e-8)
        l_energy  = (e_norm * l_state).mean()

        loss = l_state + 0.3*l_restart + 0.3*l_dt + 0.2*l_amg + lambda_energy*l_energy

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        total_loss += loss.item() * len(x)
        preds       = s_logits.argmax(dim=1)
        n_correct  += (preds == y_state).sum().item()
        n_total    += len(x)

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ce_loss   = nn.CrossEntropyLoss()
    n_correct = 0
    n_total   = 0
    total_ce  = 0.0
    energy_ratio_sum = 0.0

    for batch in loader:
        x       = batch["x"].to(device)
        y_state = batch["state"].to(device)
        y_eng   = batch["energy"].to(device)

        s_logits, _, _, _ = model(x)
        preds = s_logits.argmax(dim=1)
        n_correct  += (preds == y_state).sum().item()
        n_total    += len(x)
        total_ce   += ce_loss(s_logits, y_state).item() * len(x)

    return total_ce / n_total, n_correct / n_total


# ── Export to TorchScript ──────────────────────────────────────────────────

def export_torchscript(model, output_path: str):
    """
    Export the trained model as TorchScript for C++ inference.
    The C++ GNNAdvisor calls model.forward(features_tensor) and expects:
        Tuple[Tensor[3], Tensor[1], Tensor[1], Tensor[1]]
    """
    model.eval()
    example_input = torch.zeros(1, FEATURE_DIM)
    traced = torch.jit.trace(model, example_input)
    traced.save(output_path)
    print(f"Exported TorchScript model → {output_path}")

    # Verify round-trip
    loaded = torch.jit.load(output_path)
    out = loaded(example_input)
    print(f"  Forward pass OK: state_logits={out[0].shape}, "
          f"restart={out[1].shape}, log_dt={out[2].shape}, amg={out[3].shape}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train HAMLSS GNN advisor")
    parser.add_argument("--input",      default="data/training.jsonl")
    parser.add_argument("--output",     default="models/gnn_v1.pt")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--layers",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",     default="auto",
                        help="cpu | cuda | auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load dataset
    print(f"Loading dataset from: {args.input}")
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        print("Generate it first with:")
        print("  bin/run_dataset --output-jsonl data/training.jsonl --repeat 5")
        sys.exit(1)

    dataset = SolverDataset(args.input)
    print(f"  Loaded {len(dataset)} converged training samples")

    if len(dataset) < 50:
        print("WARNING: Very few samples (<50). Results may be unreliable.")

    # Print class distribution
    states = [dataset[i]["state"].item() for i in range(len(dataset))]
    for s, name in enumerate(STATE_NAMES):
        count = states.count(s)
        print(f"  {name}: {count} ({100*count/len(dataset):.1f}%)")

    # Train/val split
    n_val   = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # Model
    model = HAMLSSNet(
        input_dim  = FEATURE_DIM,
        hidden_dim = args.hidden_dim,
        n_layers   = args.layers,
        dropout    = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, hidden={args.hidden_dim}, "
          f"layers={args.layers}")

    # Optimiser with cosine annealing
    optimiser = optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=1e-5)

    # Training loop
    best_val_acc = 0.0
    best_epoch   = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output.replace(".pt", "_best.pt")

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>10} {'LR':>10}")
    print("-" * 65)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training",
                      disable=not TQDM_AVAILABLE):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimiser, device)
        val_loss, val_acc     = eval_epoch(model, val_loader, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {train_loss:>12.4f} {train_acc:>10.3f} "
                  f"{val_loss:>10.4f} {val_acc:>10.3f} {lr:>10.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), best_model_path)

    print(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")

    # Load best weights and export
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    export_torchscript(model, args.output)

    # Save accuracy metrics
    metrics = {
        "val_accuracy":  best_val_acc,
        "best_epoch":    best_epoch,
        "n_train":       n_train,
        "n_val":         n_val,
        "hidden_dim":    args.hidden_dim,
        "n_layers":      args.layers,
        "feature_dim":   FEATURE_DIM,
        "state_names":   STATE_NAMES,
    }
    metrics_path = args.output.replace(".pt", "_acc.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")
    print("\nDone. To use in HSPS C++:")
    print(f"  auto adv = std::make_shared<GNNAdvisor>(\"{args.output}\");")
    print( "  sel.set_ml_advisor(adv);")


if __name__ == "__main__":
    main()
