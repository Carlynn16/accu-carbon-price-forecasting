"""
LSTM and GRU regressors for ACCU price-change forecasting (Block C3).

Architecture
------------
Small recurrent nets: 1 layer, hidden=64, dropout=0.2, linear head.
Input: (batch, L, n_features) scaled feature sequences.
Output: scalar predicted h-step price change.

Training protocol (matches C2 tree models)
------------------------------------------
Phase 1 — fit on TRAIN sequences, early-stop on VAL loss (patience=10,
           max 100 epochs). Save best weights.
Phase 3 — refit on TRAIN+VAL for the same number of epochs as Phase 1
           selected, then evaluate ONCE on TEST.

Why we expect DL not to beat the random walk
---------------------------------------------
The dataset has ~1125 training samples after windowing, a 75% staleness rate
(most target changes are zero), and no persistent momentum signal beyond
a 1-day lag. A recurrent net needs far more data and cleaner signal than
this market provides to learn anything beyond "predict zero". The honest
result — DL ≈ random walk or worse — is the correct finding.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import build_results_table, compute_metrics
from src.sequences import build_sequences

# ── Hyper-parameters ──────────────────────────────────────────────────────────
SEED        = 42
HIDDEN      = 64
N_LAYERS    = 1
DROPOUT     = 0.2
BATCH_SIZE  = 32
LR          = 1e-3
MAX_EPOCHS  = 100
PATIENCE    = 10


def _set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lookup_rw(
    baseline_df: pd.DataFrame | None,
    h: int,
    split: str,
) -> tuple[float | None, float | None]:
    """Return (rw_rmse, rw_mae) for the given horizon/split from baseline_df."""
    if baseline_df is None:
        return None, None
    row = baseline_df[
        (baseline_df["model"] == "random_walk") &
        (baseline_df["horizon"] == h) &
        (baseline_df["split"] == split)
    ]
    if row.empty:
        return None, None
    return float(row["RMSE"].iloc[0]), float(row["MAE"].iloc[0])


# ── Model definitions ─────────────────────────────────────────────────────────

class LSTMRegressor(nn.Module):
    """
    Single-layer LSTM with a linear readout.

    At each time step the LSTM updates its hidden state h_t and cell c_t.
    We take h_T (the final hidden state after the full sequence) as the
    sequence representation and pass it through dropout + a linear layer.
    """

    def __init__(
        self,
        n_features: int,
        hidden:   int = HIDDEN,
        n_layers: int = N_LAYERS,
        dropout:  float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            # Dropout is only between layers; for n_layers=1 it has no effect
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)    # h_n: (n_layers, batch, hidden)
        out = self.drop(h_n[-1])       # final layer's hidden state
        return self.head(out)          # (batch, 1)


class GRURegressor(nn.Module):
    """
    Single-layer GRU with a linear readout.

    GRU merges the cell and hidden state into a single h_t, making it
    slightly simpler than LSTM. Otherwise identical training protocol.
    """

    def __init__(
        self,
        n_features: int,
        hidden:   int = HIDDEN,
        n_layers: int = N_LAYERS,
        dropout:  float = DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, h_n = self.gru(x)           # h_n: (n_layers, batch, hidden)
        out = self.drop(h_n[-1])
        return self.head(out)          # (batch, 1)


# ── Training helpers ──────────────────────────────────────────────────────────

def _to_tensors(split_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.from_numpy(split_dict["X"])                              # (N, L, F)
    y = torch.from_numpy(split_dict["y"].astype(np.float32)).unsqueeze(1)  # (N, 1)
    return X, y


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def _eval_loss(
    model:     nn.Module,
    X:         torch.Tensor,
    y:         torch.Tensor,
    criterion: nn.Module,
    device:    torch.device,
) -> float:
    model.eval()
    return criterion(model(X.to(device)), y.to(device)).item()


@torch.no_grad()
def _predict(model: nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    # Process in one shot (val/test sets are small enough)
    return model(X.to(device)).squeeze(1).cpu().numpy()


def _fit_phase1(
    model_class,
    n_features: int,
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
) -> tuple[nn.Module, int]:
    """
    Phase 1: train on (X_tr, y_tr), select best weights by val MSE.
    Returns (best_model, best_epoch_count).
    """
    _set_seeds()
    model     = model_class(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    loader    = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=False,   # preserve chronological order within each batch
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    best_epoch    = 1

    for epoch in range(1, MAX_EPOCHS + 1):
        _train_epoch(model, loader, optimizer, criterion, device)
        val_loss = _eval_loss(model, X_va, y_va, criterion, device)

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch    = epoch
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def _fit_phase3(
    model_class,
    n_features: int,
    X_trval:   torch.Tensor,
    y_trval:   torch.Tensor,
    n_epochs:  int,
    device:    torch.device,
) -> nn.Module:
    """
    Phase 3: refit on TRAIN+VAL for exactly n_epochs (same as Phase 1 selected).
    """
    _set_seeds()
    model     = model_class(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    loader    = DataLoader(
        TensorDataset(X_trval, y_trval),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    for _ in range(n_epochs):
        _train_epoch(model, loader, optimizer, criterion, device)
    return model


# ── Per-architecture runner ───────────────────────────────────────────────────

def _run_one_arch(
    model_class,
    model_name:  str,
    feat_train:  pd.DataFrame,
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    horizons:    Sequence[int],
    baseline_df: pd.DataFrame | None,
    device:      torch.device,
) -> list[dict]:
    records: list[dict] = []

    for h in horizons:
        print(f"  {model_name} h={h}: building sequences ...")
        seqs = build_sequences(feat_train, feat_val, feat_test, h)
        tr   = seqs["train"]
        va   = seqs["val"]
        te   = seqs["test"]
        n_f  = tr["n_features"]

        X_tr, y_tr = _to_tensors(tr)
        X_va, y_va = _to_tensors(va)
        X_te, _    = _to_tensors(te)

        rw_rmse_val, rw_mae_val = _lookup_rw(baseline_df, h, "val")
        rw_rmse_te,  rw_mae_te  = _lookup_rw(baseline_df, h, "test")

        # Phase 1 — fit on TRAIN, select by VAL
        print(f"  {model_name} h={h}: training (patience={PATIENCE}, max={MAX_EPOCHS}) ...")
        best_model, best_epoch = _fit_phase1(
            model_class, n_f, X_tr, y_tr, X_va, y_va, device
        )

        pred_val = _predict(best_model, X_va, device)
        m_val    = compute_metrics(
            pred_val, va["y"], va["price_anchor"],
            rw_rmse=rw_rmse_val, rw_mae=rw_mae_val,
        )
        records.append({"model": model_name, "horizon": h, "split": "val", **m_val})

        # Phase 3 — refit on TRAIN+VAL, evaluate TEST once
        X_trval = torch.cat([X_tr, X_va], dim=0)
        y_trval = torch.cat([y_tr, y_va], dim=0)
        print(f"  {model_name} h={h}: refitting on train+val ({best_epoch} epochs) ...")
        final_model = _fit_phase3(model_class, n_f, X_trval, y_trval, best_epoch, device)

        pred_te = _predict(final_model, X_te, device)
        m_te    = compute_metrics(
            pred_te, te["y"], te["price_anchor"],
            rw_rmse=rw_rmse_te, rw_mae=rw_mae_te,
        )
        records.append({"model": model_name, "horizon": h, "split": "test", **m_te})

        print(
            f"  {model_name} h={h}: stopped ep {best_epoch} | "
            f"val skill {m_val['RMSE_skill_%']:+.1f}%  "
            f"test skill {m_te['RMSE_skill_%']:+.1f}%"
        )

    return records


# ── Public entry point ────────────────────────────────────────────────────────

def run_dl_models(
    feat_train:  pd.DataFrame,
    feat_val:    pd.DataFrame,
    feat_test:   pd.DataFrame,
    horizons:    Sequence[int] = (1, 7, 30),
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run LSTM and GRU for all horizons.
    Returns a tidy results DataFrame (same schema as C2).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DL device: {device}")

    records: list[dict] = []
    for cls, name in [(LSTMRegressor, "lstm"), (GRURegressor, "gru")]:
        print(f"\n--- {name.upper()} ---")
        records.extend(_run_one_arch(
            cls, name, feat_train, feat_val, feat_test, horizons, baseline_df, device
        ))

    return build_results_table(records)
