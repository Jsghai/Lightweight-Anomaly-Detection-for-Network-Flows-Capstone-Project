import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Reproducibility 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Training hyperparameters 
LEARNING_RATE = 1e-3
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
PATIENCE      = 10      # early stopping patience (epochs)

# ── Model definitions 

MODEL_CONFIGS = {
    "small":  [64,  32,  16],
    "medium": [128, 64,  32],
    "large":  [256, 128, 64],
}


class Autoencoder(nn.Module):
    """
    Symmetric fully-connected autoencoder.

    Architecture (example for hidden_dims=[128, 64, 32]):
      Encoder: input_dim -> 128 -> 64 -> 32  (bottleneck)
      Decoder: 32 -> 64 -> 128 -> input_dim

    Hidden layers use ReLU. Output layer uses linear activation.
    Loss: Mean Squared Error (reconstruction loss).
    """

    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # ── Decoder (mirror of encoder) ───────────────────────────────────────
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))   # linear output
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Training utilities ────────────────────────────────────────────────────────

def reconstruction_error(model, x: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE between input and reconstruction."""
    with torch.no_grad():
        x_hat = model(x)
        return ((x - x_hat) ** 2).mean(dim=1)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def train_model(model_name: str,
                input_dim: int,
                X_train: np.ndarray,
                X_val: np.ndarray,
                output_dir: str,
                device: torch.device):
    """Train a single autoencoder and save the best checkpoint."""

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()} autoencoder")
    print(f"{'='*60}")

    hidden_dims = MODEL_CONFIGS[model_name]
    model = Autoencoder(input_dim, hidden_dims).to(device)

    print(f"  Architecture  : {input_dim} -> "
          f"{' -> '.join(map(str, hidden_dims))} -> "
          f"{' -> '.join(map(str, reversed(hidden_dims[:-1])))} -> {input_dim}")
    print(f"  Parameters    : {model.count_parameters():,}")
    print(f"  Model size    : {model.count_parameters() * 4 / 1024:.1f} KB "
          f"({model.count_parameters() * 4 / (1024**2):.3f} MB)")

    # ── Data loaders ──────────────────────────────────────────────────────────
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t),
                              batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    history = {"train": [], "val": []}

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer,
                                     criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Print every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} │ "
                  f"train loss: {train_loss:.6f} │ "
                  f"val loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_counter = 0
            # Save best model weights
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f"{model_name}_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best epoch: {best_epoch})")
                break

    training_time = time.time() - start_time

    print(f"\n  Best val loss : {best_val_loss:.6f} (epoch {best_epoch})")
    print(f"  Training time : {training_time:.1f}s")

    # Save training history and metadata
    np.save(os.path.join(output_dir, f"{model_name}_history.npy"),
            np.array(history["val"]))

    metadata = {
        "model_name":    model_name,
        "input_dim":     input_dim,
        "hidden_dims":   hidden_dims,
        "parameters":    model.count_parameters(),
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "training_time": training_time,
    }
    np.save(os.path.join(output_dir, f"{model_name}_meta.npy"),
            np.array([metadata], dtype=object))

    return model, metadata


# ── Inference time benchmark ──────────────────────────────────────────────────

def benchmark_inference(model, X_test: np.ndarray, device: torch.device,
                        n_runs: int = 5) -> float:
    """Measure average CPU inference time over n_runs passes."""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Warm-up
    with torch.no_grad():
        _ = model(X_t[:100])

    times = []
    for _ in range(n_runs):
        t0 = time.time()
        with torch.no_grad():
            _ = model(X_t)
        times.append(time.time() - t0)

    return float(np.mean(times))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, output_dir: str):
    device = torch.device("cpu")   # CPU only – lightweight by design
    print(f"Device: {device}")

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    input_dim = X_train.shape[1]

    print(f"  X_train : {X_train.shape}")
    print(f"  X_val   : {X_val.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Input dim: {input_dim}")

    all_metadata = {}

    # Train all three models
    for model_name in ["small", "medium", "large"]:
        model, meta = train_model(
            model_name, input_dim, X_train, X_val, output_dir, device
        )

        # Benchmark inference on full test set
        inf_time = benchmark_inference(model, X_test, device)
        meta["inference_time_s"] = inf_time
        print(f"  Inference time (full test set): {inf_time:.3f}s "
              f"({inf_time/len(X_test)*1000:.4f} ms/sample)")

        all_metadata[model_name] = meta

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EFFICIENCY SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<10} {'Params':>10} {'Size (KB)':>12} "
          f"{'Val Loss':>12} {'Train (s)':>10} {'Inf (ms/sample)':>17}")
    print(f"  {'-'*73}")
    for name, m in all_metadata.items():
        size_kb = m["parameters"] * 4 / 1024
        inf_ms  = m["inference_time_s"] / len(X_test) * 1000
        print(f"  {name:<10} {m['parameters']:>10,} {size_kb:>12.1f} "
              f"{m['best_val_loss']:>12.6f} "
              f"{m['training_time']:>10.1f} {inf_ms:>17.4f}")

    print(f"\nAll models saved to: {output_dir}/")
    print("Next step: run evaluate.py to compute detection metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train three autoencoder models on UNSW-NB15 normal traffic."
    )
    parser.add_argument("--data_dir",   default="data/processed",
                        help="Directory containing .npy files from preprocess.py")
    parser.add_argument("--output_dir", default="models",
                        help="Directory to save trained models (default: models/)")
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)