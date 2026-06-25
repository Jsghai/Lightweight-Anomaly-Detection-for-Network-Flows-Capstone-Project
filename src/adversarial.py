import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Model definition (same as autoencoder.py) ─────────────────────────────────
MODEL_CONFIGS = {
    "small":  [64,  32,  16],
    "medium": [128, 64,  32],
    "large":  [256, 128, 64],
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list):
        super().__init__()
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for h_dim in reversed(hidden_dims[:-1]):
            decoder_layers += [nn.Linear(in_dim, h_dim), nn.ReLU()]
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_model(model_name: str, input_dim: int, model_dir: str) -> Autoencoder:
    model = Autoencoder(input_dim, MODEL_CONFIGS[model_name])
    path  = os.path.join(model_dir, f"{model_name}_best.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


# ── Adversarial attack: FGSM-style evasion ───────────────────────────────────

def fgsm_evasion(model: Autoencoder,
                 X_attack: torch.Tensor,
                 epsilon: float,
                 n_steps: int = 10) -> torch.Tensor:
    """
    Iterative FGSM evasion attack.

    Goal: MINIMISE reconstruction error so attack samples look normal.
    At each step, move in the direction that DECREASES reconstruction error
    (i.e., negative gradient direction).

    Features are clipped to [0, 1] after each step to keep them valid
    (all features were Min-Max scaled to this range in preprocessing).

    Args:
        model:    trained autoencoder
        X_attack: attack samples tensor, shape (N, input_dim)
        epsilon:  total perturbation budget (L-inf norm)
        n_steps:  number of gradient steps

    Returns:
        X_adv: adversarially perturbed samples, same shape as X_attack
    """
    step_size = epsilon / n_steps
    X_adv = X_attack.clone().detach()

    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        X_hat = model(X_adv)
        # Reconstruction error (we want to MINIMISE this)
        loss = ((X_adv - X_hat) ** 2).mean()
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Move in the NEGATIVE gradient direction to reduce error
            grad_sign = X_adv.grad.sign()
            X_adv = X_adv - step_size * grad_sign
            # Clip to valid feature range [0, 1]
            X_adv = torch.clamp(X_adv, 0.0, 1.0)

    return X_adv.detach()


def detection_rate(model: Autoencoder,
                   X: torch.Tensor,
                   threshold: float) -> float:
    """Fraction of samples with reconstruction error >= threshold."""
    with torch.no_grad():
        X_hat  = model(X)
        errors = ((X - X_hat) ** 2).mean(dim=1)
    return float((errors >= threshold).float().mean())


def get_threshold_from_val(model: Autoencoder,
                            X_val: np.ndarray,
                            percentile: float = 99.0) -> float:
    """Get detection threshold from normal validation data."""
    with torch.no_grad():
        X_t   = torch.tensor(X_val, dtype=torch.float32)
        X_hat = model(X_t)
        errors = ((X_t - X_hat) ** 2).mean(dim=1).numpy()
    return float(np.percentile(errors, percentile))


# ── Denoising autoencoder defence ────────────────────────────────────────────

def train_denoising_autoencoder(input_dim: int,
                                 hidden_dims: list,
                                 X_train: np.ndarray,
                                 X_val: np.ndarray,
                                 noise_std: float,
                                 output_path: str,
                                 max_epochs: int = 100,
                                 patience: int = 10) -> Autoencoder:
    """
    Train a denoising autoencoder.
    Inputs are corrupted with Gaussian noise; target is the clean input.
    This forces the model to learn robust representations.
    """
    print(f"\n  Training denoising autoencoder (noise_std={noise_std})...")
    device = torch.device("cpu")
    model  = Autoencoder(input_dim, hidden_dims).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    train_ds = TensorDataset(X_train_t)
    val_ds   = TensorDataset(X_val_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    best_epoch       = 0

    for epoch in range(1, max_epochs + 1):
        # Training: corrupt input, reconstruct clean
        model.train()
        total_loss = 0.0
        for (x_clean,) in train_loader:
            x_noisy = x_clean + noise_std * torch.randn_like(x_clean)
            x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
            optimizer.zero_grad()
            x_hat = model(x_noisy)
            loss  = criterion(x_hat, x_clean)   # target is clean
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_clean)

        # Validation: also corrupt
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_clean,) in val_loader:
                x_noisy = x_clean + noise_std * torch.randn_like(x_clean)
                x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
                x_hat   = model(x_noisy)
                val_loss += criterion(x_hat, x_clean).item() * len(x_clean)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} │ val loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} "
                      f"(best: {best_epoch})")
                break

    model.load_state_dict(torch.load(output_path, map_location="cpu"))
    model.eval()
    print(f"  Best val loss: {best_val_loss:.6f}")
    return model


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"small": "#4C72B0", "medium": "#DD8452", "large": "#55A868"}


def plot_evasion_results(epsilons: list,
                         results: dict,
                         output_dir: str):
    """Detection rate vs epsilon for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, rates in results.items():
        ax.plot(epsilons, rates,
                label=model_name.capitalize(),
                color=COLORS[model_name],
                marker="o", lw=2)

    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, alpha=0.5,
               label="50% detection")
    ax.set_xlabel("Perturbation budget (epsilon)")
    ax.set_ylabel("Detection rate")
    ax.set_title("Evasion Attack: Detection Rate vs Perturbation Budget")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adversarial_evasion.png"), dpi=150)
    plt.close()
    print("  Saved: adversarial_evasion.png")


def plot_defence_comparison(epsilons: list,
                             vanilla_rates: list,
                             denoising_rates: list,
                             output_dir: str):
    """Vanilla vs denoising defence for medium model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epsilons, vanilla_rates,
            label="Standard autoencoder",
            color=COLORS["medium"], marker="o", lw=2)
    ax.plot(epsilons, denoising_rates,
            label="Denoising autoencoder (defence)",
            color="#C44E52", marker="s", lw=2, linestyle="--")

    ax.axhline(y=0.5, color="gray", linestyle=":", lw=1, alpha=0.5)
    ax.set_xlabel("Perturbation budget (epsilon)")
    ax.set_ylabel("Detection rate on adversarial samples")
    ax.set_title("Defence Comparison: Standard vs Denoising Autoencoder (Medium)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "adversarial_defence.png"), dpi=150)
    plt.close()
    print("  Saved: adversarial_defence.png")


def plot_error_shift(model: Autoencoder,
                     X_attack_clean: torch.Tensor,
                     X_attack_adv: torch.Tensor,
                     threshold: float,
                     model_name: str,
                     output_dir: str):
    """Show how adversarial perturbation shifts reconstruction errors."""
    with torch.no_grad():
        err_clean = ((X_attack_clean - model(X_attack_clean)) ** 2
                     ).mean(dim=1).numpy()
        err_adv   = ((X_attack_adv   - model(X_attack_adv))   ** 2
                     ).mean(dim=1).numpy()

    clip = np.percentile(np.concatenate([err_clean, err_adv]), 99)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(err_clean, bins=80, alpha=0.6, color="#DD8452",
            label="Original attack", density=True, range=(0, clip))
    ax.hist(err_adv,   bins=80, alpha=0.6, color="#4C72B0",
            label="Adversarial attack", density=True, range=(0, clip))
    ax.axvline(threshold, color="black", lw=1.5, linestyle="--",
               label=f"Threshold = {threshold:.5f}")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title(f"Error Shift Under Evasion Attack — {model_name.capitalize()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,
                             f"error_shift_{model_name}.png"), dpi=150)
    plt.close()
    print(f"  Saved: error_shift_{model_name}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir: str, model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    X_val   = np.load(os.path.join(data_dir, "X_val.npy"))
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    input_dim = X_test.shape[1]

    # Use a subset of attack samples for speed (5,000 samples)
    attack_idx = np.where(y_test == 1)[0]
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(attack_idx, size=min(5000, len(attack_idx)),
                             replace=False)
    X_attack_np = X_test[sample_idx]
    X_attack    = torch.tensor(X_attack_np, dtype=torch.float32)

    print(f"  Attack samples for experiment: {len(X_attack):,}")

    # Epsilon values to test
    epsilons = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

    # ── Experiment 1: Evasion attack on all three models ─────────────────────
    print(f"\n{'='*60}")
    print("  EXPERIMENT 1: Evasion attack")
    print(f"{'='*60}")

    evasion_results = {}

    for model_name in ["small", "medium", "large"]:
        print(f"\n  Model: {model_name.upper()}")
        model     = load_model(model_name, input_dim, model_dir)
        threshold = get_threshold_from_val(model, X_val, percentile=99)
        print(f"  Threshold (99th percentile): {threshold:.6f}")

        # Baseline detection rate (epsilon=0)
        baseline = detection_rate(model, X_attack, threshold)
        print(f"  Baseline detection rate: {baseline:.4f}")

        rates = []
        for eps in epsilons:
            if eps == 0.0:
                rate = baseline
            else:
                X_adv = fgsm_evasion(model, X_attack, epsilon=eps, n_steps=10)
                rate  = detection_rate(model, X_adv, threshold)
            rates.append(rate)
            print(f"    epsilon={eps:.2f} → detection rate: {rate:.4f}")

        evasion_results[model_name] = rates

        # Plot error shift for medium model at epsilon=0.10
        if model_name == "medium":
            X_adv_vis = fgsm_evasion(model, X_attack, epsilon=0.10, n_steps=10)
            plot_error_shift(model, X_attack, X_adv_vis,
                             threshold, model_name, output_dir)

    plot_evasion_results(epsilons, evasion_results, output_dir)

    # ── Experiment 2: Denoising defence (medium model only) ──────────────────
    print(f"\n{'='*60}")
    print("  EXPERIMENT 2: Denoising autoencoder defence (medium model)")
    print(f"{'='*60}")

    denoising_path = os.path.join(model_dir, "medium_denoising.pt")
    denoising_model = train_denoising_autoencoder(
        input_dim  = input_dim,
        hidden_dims = MODEL_CONFIGS["medium"],
        X_train    = X_train,
        X_val      = X_val,
        noise_std  = 0.05,
        output_path = denoising_path,
    )

    # Get threshold for denoising model from clean val errors
    den_threshold = get_threshold_from_val(denoising_model, X_val, percentile=99)
    print(f"\n  Denoising threshold (99th percentile): {den_threshold:.6f}")

    # Reload vanilla medium for fair comparison
    vanilla_model     = load_model("medium", input_dim, model_dir)
    vanilla_threshold = get_threshold_from_val(vanilla_model, X_val, percentile=99)

    vanilla_rates   = evasion_results["medium"]
    denoising_rates = []

    print(f"\n  Defence comparison (medium model):")
    print(f"  {'Epsilon':>8} {'Vanilla':>10} {'Denoising':>12} {'Improvement':>13}")
    print(f"  {'-'*45}")

    for eps in epsilons:
        if eps == 0.0:
            den_rate = detection_rate(denoising_model, X_attack, den_threshold)
        else:
            X_adv    = fgsm_evasion(denoising_model, X_attack,
                                    epsilon=eps, n_steps=10)
            den_rate = detection_rate(denoising_model, X_adv, den_threshold)

        denoising_rates.append(den_rate)
        van_rate    = vanilla_rates[epsilons.index(eps)]
        improvement = den_rate - van_rate
        print(f"  {eps:>8.2f} {van_rate:>10.4f} {den_rate:>12.4f} "
              f"{improvement:>+13.4f}")

    plot_defence_comparison(epsilons, vanilla_rates,
                             denoising_rates, output_dir)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ADVERSARIAL ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Evasion attack results at epsilon=0.10:")
    print(f"  {'Model':<10} {'Baseline':>10} {'Adversarial':>13} {'Drop':>8}")
    print(f"  {'-'*43}")
    eps_010_idx = epsilons.index(0.10)
    for name, rates in evasion_results.items():
        baseline = rates[0]
        adv_rate = rates[eps_010_idx]
        drop     = baseline - adv_rate
        print(f"  {name:<10} {baseline:>10.4f} {adv_rate:>13.4f} {drop:>+8.4f}")

    print(f"\n  Denoising defence at epsilon=0.10:")
    van_010 = vanilla_rates[eps_010_idx]
    den_010 = denoising_rates[eps_010_idx]
    print(f"  Vanilla medium   : {van_010:.4f}")
    print(f"  Denoising medium : {den_010:.4f}")
    print(f"  Improvement      : {den_010 - van_010:+.4f}")

    print(f"\nAll plots saved to: {output_dir}/")
    print("Adversarial experiment complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adversarial robustness experiments for autoencoder IDS."
    )
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--model_dir",  default="models")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.output_dir)