import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve,
                             f1_score, confusion_matrix)

# ── Re-use model definition ───────────────────────────────────────────────────

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


# ── Core helpers ──────────────────────────────────────────────────────────────

def load_model(model_name: str, input_dim: int, model_dir: str) -> Autoencoder:
    model = Autoencoder(input_dim, MODEL_CONFIGS[model_name])
    path  = os.path.join(model_dir, f"{model_name}_best.pt")
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def get_reconstruction_errors(model: Autoencoder,
                               X: np.ndarray) -> np.ndarray:
    """Return per-sample MSE reconstruction error."""
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32)
        X_hat = model(X_t)
        errors = ((X_t - X_hat) ** 2).mean(dim=1).numpy()
    return errors


# ── Threshold selection methods ───────────────────────────────────────────────

def get_thresholds(val_errors: np.ndarray) -> dict:
    """
    Returns a dict of {threshold_name: value} using three methods:
      1. Percentile of validation reconstruction errors
      2. Mean + k * std of validation reconstruction errors
      3. Best-F1 is computed later (needs test labels) — placeholder here
    """
    thresholds = {
        "percentile_95":  float(np.percentile(val_errors, 95)),
        "percentile_99":  float(np.percentile(val_errors, 99)),
        "mean_2std":      float(val_errors.mean() + 2 * val_errors.std()),
        "mean_3std":      float(val_errors.mean() + 3 * val_errors.std()),
    }
    return thresholds


def best_f1_threshold(errors: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold that maximises F1 score on the test set."""
    precisions, recalls, thresholds = precision_recall_curve(labels, errors)
    # avoid division by zero
    f1s = np.where((precisions + recalls) > 0,
                   2 * precisions * recalls / (precisions + recalls + 1e-9),
                   0)
    best_idx = np.argmax(f1s[:-1])   # last element has no threshold
    return float(thresholds[best_idx])


# ── Per-threshold metrics ─────────────────────────────────────────────────────

def threshold_metrics(errors: np.ndarray,
                      labels: np.ndarray,
                      threshold: float) -> dict:
    preds = (errors >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1  = f1_score(labels, preds, zero_division=0)
    return {"TPR": tpr, "FPR": fpr, "F1": f1,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


# ── Per-attack-category breakdown ────────────────────────────────────────────

def per_category_detection(errors: np.ndarray,
                            labels: np.ndarray,
                            attack_cats: np.ndarray,
                            threshold: float) -> dict:
    """
    For each attack category, compute detection rate (TPR).
    Normal traffic is excluded (label == 0).
    """
    results = {}
    attack_mask = labels == 1
    cats = np.unique(attack_cats[attack_mask])

    for cat in cats:
        cat_mask = attack_mask & (attack_cats == cat)
        cat_errors = errors[cat_mask]
        detected = (cat_errors >= threshold).sum()
        total    = len(cat_errors)
        results[cat] = {
            "detected": int(detected),
            "total":    int(total),
            "rate":     float(detected / total) if total > 0 else 0.0,
        }
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {"small": "#4C72B0", "medium": "#DD8452", "large": "#55A868"}


def plot_roc_curves(all_results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["errors"])
        auc = res["roc_auc"]
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.4f})",
                color=COLORS[name], lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curves.png"), dpi=150)
    plt.close()
    print("  Saved: roc_curves.png")


def plot_pr_curves(all_results: dict, output_dir: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in all_results.items():
        prec, rec, _ = precision_recall_curve(res["labels"], res["errors"])
        ap = res["pr_auc"]
        ax.plot(rec, prec, label=f"{name}  (AP={ap:.4f})",
                color=COLORS[name], lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=150)
    plt.close()
    print("  Saved: pr_curves.png")


def plot_efficiency_tradeoff(all_results: dict, output_dir: str):
    names  = list(all_results.keys())
    params = [all_results[n]["n_params"] for n in names]
    aucs   = [all_results[n]["roc_auc"]  for n in names]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(names):
        ax.scatter(params[i], aucs[i], color=colors[i], s=120, zorder=3)
        ax.annotate(name, (params[i], aucs[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=10)
    ax.plot(params, aucs, "k--", lw=1, alpha=0.4)
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Efficiency–Performance Tradeoff")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_tradeoff.png"), dpi=150)
    plt.close()
    print("  Saved: efficiency_tradeoff.png")


def plot_attack_category_heatmap(all_results: dict, output_dir: str):
    """Heatmap: rows = attack categories, columns = models, values = detection rate."""
    # Collect all categories across all models (use best_f1 threshold)
    all_cats = set()
    for res in all_results.values():
        all_cats.update(res["per_category"].keys())
    all_cats = sorted(all_cats)

    model_names = list(all_results.keys())
    matrix = np.zeros((len(all_cats), len(model_names)))

    for j, mname in enumerate(model_names):
        cat_res = all_results[mname]["per_category"]
        for i, cat in enumerate(all_cats):
            matrix[i, j] = cat_res.get(cat, {}).get("rate", 0.0)

    fig, ax = plt.subplots(figsize=(7, max(4, len(all_cats) * 0.6 + 1.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Detection Rate")

    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([m.capitalize() for m in model_names])
    ax.set_yticks(range(len(all_cats)))
    ax.set_yticklabels(all_cats)
    ax.set_title("Detection Rate by Attack Category (best-F1 threshold)")

    # Annotate cells
    for i in range(len(all_cats)):
        for j in range(len(model_names)):
            val = matrix[i, j]
            color = "black" if 0.3 < val < 0.8 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attack_category_heatmap.png"), dpi=150)
    plt.close()
    print("  Saved: attack_category_heatmap.png")


def plot_reconstruction_error_dist(all_results: dict,
                                   labels: np.ndarray,
                                   output_dir: str):
    """Distribution of reconstruction errors for normal vs attack per model."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    normal_mask = labels == 0
    attack_mask = labels == 1

    for ax, (name, res) in zip(axes, all_results.items()):
        errors = res["errors"]
        # clip for readability
        clip = np.percentile(errors, 99.5)
        ax.hist(errors[normal_mask], bins=80, alpha=0.6,
                color=COLORS[name], label="Normal",
                density=True, range=(0, clip))
        ax.hist(errors[attack_mask], bins=80, alpha=0.6,
                color="red", label="Attack",
                density=True, range=(0, clip))
        thr = res["thresholds"]["best_f1"]
        ax.axvline(thr, color="black", lw=1.5, linestyle="--",
                   label=f"Threshold={thr:.5f}")
        ax.set_title(f"{name.capitalize()} model")
        ax.set_xlabel("Reconstruction error (MSE)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Reconstruction Error Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: error_distributions.png")


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main(data_dir: str, model_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    X_val      = np.load(os.path.join(data_dir, "X_val.npy"))
    X_test     = np.load(os.path.join(data_dir, "X_test.npy"))
    y_test     = np.load(os.path.join(data_dir, "y_test.npy"))
    ac_test    = np.load(os.path.join(data_dir, "ac_test.npy"),
                         allow_pickle=True).astype(str)
    input_dim  = X_test.shape[1]

    print(f"  Test set  : {X_test.shape[0]:,} samples  "
          f"(normal: {(y_test==0).sum():,}  attack: {(y_test==1).sum():,})")

    all_results = {}

    for model_name in ["small", "medium", "large"]:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name.upper()}")
        print(f"{'='*60}")

        model      = load_model(model_name, input_dim, model_dir)
        val_errors = get_reconstruction_errors(model, X_val)
        errors     = get_reconstruction_errors(model, X_test)

        # ── Threshold-free metrics ────────────────────────────────────────────
        roc_auc = roc_auc_score(y_test, errors)
        pr_auc  = average_precision_score(y_test, errors)
        print(f"  ROC-AUC : {roc_auc:.4f}")
        print(f"  PR-AUC  : {pr_auc:.4f}")

        # ── Threshold selection ───────────────────────────────────────────────
        thresholds = get_thresholds(val_errors)
        thresholds["best_f1"] = best_f1_threshold(errors, y_test)

        print(f"\n  Threshold comparison:")
        print(f"  {'Method':<18} {'Threshold':>12} {'F1':>8} "
              f"{'TPR':>8} {'FPR':>8}")
        print(f"  {'-'*58}")

        thr_metrics = {}
        for thr_name, thr_val in thresholds.items():
            m = threshold_metrics(errors, y_test, thr_val)
            thr_metrics[thr_name] = {**m, "threshold": thr_val}
            print(f"  {thr_name:<18} {thr_val:>12.6f} {m['F1']:>8.4f} "
                  f"{m['TPR']:>8.4f} {m['FPR']:>8.4f}")

        # ── Per-attack-category breakdown (using best_f1 threshold) ──────────
        best_thr = thresholds["best_f1"]
        cat_results = per_category_detection(errors, y_test, ac_test, best_thr)

        print(f"\n  Per-category detection (best-F1 threshold = {best_thr:.6f}):")
        print(f"  {'Category':<20} {'Detected':>10} {'Total':>8} {'Rate':>8}")
        print(f"  {'-'*48}")
        for cat, cr in sorted(cat_results.items(),
                               key=lambda x: -x[1]["rate"]):
            print(f"  {cat:<20} {cr['detected']:>10,} "
                  f"{cr['total']:>8,} {cr['rate']:>8.4f}")

        # Store for plotting
        n_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
        all_results[model_name] = {
            "errors":       errors,
            "labels":       y_test,
            "roc_auc":      roc_auc,
            "pr_auc":       pr_auc,
            "thresholds":   thresholds,
            "thr_metrics":  thr_metrics,
            "per_category": cat_results,
            "n_params":     n_params,
        }

    # ── Generate all plots ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating plots...")
    print(f"{'='*60}")

    plot_roc_curves(all_results, output_dir)
    plot_pr_curves(all_results, output_dir)
    plot_efficiency_tradeoff(all_results, output_dir)
    plot_attack_category_heatmap(all_results, output_dir)
    plot_reconstruction_error_dist(all_results, y_test, output_dir)

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<10} {'ROC-AUC':>9} {'PR-AUC':>9} "
          f"{'F1 (best)':>11} {'TPR':>8} {'FPR':>8}")
    print(f"  {'-'*60}")
    for name, res in all_results.items():
        m = res["thr_metrics"]["best_f1"]
        print(f"  {name:<10} {res['roc_auc']:>9.4f} {res['pr_auc']:>9.4f} "
              f"{m['F1']:>11.4f} {m['TPR']:>8.4f} {m['FPR']:>8.4f}")

    print(f"\nAll plots saved to: {output_dir}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained autoencoders on UNSW-NB15 test set."
    )
    parser.add_argument("--data_dir",   default="data/processed")
    parser.add_argument("--model_dir",  default="models")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.output_dir)