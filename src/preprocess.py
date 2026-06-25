import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Column names 
CATEGORICAL_COLS = ["proto", "service", "state"]
LABEL_COL        = "label"       # 0 = normal, 1 = attack
ATTACK_CAT_COL   = "attack_cat"  # specific attack category
DROP_COLS        = [LABEL_COL, ATTACK_CAT_COL]

# Validation split fraction (taken from normal training samples only)
VAL_FRACTION = 0.10
RANDOM_SEED  = 42


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV, stripping whitespace from column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"  Loaded {path}  →  {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df


def separate_targets(df: pd.DataFrame):

    y  = df[LABEL_COL].astype(int)
    ac = df[ATTACK_CAT_COL].astype(str).str.strip() if ATTACK_CAT_COL in df.columns else None
    X  = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y, ac


def encode_categoricals(train_X: pd.DataFrame,
                        test_X:  pd.DataFrame):
  
    present = [c for c in CATEGORICAL_COLS if c in train_X.columns]
    if not present:
        print("  No categorical columns found – skipping OHE.")
        return train_X, test_X

    print(f"  One-hot encoding: {present}")
    train_encoded = pd.get_dummies(train_X, columns=present, drop_first=False)
    test_encoded  = pd.get_dummies(test_X,  columns=present, drop_first=False)

    # Align columns – test may have unseen categories (fill 0) or missing ones
    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )
    print(f"  Feature dims after OHE → train: {train_encoded.shape[1]}, "
          f"test: {test_encoded.shape[1]}")
    return train_encoded, test_encoded


def fit_scaler_on_normal(train_X: pd.DataFrame,
                         train_y: pd.Series) -> MinMaxScaler:
    """Fit MinMaxScaler using ONLY normal (label=0) training samples."""
    normal_mask = train_y == 0
    print(f"  Fitting scaler on {normal_mask.sum():,} normal training samples "
          f"(out of {len(train_y):,} total).")
    scaler = MinMaxScaler()
    scaler.fit(train_X[normal_mask])
    return scaler


def make_validation_split(train_X: np.ndarray,
                          train_y: np.ndarray,
                          train_ac: np.ndarray | None):
    """
    Create a validation set from the normal training samples only.
    The autoencoder is trained on normal data, so validation is also normal-only.
    """
    normal_idx = np.where(train_y == 0)[0]
    tr_idx, val_idx = train_test_split(
        normal_idx, test_size=VAL_FRACTION, random_state=RANDOM_SEED
    )

    X_train_normal  = train_X[tr_idx]
    y_train_normal  = train_y[tr_idx]

    X_val_normal    = train_X[val_idx]
    y_val_normal    = train_y[val_idx]

    ac_train = train_ac[tr_idx]  if train_ac is not None else None
    ac_val   = train_ac[val_idx] if train_ac is not None else None

    print(f"  Train (normal only): {X_train_normal.shape[0]:,} samples")
    print(f"  Val   (normal only): {X_val_normal.shape[0]:,} samples")
    return X_train_normal, y_train_normal, ac_train, \
           X_val_normal,   y_val_normal,   ac_val


def save_outputs(output_dir: str, **arrays):
    """Save numpy arrays to output_dir/{name}.npy"""
    os.makedirs(output_dir, exist_ok=True)
    for name, arr in arrays.items():
        if arr is None:
            continue
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  Saved {path}  shape={arr.shape}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(train_path: str, test_path: str, output_dir: str):
    print("\n[1/6] Loading data...")
    train_df = load_data(train_path)
    test_df  = load_data(test_path)

    print("\n[2/6] Separating features and targets...")
    train_X, train_y, train_ac = separate_targets(train_df)
    test_X,  test_y,  test_ac  = separate_targets(test_df)

    print(f"  Train labels  → normal: {(train_y==0).sum():,}  "
          f"attack: {(train_y==1).sum():,}")
    print(f"  Test  labels  → normal: {(test_y==0).sum():,}  "
          f"attack: {(test_y==1).sum():,}")

    print("\n[3/6] One-hot encoding categorical features...")
    train_X, test_X = encode_categoricals(train_X, test_X)

    print("\n[4/6] Fitting MinMax scaler on normal training samples...")
    scaler = fit_scaler_on_normal(train_X, train_y)

    print("\n[5/6] Scaling all splits...")
    train_X_scaled = scaler.transform(train_X).astype(np.float32)
    test_X_scaled  = scaler.transform(test_X).astype(np.float32)
    train_y_arr    = train_y.values.astype(np.int32)
    test_y_arr     = test_y.values.astype(np.int32)
    train_ac_arr   = train_ac.values if train_ac is not None else None
    test_ac_arr    = test_ac.values  if test_ac  is not None else None

    print("\n[6/6] Creating train/val split (normal only) and saving...")
    (X_train, y_train, ac_train,
     X_val,   y_val,   ac_val) = make_validation_split(
        train_X_scaled, train_y_arr, train_ac_arr
    )

    save_outputs(
        output_dir,
        # Training set (normal only – used to train the autoencoder)
        X_train=X_train,
        y_train=y_train,
        ac_train=ac_train,
        # Validation set (normal only – used for early stopping & threshold tuning)
        X_val=X_val,
        y_val=y_val,
        ac_val=ac_val,
        # Full test set (normal + attack – used for final evaluation only)
        X_test=test_X_scaled,
        y_test=test_y_arr,
        ac_test=test_ac_arr,
    )

    # Save feature dimension so models can read it without reloading data
    feature_dim = np.array([X_train.shape[1]], dtype=np.int32)
    np.save(os.path.join(output_dir, "feature_dim.npy"), feature_dim)
    print(f"\n  Feature dimension: {X_train.shape[1]}")
    print("\nPreprocessing complete.")


#  CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess UNSW-NB15 for autoencoder-based anomaly detection."
    )
    parser.add_argument(
        "--train_path", required=True,
        help="Path to UNSW_NB15_training-set.csv"
    )
    parser.add_argument(
        "--test_path", required=True,
        help="Path to UNSW_NB15_testing-set.csv"
    )
    parser.add_argument(
        "--output_dir", default="data/processed",
        help="Directory to save processed .npy files (default: data/processed)"
    )
    args = parser.parse_args()
    preprocess(args.train_path, args.test_path, args.output_dir)