import argparse
import csv
import json
import sqlite3
import joblib
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from src.config import ARTIFACTS_DIR, DB_PATH

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def ensure_column(conn: sqlite3.Connection):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if "intensity_score" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN intensity_score REAL;")
        conn.commit()


def load_aligned_data():
    embeddings = np.load(ARTIFACTS_DIR / "embeddings.npy")
    with open(ARTIFACTS_DIR / "appid_map.json", "r", encoding="utf-8") as f:
        appids = json.load(f)

    idx_by_appid = {int(a): i for i, a in enumerate(appids)}

    silver_path = ARTIFACTS_DIR / "silver_labels.csv"
    X_list, y_list, matched_appids = [], [], []

    with open(silver_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            appid = int(row["appid"])
            if appid in idx_by_appid:
                X_list.append(embeddings[idx_by_appid[appid]])
                y_list.append(float(row["intensity_score"]))
                matched_appids.append(appid)

    return np.vstack(X_list), np.array(y_list, dtype=np.float32), matched_appids


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    r2 = r2_score(y_true, y_pred_clipped)
    mae = mean_absolute_error(y_true, y_pred_clipped)
    rho = float(spearmanr(y_true, y_pred_clipped).statistic)
    print(f"[{name:<10}] R2={r2:.4f} | Spearman={rho:.4f} | MAE={mae:.4f}")
    return {"name": name, "r2": r2, "rho": rho, "mae": mae}


def run_train(mode: str = "rank"):
    X, y_raw, matched_appids = load_aligned_data()
    print(f"Aligned {len(y_raw)} games (X={X.shape}).")

    # 1. Inspect raw NLI distribution
    p10, p25, p50, p75, p90 = np.percentile(y_raw, [10, 25, 50, 75, 90])
    print(
        f"Raw NLI distribution: "
        f"p10={p10:.3f} | p25={p25:.3f} | p50={p50:.3f} | p75={p75:.3f} | p90={p90:.3f}"
    )

    # Choose target scale for training
    if mode == "rank":
        y_target = ((rankdata(y_raw, method="average") - 1.0) / (len(y_raw) - 1.0)).astype(np.float32)
    else:
        y_target = y_raw

    # 2. Benchmark on 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_target, test_size=0.2, random_state=42
    )

    print(f"\nBenchmarking on 80/20 split (mode={mode!r}):")
    candidates = []

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 25))
    ridge.fit(X_train, y_train)
    res_ridge = evaluate("RidgeCV", y_val, ridge.predict(X_val))
    candidates.append((res_ridge["rho"], "RidgeCV", lambda: RidgeCV(alphas=np.logspace(-2, 3, 25))))

    if HAS_LGBM:
        lgbm = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            colsample_bytree=0.7,
            random_state=42,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train)
        res_lgbm = evaluate("LightGBM", y_val, lgbm.predict(X_val))
        candidates.append((
            res_lgbm["rho"],
            "LightGBM",
            lambda: LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=15,
                colsample_bytree=0.7,
                random_state=42,
                verbose=-1,
            ),
        ))
    else:
        print("[LightGBM  ] Skipped (not installed).")

    # 3. Refit winner on 100% of data and save for future unseen games
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_rho, best_name, make_model = candidates[0]
    print(f"\nWinner: {best_name} (Spearman={best_rho:.4f}). Refitting on 100% of data...")

    final_model = make_model()
    final_model.fit(X, y_target)

    model_path = ARTIFACTS_DIR / "intensity_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved model to {model_path}")

    # 4. Determine scores to write into SQLite
    if mode == "smoothed":
        db_scores = np.clip(final_model.predict(X), 0.0, 1.0)
    else:
        db_scores = y_target

    sp10, sp50, sp90 = np.percentile(db_scores, [10, 50, 90])
    print(f"Saved DB score distribution ({mode}): p10={sp10:.3f} | p50={sp50:.3f} | p90={sp90:.3f}")

    conn = sqlite3.connect(DB_PATH)
    ensure_column(conn)
    updates = [
        (round(float(score), 4), appid)
        for appid, score in zip(matched_appids, db_scores)
    ]
    conn.executemany("UPDATE games SET intensity_score = ? WHERE appid = ?", updates)
    conn.commit()
    conn.close()
    print(f"Updated games.intensity_score for {len(updates)} rows in SQLite.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["rank", "raw", "smoothed"],
        default="rank",
        help="rank=percentile rank (Option A), smoothed=model predictions (Option B), raw=unmodified NLI",
    )
    args = parser.parse_args()
    run_train(mode=args.mode)
