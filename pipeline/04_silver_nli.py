import argparse
import csv
import json
import sqlite3
import torch
from transformers import pipeline
from src.config import ARTIFACTS_DIR, DB_PATH

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS = [
    "relaxing, cozy, and low-stress gameplay",
    "intense, stressful, and high-pressure gameplay",
]


def load_game_texts(limit: int | None = None) -> list[tuple[int, str, str]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Random stratified-ish order so --limit 500 isn't biased toward old appids
    query = """
        SELECT appid, name, short_description
        FROM games
        WHERE is_eligible = 1 AND is_software = 0
        ORDER BY appid ASC
    """
    games = conn.execute(query).fetchall()

    reviews_by_app: dict[int, list[str]] = {}
    for row in conn.execute(
        """
        SELECT appid, review_text
        FROM reviews
        ORDER BY appid ASC, votes_helpful DESC
        """
    ):
        bucket = reviews_by_app.setdefault(row["appid"], [])
        if len(bucket) < 5:
            bucket.append(row["review_text"])
    conn.close()

    # Deterministic shuffle via stride if limiting, so genres stay mixed
    if limit and limit < len(games):
        step = len(games) / limit
        games = [games[int(i * step)] for i in range(limit)]

    items = []
    for g in games:
        revs = " ".join(reviews_by_app.get(g["appid"], []))[:900]
        desc = (g["short_description"] or "").strip()[:300]
        # Reviews first: players describe vibe/stress much better than store copy
        text = f"{revs} {desc}".strip()
        if not text:
            text = g["name"]
        items.append((int(g["appid"]), g["name"], text))
    return items


def run_nli(limit: int | None = None, batch_size: int = 8):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    items = load_game_texts(limit=limit)
    print(f"Loaded {len(items)} games for NLI intensity scoring.")

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Running {MODEL_NAME} on {'CUDA (fp16)' if use_cuda else 'CPU (fp32)'}...")

    clf = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
        torch_dtype=dtype,
    )

    texts = [x[2] for x in items]
    results = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        outputs = clf(
            batch_texts,
            candidate_labels=LABELS,
            hypothesis_template="This game features {}.",
            multi_label=True,
            truncation=True,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for (appid, name, _), out in zip(items[start : start + batch_size], outputs):
            score_map = dict(zip(out["labels"], out["scores"]))
            p_chill = float(score_map[LABELS[0]])
            p_intense = float(score_map[LABELS[1]])
            intensity = p_intense / (p_intense + p_chill + 1e-8)
            results.append((appid, name, round(p_chill, 4), round(p_intense, 4), round(intensity, 4)))

        done = min(start + batch_size, len(texts))
        if done % 80 == 0 or done == len(texts):
            print(f"[NLI] {done}/{len(texts)} done.")

    out_path = ARTIFACTS_DIR / "silver_labels.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["appid", "name", "p_chill", "p_intense", "intensity_score"])
        writer.writerows(results)

    print(f"\nSaved {len(results)} silver labels to {out_path}")

    ranked = sorted(results, key=lambda x: x[4])
    print("\nMost Chill (lowest intensity):")
    for r in ranked[:10]:
        print(f"  {r[4]:.4f} (chill={r[2]:.2f}, int={r[3]:.2f}) | {r[1]}")

    print("\nMost Intense (highest intensity):")
    for r in ranked[-10:][::-1]:
        print(f"  {r[4]:.4f} (chill={r[2]:.2f}, int={r[3]:.2f}) | {r[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of games to score (default: all)")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    run_nli(limit=args.limit, batch_size=args.batch_size)
