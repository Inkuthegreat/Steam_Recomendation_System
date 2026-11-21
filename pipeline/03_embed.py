import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import ARTIFACTS_DIR, DB_PATH

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_game_documents(conn: sqlite3.Connection) -> tuple[list[int], list[str]]:
    conn.row_factory = sqlite3.Row
    games = conn.execute(
        """
        SELECT appid, name, short_description, genres_json, tags_json
        FROM games
        WHERE is_eligible = 1 AND is_software = 0
        ORDER BY appid ASC
        """
    ).fetchall()

    # Pre-load top 8 helpful reviews per game
    reviews_by_app: dict[int, list[str]] = {}
    for row in conn.execute(
        """
        SELECT appid, review_text
        FROM reviews
        ORDER BY appid ASC, votes_helpful DESC
        """
    ):
        bucket = reviews_by_app.setdefault(row["appid"], [])
        if len(bucket) < 8:
            bucket.append(row["review_text"])

    appids: list[int] = []
    docs: list[str] = []

    for g in games:
        genres = json.loads(g["genres_json"]) if g["genres_json"] else []
        tags_dict = json.loads(g["tags_json"]) if g["tags_json"] else {}
        top_tags = [
            t for t, _ in sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        desc = (g["short_description"] or "").strip()
        revs = " ".join(reviews_by_app.get(g["appid"], []))
        # Cap raw review text so one giant review doesn't drown out the rest
        revs = revs[:1500]

        parts = [
            f"Title: {g['name']}.",
            f"Genres: {', '.join(genres)}." if genres else "",
            f"Tags: {', '.join(top_tags)}." if top_tags else "",
            f"Description: {desc}" if desc else "",
            f"Player reviews: {revs}" if revs else "",
        ]
        doc = " ".join(p for p in parts if p)

        appids.append(int(g["appid"]))
        docs.append(doc)

    return appids, docs


def run_embed():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    appids, docs = build_game_documents(conn)
    conn.close()

    print(f"Loaded {len(docs)} game documents. Encoding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        docs,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalized -> dot product == cosine similarity
        convert_to_numpy=True,
    ).astype(np.float32)

    emb_path = ARTIFACTS_DIR / "embeddings.npy"
    map_path = ARTIFACTS_DIR / "appid_map.json"

    np.save(emb_path, embeddings)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(appids, f)

    print(f"Saved {embeddings.shape} float32 matrix to {emb_path}")
    print(f"Saved {len(appids)} appids to {map_path}")


if __name__ == "__main__":
    run_embed()
