import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import ARTIFACTS_DIR, DB_PATH

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VibeRecommender:
    def __init__(self):
        self._model: SentenceTransformer | None = None

        # 1. Load precomputed L2-normalized embeddings and aligned appid order
        self.embeddings = np.load(ARTIFACTS_DIR / "embeddings.npy").astype(np.float32)
        with open(ARTIFACTS_DIR / "appid_map.json", "r", encoding="utf-8") as f:
            self.appids: list[int] = [int(a) for a in json.load(f)]

        self.idx_by_appid: dict[int, int] = {a: i for i, a in enumerate(self.appids)}

        # 2. Load game metadata from SQLite aligned to self.appids
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                appid,
                name,
                short_description,
                genres_json,
                tags_json,
                median_review_playtime,
                commitment_score,
                intensity_score
            FROM games
            WHERE is_eligible = 1 AND is_software = 0
            """
        ).fetchall()
        conn.close()

        raw_by_appid = {int(r["appid"]): dict(r) for r in rows}

        n = len(self.appids)
        self.games: list[dict] = []
        self.commitment = np.full(n, np.nan, dtype=np.float32)
        self.intensity = np.full(n, np.nan, dtype=np.float32)
        self.is_singleplayer = np.zeros(n, dtype=bool)
        self.valid_mask = np.zeros(n, dtype=bool)

        for i, appid in enumerate(self.appids):
            g = raw_by_appid.get(appid)
            if not g:
                self.games.append({"appid": appid, "name": f"Unknown ({appid})"})
                continue

            tags_dict = json.loads(g["tags_json"]) if g["tags_json"] else {}
            genres = json.loads(g["genres_json"]) if g["genres_json"] else []
            top_tags = [
                t for t, _ in sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)[:8]
            ]

            g["genres"] = genres
            g["top_tags"] = top_tags
            g["store_url"] = f"https://store.steampowered.com/app/{appid}/"
            self.games.append(g)

            if g["commitment_score"] is not None and g["intensity_score"] is not None:
                self.commitment[i] = float(g["commitment_score"])
                self.intensity[i] = float(g["intensity_score"])
                self.valid_mask[i] = True

            # Singleplayer check from Steam tags (avoids forever-multiplayer grind when toggled)
            self.is_singleplayer[i] = "Singleplayer" in tags_dict

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def all_titles(self) -> list[tuple[int, str]]:
        """Sorted (appid, name) pairs for UI dropdowns."""
        pairs = [
            (self.appids[i], self.games[i]["name"])
            for i in range(len(self.appids))
            if self.valid_mask[i]
        ]
        pairs.sort(key=lambda x: x[1].lower())
        return pairs

    def recommend(
        self,
        query: str = "",
        seed_appid: int | None = None,
        commitment_range: tuple[float, float] = (0.0, 1.0),
        intensity_range: tuple[float, float] = (0.0, 1.0),
        singleplayer_only: bool = False,
        top_k: int = 15,
    ) -> list[dict]:
        # 1. Hard filter on vibe ranges + optional singleplayer toggle
        c_min, c_max = commitment_range
        i_min, i_max = intensity_range

        mask = (
            self.valid_mask
            & (self.commitment >= c_min)
            & (self.commitment <= c_max)
            & (self.intensity >= i_min)
            & (self.intensity <= i_max)
        )
        if singleplayer_only:
            mask &= self.is_singleplayer

        seed_idx = self.idx_by_appid.get(int(seed_appid)) if seed_appid is not None else None
        if seed_idx is not None:
            mask[seed_idx] = False  # Don't recommend the seed game itself

        candidate_indices = np.flatnonzero(mask)
        if len(candidate_indices) == 0:
            return []

        # 2. Build target vector (query, seed game, or both)
        vecs = []
        if query and query.strip():
            q_vec = self.model.encode(
                query.strip(),
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            vecs.append(q_vec)

        if seed_idx is not None:
            vecs.append(self.embeddings[seed_idx])

        # 3. Score candidates
        if vecs:
            target = np.mean(vecs, axis=0)
            norm = np.linalg.norm(target)
            if norm > 0:
                target = target / norm
            # Dot product == cosine similarity for L2-normalized vectors
            scores = self.embeddings[candidate_indices] @ target
        else:
            # Pure vibe-browsing mode: rank by proximity to the center of the selected vibe box
            c_mid = 0.5 * (c_min + c_max)
            i_mid = 0.5 * (i_min + i_max)
            dist = np.hypot(
                self.commitment[candidate_indices] - c_mid,
                self.intensity[candidate_indices] - i_mid,
            )
            scores = 1.0 - dist

        # 4. Sort and format top_k results
        k = min(top_k, len(candidate_indices))
        top_local = np.argpartition(scores, -k)[-k:]
        top_local = top_local[np.argsort(scores[top_local])[::-1]]

        results = []
        for loc in top_local:
            idx = int(candidate_indices[loc])
            g = self.games[idx]
            results.append(
                {
                    "appid": g["appid"],
                    "name": g["name"],
                    "score": round(float(scores[loc]), 4),
                    "commitment_score": round(float(self.commitment[idx]), 3),
                    "intensity_score": round(float(self.intensity[idx]), 3),
                    "median_hours": round((g["median_review_playtime"] or 0) / 60.0, 1),
                    "top_tags": g["top_tags"],
                    "short_description": g["short_description"] or "",
                    "store_url": g["store_url"],
                }
            )
        return results


if __name__ == "__main__":
    rec = VibeRecommender()

    print("--- Test 1: Chill, low-commitment singleplayer puzzle/cozy ---")
    for r in rec.recommend(
        query="cozy relaxing puzzle exploration",
        commitment_range=(0.0, 0.45),
        intensity_range=(0.0, 0.35),
        singleplayer_only=True,
        top_k=5,
    ):
        print(
            f"  sim={r['score']:.3f} | com={r['commitment_score']:.2f} | "
            f"int={r['intensity_score']:.2f} | {r['median_hours']:>5.1f}h | {r['name']}"
        )

    print("\n--- Test 2: Intense, high-commitment singleplayer action/RPG ---")
    for r in rec.recommend(
        query="brutal tactical combat dark fantasy",
        commitment_range=(0.55, 1.0),
        intensity_range=(0.65, 1.0),
        singleplayer_only=True,
        top_k=5,
    ):
        print(
            f"  sim={r['score']:.3f} | com={r['commitment_score']:.2f} | "
            f"int={r['intensity_score']:.2f} | {r['median_hours']:>5.1f}h | {r['name']}"
        )
