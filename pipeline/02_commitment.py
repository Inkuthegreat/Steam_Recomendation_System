import json
import math
import sqlite3
import statistics
from src.config import DB_PATH

# Exact tag strings verified against your steam_recs.db
# Noise tags (e.g., Jump Scare, Bikes, Hentai, Dwarves, Horses) are intentionally excluded.
SHORT_TAGS = {
    "Short": 1.0,
    "Runner": 0.9,
    "Escape Room": 0.85,
    "Hidden Object": 0.8,
    "Walking Simulator": 0.8,
    "Shoot 'Em Up": 0.8,
    "Score Attack": 0.8,
    "Arcade": 0.75,
    "Bullet Hell": 0.75,
    "Interactive Fiction": 0.75,
    "Point & Click": 0.7,
    "Puzzle-Platformer": 0.7,
    "Minigames": 0.8,
    "Casual": 0.65,
    "Linear": 0.6,
    "Rhythm": 0.6,
    "Party Game": 0.6,
    "Puzzle": 0.55,
    "Rogue-like": 0.45,
    "Rogue-lite": 0.45,
    "Action Roguelike": 0.45,
}

LONG_TAGS = {
    "Grand Strategy": 1.0,
    "4X": 1.0,
    "MMORPG": 1.0,
    "Space Sim": 0.95,
    "Extraction Shooter": 0.9,
    "MOBA": 0.9,
    "CRPG": 0.9,
    "JRPG": 0.9,
    "Strategy RPG": 0.85,
    "Party-Based RPG": 0.85,
    "Tactical RPG": 0.85,
    "Real-Time with Pause": 0.85,
    "Open World Survival Craft": 0.85,
    "Automation": 0.8,
    "Colony Sim": 0.8,
    "Turn-Based Strategy": 0.75,
    "City Builder": 0.75,
    "Open World": 0.7,
    "RPG": 0.65,
    "Sandbox": 0.65,
    "Story Rich": 0.55,
}

NON_GAME_TAGS = {
    "Utilities",
    "Software",
    "Design & Illustration",
    "Animation & Modeling",
    "Video Production",
    "Audio Production",
    "Software Training",
    "Benchmark",
}

# Calibrated to reviewer playtime distribution: 60 min (1 hr) to 7200 min (120 hrs)
LOG_MIN = math.log1p(60)
LOG_MAX = math.log1p(7200)


def ensure_columns(conn: sqlite3.Connection):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(games)").fetchall()}
    if "commitment_score" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN commitment_score REAL;")
    if "is_software" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN is_software INTEGER DEFAULT 0;")
    conn.commit()


def parse_tags(tags_raw: str | None) -> dict[str, float]:
    if not tags_raw:
        return {}
    try:
        parsed = json.loads(tags_raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def is_non_game(tags: dict[str, float]) -> bool:
    if not tags:
        return False
    # Flag if any non-game tag is among the top 3 most-voted tags
    top_tags = sorted(tags.items(), key=lambda x: float(x[1] or 0), reverse=True)[:3]
    return any(t in NON_GAME_TAGS for t, _ in top_tags)


def compute_tag_score(tags: dict[str, float]) -> tuple[float, float]:
    """Returns (tag_commitment_0_to_1, confidence_0_to_1)."""
    if not tags:
        return 0.5, 0.0

    short_mass = 0.0
    long_mass = 0.0
    total_relevant_votes = 0.0

    for tag_name, votes in tags.items():
        v = float(votes or 0)
        if v <= 0:
            continue
        if tag_name in SHORT_TAGS:
            short_mass += v * SHORT_TAGS[tag_name]
            total_relevant_votes += v
        if tag_name in LONG_TAGS:
            long_mass += v * LONG_TAGS[tag_name]
            total_relevant_votes += v

    if total_relevant_votes <= 0:
        return 0.5, 0.0

    net = (long_mass - short_mass) / total_relevant_votes  # [-1, 1]
    score = max(0.0, min(1.0, 0.5 + 0.5 * net))
    confidence = min(1.0, total_relevant_votes / 500.0)
    return score, confidence


def normalize_playtime(minutes: float | None, fallback_log: float) -> tuple[float, bool]:
    if minutes is not None and minutes > 0:
        val = (math.log1p(minutes) - LOG_MIN) / (LOG_MAX - LOG_MIN)
        return max(0.0, min(1.0, val)), True
    val = (fallback_log - LOG_MIN) / (LOG_MAX - LOG_MIN)
    return max(0.0, min(1.0, val)), False


def run_commitment():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_columns(conn)

    rows = conn.execute(
        """
        SELECT appid, name, median_review_playtime, tags_json
        FROM games
        WHERE is_eligible = 1
        """
    ).fetchall()

    valid_logs = [
        math.log1p(r["median_review_playtime"])
        for r in rows
        if r["median_review_playtime"] and r["median_review_playtime"] > 0
    ]
    fallback_log = statistics.median(valid_logs) if valid_logs else math.log1p(600)

    updates = []
    missing_pt = 0
    software_count = 0

    for r in rows:
        tags = parse_tags(r["tags_json"])
        soft_flag = 1 if is_non_game(tags) else 0
        if soft_flag:
            software_count += 1

        pt_norm, has_pt = normalize_playtime(r["median_review_playtime"], fallback_log)
        if not has_pt:
            missing_pt += 1

        tag_score, tag_conf = compute_tag_score(tags)

        if has_pt:
            tag_weight = 0.20 * tag_conf
            score = (1.0 - tag_weight) * pt_norm + tag_weight * tag_score
        else:
            tag_weight = 0.75 * tag_conf
            score = (1.0 - tag_weight) * pt_norm + tag_weight * tag_score

        score = round(max(0.0, min(1.0, score)), 4)
        updates.append((score, soft_flag, r["appid"]))

    conn.executemany(
        "UPDATE games SET commitment_score = ?, is_software = ? WHERE appid = ?",
        updates,
    )
    conn.commit()

    scores = sorted(u[0] for u in updates if u[1] == 0)
    print(f"Updated {len(updates)} eligible rows.")
    print(f"Flagged as non-game software (is_software=1): {software_count}")
    print(f"Missing positive median_review_playtime: {missing_pt}/{len(updates)}")
    if scores:
        n = len(scores)
        print(
            f"Distribution (games only): "
            f"p10={scores[int(0.10*n)]:.3f} | "
            f"p50={scores[int(0.50*n)]:.3f} | "
            f"p90={scores[int(0.90*n)]:.3f}"
        )

    print("\nLowest commitment (short_burst):")
    for g in conn.execute(
        """
        SELECT name, median_review_playtime, commitment_score
        FROM games
        WHERE is_eligible = 1 AND is_software = 0 AND commitment_score IS NOT NULL
        ORDER BY commitment_score ASC
        LIMIT 10
        """
    ).fetchall():
        print(f"  {g['commitment_score']:.4f} | {g['median_review_playtime']:>5} min | {g['name']}")

    print("\nHighest commitment (long_campaign):")
    for g in conn.execute(
        """
        SELECT name, median_review_playtime, commitment_score
        FROM games
        WHERE is_eligible = 1 AND is_software = 0 AND commitment_score IS NOT NULL
        ORDER BY commitment_score DESC
        LIMIT 10
        """
    ).fetchall():
        print(f"  {g['commitment_score']:.4f} | {g['median_review_playtime']:>5} min | {g['name']}")

    conn.close()


if __name__ == "__main__":
    run_commitment()
