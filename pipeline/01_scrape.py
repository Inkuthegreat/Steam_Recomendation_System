import argparse
import json
import random
import re
import sqlite3
import statistics
import time
import requests
from src.config import DB_PATH

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
})


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            appid INTEGER PRIMARY KEY,
            name TEXT,
            owners TEXT,
            ccu INTEGER,
            positive INTEGER,
            negative INTEGER,
            is_eligible INTEGER DEFAULT 0,
            short_description TEXT,
            genres_json TEXT,
            tags_json TEXT,
            median_review_playtime INTEGER,
            scraped_store INTEGER DEFAULT 0,
            scraped_reviews INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            recommendationid TEXT PRIMARY KEY,
            appid INTEGER,
            review_text TEXT,
            playtime_forever INTEGER,
            voted_up INTEGER,
            votes_helpful INTEGER,
            FOREIGN KEY (appid) REFERENCES games(appid)
        )
    """)
    conn.commit()
    return conn


def get_json_with_backoff(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code in (429, 403):
                wait = (60 * (2 ** attempt)) + random.uniform(1, 5)
                print(f"[{resp.status_code}] Rate limited on {url}. Sleeping {wait:.1f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            if not resp.text.strip():
                raise ValueError("Empty response body")

            return resp.json()
        except (requests.RequestException, ValueError) as e:
            preview = ""
            if "resp" in locals() and resp is not None:
                preview = resp.text[:120].replace("\n", " ")
            wait = (5 * (2 ** attempt)) + random.uniform(0.5, 2.0)
            print(f"Error on {url} ({params}): {e} | Preview: {preview!r}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    return None

def stage_seed(pages=10):
    """Stage 1: Pull top 10,000 games from SteamSpy and filter locally."""
    conn = init_db()
    cur = conn.cursor()

    for page in range(pages):
        print(f"[Seed] Fetching SteamSpy page {page}/{pages - 1}...")
        data = get_json_with_backoff(
            "https://steamspy.com/api.php",
            params={"request": "all", "page": page},
        )
        if not data:
            print(f"Skipping page {page} after repeated failures.")
            continue

        rows = []
        for g in data.values():
            pos = int(g.get("positive", 0))
            neg = int(g.get("negative", 0))
            total = pos + neg
            ratio = (pos / total) if total > 0 else 0.0
            eligible = 1 if (total >= 100 and ratio >= 0.40) else 0

            rows.append((
                int(g["appid"]),
                g.get("name", ""),
                g.get("owners", ""),
                int(g.get("ccu", 0)),
                pos,
                neg,
                eligible,
            ))

        cur.executemany(
            """
            INSERT INTO games (appid, name, owners, ccu, positive, negative, is_eligible)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(appid) DO UPDATE SET
                owners=excluded.owners,
                ccu=excluded.ccu,
                positive=excluded.positive,
                negative=excluded.negative,
                is_eligible=excluded.is_eligible
            """,
            rows,
        )
        conn.commit()
        print(f"[Seed] Saved {len(rows)} games from page {page}.")

        if page < pages - 1:
            time.sleep(62)  # SteamSpy 'all' limit: 1 req / 60s

    eligible_count = cur.execute("SELECT COUNT(*) FROM games WHERE is_eligible = 1").fetchone()[0]
    total_count = cur.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"[Seed] Complete. {eligible_count}/{total_count} games passed filters.")
    conn.close()


def stage_store(limit=None):
    """Stage 2: Fetch Steam Store short_description/genres + SteamSpy tags."""
    conn = init_db()
    cur = conn.cursor()

    query = "SELECT appid FROM games WHERE is_eligible = 1 AND scraped_store = 0"
    if limit:
        query += f" LIMIT {int(limit)}"
    appids = [r[0] for r in cur.execute(query).fetchall()]
    print(f"[Store] Scraping metadata for {len(appids)} games...")

    for idx, appid in enumerate(appids, 1):
        # 1. Steam Store API
        store_data = get_json_with_backoff(
            "https://store.steampowered.com/api/appdetails",
            params={"appids": appid, "l": "english"},
        )
        short_desc, genres = "", []
        if store_data and str(appid) in store_data and store_data[str(appid)].get("success"):
            info = store_data[str(appid)]["data"]
            short_desc = info.get("short_description", "")
            genres = [g["description"] for g in info.get("genres", [])]

        time.sleep(1.5)  # Safe Storefront ceiling

        # 2. SteamSpy tags
        spy_data = get_json_with_backoff(
            "https://steamspy.com/api.php",
            params={"request": "appdetails", "appid": appid},
        )
        tags = spy_data.get("tags", {}) if isinstance(spy_data, dict) else {}
        if not isinstance(tags, dict):
            tags = {}

        cur.execute(
            """
            UPDATE games
            SET short_description = ?, genres_json = ?, tags_json = ?, scraped_store = 1
            WHERE appid = ?
            """,
            (short_desc, json.dumps(genres), json.dumps(tags), appid),
        )
        conn.commit()

        if idx % 25 == 0 or idx == len(appids):
            print(f"[Store] {idx}/{len(appids)} done.")
        time.sleep(1.0)

    conn.close()


def clean_review(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stage_reviews(limit=None, num_per_page=20):
    """Stage 3: Fetch top helpful English reviews via IUserReviewsService."""
    conn = init_db()
    cur = conn.cursor()

    query = "SELECT appid FROM games WHERE is_eligible = 1 AND scraped_reviews = 0"
    if limit:
        query += f" LIMIT {int(limit)}"
    appids = [r[0] for r in cur.execute(query).fetchall()]
    print(f"[Reviews] Scraping reviews for {len(appids)} games...")

    url = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"

    for idx, appid in enumerate(appids, 1):
        payload = {
            "appid": appid,
            "filter": 0,                 # 0 = Most Helpful
            "languages": ["english"],
            "purchase_type": 1,          # 1 = All purchases
            "num_per_page": num_per_page,
            "use_review_quality": True,  # Downranks ASCII art & memes
        }

        data = get_json_with_backoff(
            url,
            params={"input_json": json.dumps(payload)},
        )

        # Service interface wraps output in "response"
        resp_body = data.get("response", {}) if isinstance(data, dict) else {}
        reviews_list = resp_body.get("reviews", [])

        playtimes = []
        review_rows = []
        for rev in reviews_list:
            pt = int(rev.get("author", {}).get("playtime_forever", 0))
            if pt > 0:
                playtimes.append(pt)

            text = clean_review(rev.get("review", ""))
            if len(text.split()) >= 15 and text.isascii():
                review_rows.append((
                    str(rev["recommendationid"]),
                    appid,
                    text,
                    pt,
                    1 if rev.get("voted_up") else 0,
                    int(rev.get("votes_helpful", 0)),
                ))

        if review_rows:
            cur.executemany(
                """
                INSERT OR IGNORE INTO reviews
                (recommendationid, appid, review_text, playtime_forever, voted_up, votes_helpful)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                review_rows,
            )

        median_pt = int(statistics.median(playtimes)) if playtimes else 0
        cur.execute(
            "UPDATE games SET median_review_playtime = ?, scraped_reviews = 1 WHERE appid = ?",
            (median_pt, appid),
        )
        conn.commit()

        if idx % 25 == 0 or idx == len(appids):
            print(f"[Reviews] {idx}/{len(appids)} done.")
        time.sleep(1.1)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["seed", "store", "reviews"], required=True)
    parser.add_argument("--pages", type=int, default=10, help="SteamSpy pages (1000 games/page)")
    parser.add_argument("--limit", type=int, default=None, help="Limit games for testing")
    args = parser.parse_args()

    if args.stage == "seed":
        stage_seed(pages=args.pages)
    elif args.stage == "store":
        stage_store(limit=args.limit)
    elif args.stage == "reviews":
        stage_reviews(limit=args.limit)
