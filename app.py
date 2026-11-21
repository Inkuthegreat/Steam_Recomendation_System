import streamlit as st
from src.recommender import VibeRecommender

st.set_page_config(page_title="Steam Vibe Recommender", layout="wide")


@st.cache_resource(show_spinner="Loading catalog and embeddings...")
def get_recommender() -> VibeRecommender:
    return VibeRecommender()


rec = get_recommender()

st.title("Steam Vibe Recommender")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Vibe Filters")

    commitment_range = st.slider(
        "Time Investment (Quick ↔ Massive)",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        help="0.0 = short burst / quick finish, 1.0 = 100+ hr time sink",
    )

    intensity_range = st.slider(
        "Intensity (Chill ↔ Intense)",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        help="0.0 = cozy/relaxing, 1.0 = stressful/high-pressure",
    )

    singleplayer_only = st.toggle("Singleplayer Only", value=True)
    top_k = st.slider("Results", min_value=5, max_value=30, value=12, step=1)

# --- Main Search Inputs ---
col_q, col_seed = st.columns([2, 1])

with col_q:
    query = st.text_input(
        "Describe what you're in the mood for",
        placeholder="e.g. atmospheric detective mystery, brutal tactical combat, cozy puzzle...",
    )

with col_seed:
    titles = rec.all_titles()
    title_options = [None] + titles
    selected_seed = st.selectbox(
        "Similar to (optional)",
        options=title_options,
        format_func=lambda x: "— None —" if x is None else x[1],
    )
    seed_appid = selected_seed[0] if selected_seed else None

# --- Run Recommendation ---
results = rec.recommend(
    query=query,
    seed_appid=seed_appid,
    commitment_range=commitment_range,
    intensity_range=intensity_range,
    singleplayer_only=singleplayer_only,
    top_k=top_k,
)

st.caption(f"Showing {len(results)} matches")

if not results:
    st.warning("No games match those filters. Try widening the Commitment or Intensity sliders.")
else:
    for r in results:
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.subheader(r["name"])
                if r["short_description"]:
                    st.write(r["short_description"])
                if r["top_tags"]:
                    st.caption("Tags: " + " · ".join(r["top_tags"]))
            with right:
                st.metric("Match", f"{r['score']:.2f}")
                st.caption(
                    f"Commitment: **{r['commitment_score']:.2f}** (~{r['median_hours']}h)\n\n"
                    f"Intensity: **{r['intensity_score']:.2f}**"
                )
                st.link_button("Steam Store", r["store_url"], width="stretch")
