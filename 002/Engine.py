import streamlit as st
import requests
import re
import json
from collections import Counter
from openai import OpenAI

# ─── Configuration & Secrets ────────────────────────────────────────────────
# ~/.streamlit/secrets.toml should contain:
# [secrets]
# OPENAI_API_KEY = "sk-..."
# APIFY_BASE_URL = "http://167.99.6.240:8002"
client         = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
APIFY_BASE_URL = st.secrets["APIFY_BASE_URL"]

# Sidebar: number of posts to fetch per seed
st.sidebar.header("Settings")
RESULTS_LIMIT = st.sidebar.slider("Top media per hashtag", 5, 50, 20)

# Regex for extracting hashtags
HASHTAG_RE = re.compile(r"#(\w+)")

def generate_initial_hashtags(topic: str, brand: str, market: str) -> list[str]:
    """
    Step 1: Call OpenAI to generate 8–10 seed hashtags.
    Strips any surrounding markdown/text so we can json.loads the array.
    """
    prompt = (
        f"Generate 8–10 hashtags for a brand about “{brand}”, "
        f"interested in “{topic}” and targeting the “{market}” market. "
        "Return them *only* as a JSON array of strings—no extra text or markdown fences."
    )
    # Try both models
    for model in ("gpt-4o-mini", "gpt-3.5-turbo"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
            )
            raw = resp.choices[0].message.content

            # Extract the JSON array between the first '[' and last ']'
            start = raw.find("[")
            end   = raw.rfind("]")
            if start == -1 or end == -1:
                st.error(f"No JSON array found in {model} response.")
                continue

            snippet = raw[start : end + 1]
            tags = json.loads(snippet)
            if isinstance(tags, list):
                return tags
            st.error(f"Parsed JSON is not a list (model {model}):\n{snippet}")
            return []
        except json.JSONDecodeError:
            st.error(f"Failed to parse JSON from {model}:\n{snippet}")
            return []
        except Exception as e:
            st.warning(f"OpenAI error with {model}: {e}")

    st.error("All OpenAI attempts failed. Check your key & model access.")
    return []

def fetch_top_media(hashtags: list[str]) -> list[dict]:
    """
    Step 2: Call Apify endpoint to fetch top media for those hashtags.
    """
    try:
        r = requests.post(
            f"{APIFY_BASE_URL}/hashtags/top-media",
            json={"hashtags": hashtags, "results_limit": RESULTS_LIMIT},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("media", [])
    except Exception as e:
        st.error(f"Error fetching top media: {e}")
        return []

def extract_hashtags_from_media(media_list: list[dict]) -> list[str]:
    """
    Step 3: Extract all hashtags (e.g. #tag) from each media's caption.
    """
    out = []
    for item in media_list:
        caption = item.get("caption", "") or ""
        out.extend([f"#{h}" for h in HASHTAG_RE.findall(caption)])
    return out

def enrich_hashtags(seeds: list[str]) -> list[str]:
    """
    Step 4: From fetched media, return the 20 most frequent new hashtags.
    """
    media  = fetch_top_media(seeds)
    raw    = extract_hashtags_from_media(media)
    counts = Counter(raw)
    seedset = {s.lstrip("#").lower() for s in seeds}
    for tag in list(counts):
        if tag.lstrip("#").lower() in seedset:
            del counts[tag]
    return [tag for tag, _ in counts.most_common(20)]

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.title("Hashtag Mapping Prototype")

# Step 1: Define Campaign
st.header("Step 1: Define Your Campaign")
topic  = st.text_input("Topic")
brand  = st.text_input("Brand")
market = st.text_input("Market")

if st.button("Generate Initial Hashtags"):
    if topic and brand and market:
        st.session_state.initial = generate_initial_hashtags(topic, brand, market)
    else:
        st.warning("Please fill in Topic, Brand, and Market.")

# Step 2: Confirm Seeds & Enrich
if "initial" in st.session_state:
    st.header("Step 2: Confirm Seed Hashtags")
    selected_initial = st.multiselect(
        "Select hashtags to keep",
        options=st.session_state.initial,
        default=st.session_state.initial
    )
    if st.button("Enrich Hashtags"):
        st.session_state.enriched = enrich_hashtags(selected_initial)

# Step 3: Finalize
if "enriched" in st.session_state:
    st.header("Step 3: Select Final Hashtags")
    selected_final = st.multiselect(
        "Pick your final hashtags",
        options=st.session_state.enriched,
        default=st.session_state.enriched
    )
    if st.button("Finalize"):
        st.session_state.final = selected_final

# Output final list
if "final" in st.session_state:
    st.header("Final Hashtags")
    st.write(st.session_state.final)
