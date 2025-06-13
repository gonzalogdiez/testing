import streamlit as st
import requests
import re
import json
from collections import Counter

# New style v1 client
from openai import OpenAI
from openai.error import OpenAIError

# ─── Configuration & Secrets ────────────────────────────────────────────────
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
APIFY_BASE_URL = st.secrets["APIFY_BASE_URL"]

st.sidebar.header("Settings")
RESULTS_LIMIT = st.sidebar.slider("Top media per hashtag", 5, 50, 20)

HASHTAG_RE = re.compile(r"#(\w+)")

def generate_initial_hashtags(topic: str, brand: str, market: str) -> list[str]:
    """
    Calls OpenAI to generate seed hashtags.
    Catches any API errors and shows them in Streamlit.
    Falls back to gpt-3.5-turbo if gpt-4o-mini fails.
    """
    prompt = (
        f"Generate 8–10 hashtags for a brand about “{brand}”, "
        f"interested in “{topic}” and targeting the “{market}” market. "
        "Return them as a JSON array of strings."
    )

    for model in ("gpt-4o-mini", "gpt-3.5-turbo"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.7,
                max_tokens=150,
            )
            content = resp.choices[0].message.content.strip()
            tags = json.loads(content)
            if isinstance(tags, list):
                return tags
            st.error(f"OpenAI returned non-list JSON when using {model}: {content}")
            return []
        except OpenAIError as oe:
            st.warning(f"OpenAI model {model} error: {oe}")
        except json.JSONDecodeError:
            st.error(f"Failed to parse JSON from OpenAI {model} response:\n{content}")
            return []
        except Exception as e:
            st.error(f"Unexpected error with OpenAI {model}: {e}")

    # If we exhausted both models
    st.error("All OpenAI attempts failed. Check your API key, model access, and network.")
    return []

def fetch_top_media(hashtags: list[str]) -> list[dict]:
    try:
        r = requests.post(
            f"{APIFY_BASE_URL}/hashtags/top-media",
            json={"hashtags": hashtags, "results_limit": RESULTS_LIMIT},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("media", [])
    except Exception as e:
        st.error(f"Error fetching top media: {e}")
        return []

def extract_hashtags_from_media(media_list: list[dict]) -> list[str]:
    out = []
    for item in media_list:
        caption = item.get("caption","") or ""
        out.extend([f"#{h}" for h in HASHTAG_RE.findall(caption)])
    return out

def enrich_hashtags(seeds: list[str]) -> list[str]:
    media  = fetch_top_media(seeds)
    raw    = extract_hashtags_from_media(media)
    counts = Counter(raw)
    seedset = {s.lstrip("#").lower() for s in seeds}
    for tag in list(counts):
        if tag.lstrip("#").lower() in seedset:
            del counts[tag]
    return [tag for tag,_ in counts.most_common(20)]

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.title("Hashtag Mapping Prototype")

# Step 1
st.header("Step 1: Define Your Campaign")
topic  = st.text_input("Topic")
brand  = st.text_input("Brand")
market = st.text_input("Market")
if st.button("Generate Initial Hashtags"):
    if topic and brand and market:
        st.session_state.initial = generate_initial_hashtags(topic,brand,market)
    else:
        st.warning("Please fill in Topic, Brand, and Market.")

# Step 2
if "initial" in st.session_state:
    st.header("Step 2: Confirm Seed Hashtags")
    sel = st.multiselect(
        "Select hashtags to keep",
        options=st.session_state.initial,
        default=st.session_state.initial
    )
    if st.button("Enrich Hashtags"):
        st.session_state.enriched = enrich_hashtags(sel)

# Step 3
if "enriched" in st.session_state:
    st.header("Step 3: Select Final Hashtags")
    final_sel = st.multiselect(
        "Pick your final hashtags",
        options=st.session_state.enriched,
        default=st.session_state.enriched
    )
    if st.button("Finalize"):
        st.session_state.final = final_sel

# Output
if "final" in st.session_state:
    st.header("Final Hashtags")
    st.write(st.session_state.final)
