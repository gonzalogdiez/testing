# streamlit_app.py

import streamlit as st
import requests
import re
import json
from collections import Counter
from openai import OpenAI

# ─── Configuration & Secrets ────────────────────────────────────────────────
# ~/.streamlit/secrets.toml:
# [secrets]
# OPENAI_API_KEY = "sk-..."
# APIFY_BASE_URL   = "http://167.99.6.240:8002"
client         = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
APIFY_BASE_URL = st.secrets["APIFY_BASE_URL"]

# Sidebar settings
st.sidebar.header("Settings")
RESULTS_LIMIT = st.sidebar.slider(
    "Top media per hashtag",
    min_value=5, max_value=50, value=20,
    key="settings_results_limit"
)

HASHTAG_RE = re.compile(r"#(\w+)")

def generate_initial_hashtags(topic: str, brand: str, market: str) -> list[str]:
    prompt = (
        f"Generate 8–10 hashtags for a brand about “{brand}”, "
        f"interested in “{topic}” and targeting the “{market}” market. "
        "Return them only as a JSON array of strings—no extra text or fences."
    )
    for model in ("gpt-4o-mini", "gpt-3.5-turbo"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.7,
                max_tokens=150
            )
            raw = resp.choices[0].message.content
            # extract array substring
            start, end = raw.find("["), raw.rfind("]")
            if start == -1 or end == -1:
                st.error(f"No JSON array found in {model} response.")
                continue
            snippet = raw[start:end+1]
            tags = json.loads(snippet)
            if isinstance(tags, list):
                return tags
            st.error(f"Parsed JSON not a list from {model}:\n{snippet}")
            return []
        except json.JSONDecodeError:
            st.error(f"JSON parse error from {model}:\n{snippet}")
            return []
        except Exception as e:
            st.warning(f"OpenAI error with {model}: {e}")
    st.error("All OpenAI attempts failed. Check API key & model access.")
    return []

def fetch_top_media(hashtags: list[str]) -> list[dict]:
    url = f"{APIFY_BASE_URL}/hashtags/top-media"
    payload = {"hashtags": hashtags, "results_limit": RESULTS_LIMIT}
    try:
        st.write(f"▶️ POST {url} payload={payload}")
        r = requests.post(url, json=payload, timeout=30)
        st.write(f"⏪ Status: {r.status_code}")
        data = r.json()
        st.write("⏪ Keys:", list(data.keys()))
        media = data.get("media") or data.get("results") or data.get("items") or []
        st.write(f"▶️ Returning {len(media)} media items")
        return media
    except Exception as e:
        st.error(f"Error fetching top media: {e}")
        return []

def extract_hashtags_from_media(media_list: list[dict]) -> list[str]:
    out = []
    for item in media_list:
        caption = item.get("caption", "") or ""
        out.extend(f"#{h}" for h in HASHTAG_RE.findall(caption))
    return out

def enrich_hashtags(seeds: list[str]) -> list[str]:
    media  = fetch_top_media(seeds)
    raw    = extract_hashtags_from_media(media)
    counts = Counter(raw)
    seedset = {s.lstrip("#").lower() for s in seeds}
    for tag in list(counts):
        if tag.lstrip("#").lower() in seedset:
            counts.pop(tag, None)
    return [tag for tag,_ in counts.most_common(20)]

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.title("Hashtag Mapping Prototype")

# Step 1
st.header("Step 1: Define Your Campaign")
topic  = st.text_input("Topic", key="ui_topic")
brand  = st.text_input("Brand", key="ui_brand")
market = st.text_input("Market", key="ui_market")

if st.button("Generate Initial Hashtags", key="btn_generate"):
    if topic and brand and market:
        st.session_state.initial = generate_initial_hashtags(topic, brand, market)
    else:
        st.warning("Please fill in Topic, Brand, and Market.")

# Step 2
if "initial" in st.session_state:
    st.header("Step 2: Confirm Seed Hashtags")
    selected_initial = st.multiselect(
        "Select hashtags to keep",
        options=st.session_state.initial,
        default=st.session_state.initial,
        key="ui_select_initial"
    )
    if st.button("Enrich Hashtags", key="btn_enrich"):
        st.write("🔍 Enriching seeds:", selected_initial)
        st.session_state.enriched = enrich_hashtags(selected_initial)

# Step 3
if "enriched" in st.session_state:
    st.header("Step 3: Select Final Hashtags")
    selected_final = st.multiselect(
        "Pick your final hashtags",
        options=st.session_state.enriched,
        default=st.session_state.enriched,
        key="ui_select_final"
    )
    if st.button("Finalize", key="btn_finalize"):
        st.session_state.final = selected_final

# Output
if "final" in st.session_state:
    st.header("Final Hashtags")
    st.write(st.session_state.final)
