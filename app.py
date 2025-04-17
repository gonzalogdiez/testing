import math
import os
import json
import gdown
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

@st.cache_data
def load_data():
    os.makedirs("raw", exist_ok=True)
    url = "https://drive.google.com/uc?export=download&id=1WvkdezotvWru3157q6YKDkdUQuXgUQ6O"
    path = "raw/data.json"
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)
    with open(path, encoding="utf-8") as f:
        return json.load(f)

data = load_data()

users = pd.json_normalize(data['result']['users'])[['pk','username']].astype(str)
posts = pd.json_normalize(data['result']['media_posts'])[['pk','user_id','play_count','like_count']]
likers = pd.json_normalize(data['result']['likers'], record_path=['users'], meta=['media_id'])[['pk','username','media_id']]

posts['user_id'] = posts['user_id'].astype(str)
users['pk'] = users['pk'].astype(str)
likers['pk'] = likers['pk'].astype(str)

posts = (
    posts
    .merge(users, left_on='user_id', right_on='pk', how='left')
    .rename(columns={'play_count':'view_count','username':'influencerusername'})
)
likers = (
    likers
    .merge(posts[['pk','influencerusername']], left_on='media_id', right_on='pk', how='left')
)
likers.dropna(subset=['influencerusername'], inplace=True)

@st.cache_data
def sample_pairs(lk):
    samples = []
    for inf, grp in lk.groupby('influencerusername'):
        n = len(grp)
        k = min(max(int(0.03 * n), 5), n)
        samples.append(grp.sample(k, random_state=42))
    return pd.concat(samples, ignore_index=True)

sampled_pairs = sample_pairs(likers)

def run_analysis(lk, ps, threshold):
    res = {}
    res['total_unique_audience'] = lk['username'].nunique()
    uic = lk.groupby('username')['influencerusername'].nunique()
    core = uic[uic >= threshold].index
    res['total_core_users'] = len(core)
    res['core_percentage'] = len(core) / res['total_unique_audience'] * 100
    res['influencer_to_core'] = {
        inf: set(grp['username']).intersection(core)
        for inf, grp in lk.groupby('influencerusername')
    }
    reach_map = lk.groupby('influencerusername')['username'].nunique().to_dict()

    valid_all = ps[ps['view_count'] > 0]
    global_ratio = (valid_all['view_count'] / valid_all['like_count']).median()

    summary = []
    for inf, aud in res['influencer_to_core'].items():
        df_inf = ps[ps['influencerusername'] == inf].copy()
        valid_inf = df_inf[df_inf['view_count'] > 0]
        ratio = (valid_inf['view_count'] / valid_inf['like_count']).median() if not valid_inf.empty else global_ratio
        df_inf['view_est'] = np.where(
            df_inf['view_count'] > 0,
            df_inf['view_count'],
            df_inf['like_count'] * ratio
        )
        summary.append({
            'influencerusername':    inf,
            'median_est_view_count': df_inf['view_est'].median(),
            'total_engagement':      df_inf['like_count'].sum(),
            'user_reach':            reach_map.get(inf, 0),
            'core_users_reached':    len(aud),
            'num_posts':             len(df_inf)
        })

    df_inf = pd.DataFrame(summary).sort_values('user_reach', ascending=False).reset_index(drop=True)
    res['df_influencers'] = df_inf

    sorted_inf = sorted(res['influencer_to_core'].items(), key=lambda x: len(x[1]), reverse=True)
    marg, cum = [], set()
    for inf, aud in sorted_inf:
        marg.append((inf, len(aud - cum)))
        cum |= aud
    res['marginal_gains'] = marg

    return res

@st.cache_data
def get_analysis(lk, ps, thr):
    return run_analysis(lk, ps, thr)

st.title("Influencer Analysis")

threshold = st.slider("Min connections for core users", 2, 6, 2)
results = get_analysis(likers, posts, threshold)
df_inf = results['df_influencers'].set_index('influencerusername')

st.header("Overview Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Audience", results['total_unique_audience'])
c2.metric("Core Users", results['total_core_users'])
c3.metric("Core %", f"{results['core_percentage']:.1f}%")
c4.metric("Avg Posts", f"{df_inf['num_posts'].mean():.1f}")

st.header("Coverage Metrics")
half, full = results['total_core_users'] * 0.5, results['total_core_users']
cover50, cover100, covered = [], [], set()
for inf, _ in results['marginal_gains']:
    if len(covered) < half:
        cover50.append(inf)
    covered |= results['influencer_to_core'][inf]
covered = set()
for inf, _ in results['marginal_gains']:
    cover100.append(inf)
    covered |= results['influencer_to_core'][inf]
    if len(covered) >= full:
        break
st.write(f"50% coverage ({len(cover50)}): {', '.join(cover50)}")
st.write(f"100% coverage ({len(cover100)}): {', '.join(cover100)}")

st.header("Campaign Planner")
if 'exclude' not in st.session_state:
    st.session_state.exclude = []
if 'include' not in st.session_state:
    st.session_state.include = []

with st.form("exclude_form"):
    excl = st.multiselect("Exclude influencers", df_inf.index.tolist(), default=st.session_state.exclude)
    if st.form_submit_button("Save"):
        st.session_state.exclude = excl

remaining = [i for i in df_inf.index if i not in st.session_state.exclude]
with st.form("include_form"):
    inc = st.multiselect("Include influencers", remaining, default=st.session_state.include)
    target = st.slider("Target core coverage (%)", 10, 100, 50, 5)
    if st.form_submit_button("Go"):
        st.session_state.include = inc
        needed = target / 100 * results['total_core_users']
        sel = set(st.session_state.include)
        covered = set().union(*(results['influencer_to_core'][i] for i in sel))
        for inf, _ in results['marginal_gains']:
            if inf in sel or inf in st.session_state.exclude:
                continue
            new_cov = len(covered | results['influencer_to_core'][inf])
            if new_cov > len(covered):
                sel.add(inf)
                covered |= results['influencer_to_core'][inf]
            if len(covered) >= needed:
                break
        st.session_state.auto = list(sel - set(st.session_state.include))
        st.success(f"Auto‑selected {len(st.session_state.auto)} more")

final = list(set(st.session_state.include + st.session_state.get('auto', [])))

st.subheader("Campaign Metrics")
imp = int(df_inf.loc[final, 'median_est_view_count'].sum())
eng = int(df_inf.loc[final, 'total_engagement'].sum())
rch = int(df_inf.loc[final, 'user_reach'].sum())
cr  = int(df_inf.loc[final, 'core_users_reached'].sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Views", f"{imp:,}")
m2.metric("Total Engagement", f"{eng:,}")
m3.metric("Core Impr (~30%)", f"{int(imp * 0.3):,}")
m4.metric("Core Reach", f"{cr:,}")

net = Network(height="600px", width="100%", notebook=True)
reach_map = likers.groupby('influencerusername')['username'].nunique()
for inf in sampled_pairs['influencerusername'].unique():
    r = reach_map.get(inf, 0)
    size = 20 + 2 * math.sqrt(r)
    label = inf if inf in final else ""
    color = "green" if inf in final else "gray"
    net.add_node(inf, label=label, title=str(r), size=size, color=color)

aud = (
    sampled_pairs
    .groupby('username')['influencerusername']
    .apply(lambda x: frozenset(x))
    .reset_index(name='inf_set')
)
aud = aud.groupby('inf_set')['username'].apply(list).reset_index(name='users')
aud['id'] = aud.index.map(lambda i: f"G{i}")

for _, row in aud.iterrows():
    sz = 5 + 10 * math.log(len(row['users']) + 1)
    net.add_node(row['id'], label=str(len(row['users'])), size=sz, color="lightblue")
    w = 1 + math.log(len(row['users']) + 1)
    for inf in row['inf_set']:
        net.add_edge(row['id'], inf, width=w, color="rgba(0,0,0,0.2)")

html = net.generate_html(notebook=True)
components.html(html, height=650)

st.header("Influencer Details")
detail = df_inf.reset_index().rename(columns={
    'median_est_view_count': 'Median Views',
    'total_engagement':      'Engagement',
    'user_reach':            'Reach',
    'core_users_reached':    'Core Users',
    'num_posts':             'Posts'
})
detail['Selected'] = detail['influencerusername'].isin(final)
st.dataframe(detail, use_container_width=True)
