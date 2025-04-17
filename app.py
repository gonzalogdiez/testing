import math
import json
import pandas as pd
import networkx as nx
from pyvis.network import Network
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import os
import gdown

# ----------------------------
# Load & cache raw JSON data
# ----------------------------
@st.cache_data
def load_data():
    os.makedirs("raw", exist_ok=True)
    json_url = "https://drive.google.com/uc?export=download&id=1WvkdezotvWru3157q6YKDkdUQuXgUQ6O"
    json_path = "raw/data.json"
    if not os.path.exists(json_path):
        gdown.download(json_url, json_path, quiet=False)
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data()

# ----------------------------
# Normalize into DataFrames
# ----------------------------
i  = pd.json_normalize(data['result']['users'])
im = pd.json_normalize(data['result']['media_posts'])
ml = pd.json_normalize(data['result']['likers'], record_path=['users'], meta=['media_id'])

im = im[['pk','user_id','play_count','like_count']]
i  = i[['pk','username']]
ml = ml[['pk','username','media_id']]

im['user_id'] = im['user_id'].astype(str)
i['pk']        = i['pk'].astype(str)
ml['pk']       = ml['pk'].astype(str)

# ----------------------------
# Merge influencer usernames
# ----------------------------
def add_influencer_username(im, i):
    return im.merge(
        i[['pk','username']],
        left_on='user_id',
        right_on='pk',
        how='left',
        suffixes=('','_influencer')
    )

im_with_username = add_influencer_username(im, i)
ml_with_username = ml.merge(
    im_with_username[['pk','user_id','play_count','like_count','username']],
    left_on='media_id', right_on='pk', how='left'
)

im_with_username.rename(
    columns={'play_count':'view_count','username':'influencerusername'},
    inplace=True
)
ml_with_username.rename(
    columns={'username_x':'username','username_y':'influencerusername'},
    inplace=True
)
ml_with_username.dropna(subset=['influencerusername'], inplace=True)

# ----------------------------
# Sample for network layout
# ----------------------------
@st.cache_data
def compute_sampled_pairs(ml):
    samples = []
    for inf, grp in ml.groupby('influencerusername'):
        n = len(grp)
        k = max(int(0.03 * n), 5)
        k = min(k, n)
        samples.append(grp.sample(n=k, random_state=42))
    return pd.concat(samples, ignore_index=True)

sampled_pairs = compute_sampled_pairs(ml_with_username)

# ----------------------------
# Core analysis
# ----------------------------
def run_analysis(ml, im, threshold=2):
    results = {}
    total_unique = ml['username'].nunique()
    results['total_unique_audience'] = total_unique

    uic  = ml.groupby('username')['influencerusername'].nunique()
    core = uic[uic >= threshold]
    results['total_core_users'] = core.count()
    results['core_percentage'] = results['total_core_users'] / total_unique * 100
    core_set = set(core.index)

    inf2core = {
        inf: set(grp['username']).intersection(core_set)
        for inf, grp in ml.groupby('influencerusername')
    }
    results['influencer_to_core'] = inf2core

    valid = im[im['view_count'] > 0].copy()
    valid['ratio'] = valid['view_count'] / valid['like_count']
    med_ratio = valid['ratio'].median()
    results['median_ratio'] = med_ratio
    im['view_count_est'] = im['view_count'].fillna(im['like_count'] * med_ratio)

    influencer_summary = []
    # pre‐compute total reach per influencer
    total_reach_map = ml_with_username.groupby('influencerusername')['username']\
                                      .nunique().to_dict()

    for inf, core_users in inf2core.items():
        # number of unique viewers
        reach = total_reach_map.get(inf, 0)
        # number of posts
        num_posts = int(im_with_username[
            im_with_username['influencerusername'] == inf
        ].shape[0])
        # median estimated views
        median_views = im_with_username[
            im_with_username['influencerusername'] == inf
        ]['view_count_est'].median() if num_posts > 0 else np.nan

        influencer_summary.append({
            'influencerusername':     inf,
            'median_est_view_count':  median_views,
            'user_reach':             reach,
            'core_users_reached':     len(core_users),
            'num_posts':              num_posts
        })

    results['df_influencers'] = (
        pd.DataFrame(influencer_summary)
          .sort_values('user_reach', ascending=False)
          .reset_index(drop=True)
    )

    sorted_inf = sorted(inf2core.items(), key=lambda x: len(x[1]), reverse=True)
    marg = []
    cum = set()
    for inf, aud in sorted_inf:
        new_gain = len(aud - cum)
        marg.append((inf, new_gain))
        cum |= aud
    results['marginal_gains'] = marg

    return results

@st.cache_data
def get_analysis(ml, im, threshold):
    return run_analysis(ml, im, threshold)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Influencer Analysis Dashboard")

core_threshold = st.slider(
    "Min distinct influencer connections for core users",
    2, 6, 2
)
results = get_analysis(ml_with_username, im_with_username, core_threshold)

# Overview Metrics
st.header("Overview Metrics")
c1, c2, c3, c4 = st.columns(4)
aud_est  = int(results['total_unique_audience'] / results['median_ratio'])
core_est = int(results['total_core_users'] * results['median_ratio'])
c1.metric("Unique Audience", results['total_unique_audience'])
c2.metric("Audience Estimate", aud_est)
c3.metric("Core Users Estimate", core_est)
c4.metric("Core Audience %", f"{results['core_percentage']:.2f}%")

# Coverage Metrics with influencer names
st.header("Coverage Metrics")
# compute minimal sets
half_core = results['total_core_users'] * 0.5
full_core = results['total_core_users']
cover50, cover100 = [], []
cov_cum = set()
for inf, gain in results['marginal_gains']:
    if len(cov_cum) < half_core:
        cover50.append(inf)
    cov_cum |= results['influencer_to_core'][inf]
# reset and get full cover
cov_cum = set()
for inf, gain in results['marginal_gains']:
    cover100.append(inf)
    cov_cum |= results['influencer_to_core'][inf]
    if len(cov_cum) >= full_core:
        break

st.write(f"• 50% coverage ({len(cover50)} influencers): {', '.join(cover50)}")
st.write(f"• 100% coverage ({len(cover100)} influencers): {', '.join(cover100)}")

# ----------------------------
# Campaign Planner — two‑step flow
# ----------------------------
st.header("Campaign Planner")
all_inf = results['df_influencers']['influencerusername'].tolist()

# Step 1: Exclude
if 'must_exclude' not in st.session_state:
    st.session_state.must_exclude = []

with st.form("exclude_form"):
    excl = st.multiselect(
        label="1) Exclude influencers",
        options=all_inf,
        default=st.session_state.must_exclude,
        label_visibility="visible"
    )
    submit_excl = st.form_submit_button("Save Exclusions")

if submit_excl:
    st.session_state.must_exclude = excl
    st.success(f"Excluded {len(excl)} influencers")

# Step 2: Include + auto‑select to target coverage
remaining = [i for i in all_inf if i not in st.session_state.must_exclude]
if 'must_include' not in st.session_state:
    st.session_state.must_include = []

with st.form("include_form"):
    inc = st.multiselect(
        label="2) Must‑include influencers",
        options=remaining,
        default=st.session_state.must_include,
        label_visibility="visible"
    )
    target_pct = st.slider(
        "Desired core coverage (%)",
        min_value=10, max_value=100, value=50, step=5
    )
    submit_inc = st.form_submit_button("Save Includes & Compute")

if submit_inc:
    st.session_state.must_include = inc
    # greedy auto‑selection
    inf2core = results['influencer_to_core']
    total_core = results['total_core_users']
    needed = target_pct / 100 * total_core
    selected = set(st.session_state.must_include)
    covered = set().union(*(inf2core[i] for i in selected))
    for influencer, gain in results['marginal_gains']:
        if influencer in selected or influencer in st.session_state.must_exclude:
            continue
        new_cov = len(covered | inf2core[influencer])
        if new_cov <= len(covered):
            continue
        selected.add(influencer)
        covered |= inf2core[influencer]
        if len(covered) >= needed:
            break
    st.session_state.auto_selected = [i for i in selected if i not in st.session_state.must_include]
    st.success(f"Auto‑selected {len(st.session_state.auto_selected)} more to reach ~{target_pct}% coverage")

# Build final selection
must_excl = set(st.session_state.must_exclude)
must_inc  = set(st.session_state.must_include)
auto_sel  = set(st.session_state.get('auto_selected', []))
final_selected = list(must_inc | auto_sel)

# ----------------------------
# Campaign Metrics
# ----------------------------
st.subheader("Campaign Metrics")

# grab the DataFrame you built in run_analysis (it already has real core_users_reached)
df_inf = results['df_influencers'].copy().set_index('influencerusername')

# sum over your final_selected list
imp_total   = int(df_inf.loc[final_selected, 'median_est_view_count'].sum())
reach_total = int(df_inf.loc[final_selected, 'user_reach'].sum())
core_reach  = int(df_inf.loc[final_selected, 'core_users_reached'].sum())
core_impr   = int(imp_total * 0.3)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Impressions",          f"{imp_total:,}")
m2.metric("Estimated Reach",            f"{reach_total:,}")
m3.metric("Core Audience Impr. (~30%)", f"{core_impr:,}")
m4.metric("Core Audience Reach",        f"{core_reach:,}")


# ----------------------------
# PyVis Network Visualization (highlight final_selected)
# ----------------------------
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True)
net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=250, spring_strength=0.001)

reach_map = ml_with_username.groupby('influencerusername')['username'].nunique()
total_aud = reach_map.sum()

for inf in sampled_pairs['influencerusername'].unique():
    r   = reach_map.get(inf, 0)
    sz  = 20 + 2 * math.sqrt(r)
    # only show label if selected
    lbl = inf if inf in final_selected else ""
    clr = "green" if inf in final_selected else "gray"
    net.add_node(inf, label=lbl, title=f"{inf}: {r} users", size=sz, color=clr)

# audience groups
aud = (sampled_pairs.groupby('username')['influencerusername']
       .apply(lambda x: frozenset(x)).reset_index(name='influencers_set'))
aud = aud.groupby('influencers_set')['username'].apply(list).reset_index(name='audience_list')
aud['group_id'] = aud.index.map(lambda i: f"Group_{i}")

for _, row in aud.iterrows():
    size = 5 + 10 * math.log(len(row['audience_list']) + 1)
    net.add_node(row['group_id'], label=str(len(row['audience_list'])),
                 title=f"Size: {len(row['audience_list'])}", size=size, color="lightblue")

for _, row in aud.iterrows():
    w = 1 + math.log(len(row['audience_list']) + 1)
    for inf in row['influencers_set']:
        clr = "rgba(0,0,0,0.2)"
        net.add_edge(row['group_id'], inf, width=w, color=clr)

graph_html = net.generate_html(notebook=True)
components.html(graph_html, height=850, scrolling=True)

# ----------------------------
# Influencer Detail Table
# ----------------------------
st.header("Influencer Details")
detail_df = results['df_influencers'].copy()
detail_df['Selected'] = detail_df['influencerusername'].isin(final_selected)

st.dataframe(
    detail_df.rename(columns={
        'influencerusername':  'Influencer',
        'user_reach':          'Reach',
        'core_users_reached':  'Core Users Reached',
        'median_est_view_count':'Median Views',
        'num_posts':           'Num Posts'
    })[[
        'Influencer','Reach','Core Users Reached','Median Views','Num Posts','Selected'
    ]],
    use_container_width=True
)
st.subheader("Raw Influencer Data")
st.dataframe(df_inf)

