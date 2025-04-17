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
        k = max(int(0.03*n), 5)
        k = min(k, n)
        samples.append(grp.sample(n=k, random_state=42))
    return pd.concat(samples, ignore_index=True)

sampled_pairs = compute_sampled_pairs(ml_with_username)

# ----------------------------
# Core analysis
# ----------------------------
def run_analysis(ml, im, threshold=2):
    results = {}
    tot_uniq = ml['username'].nunique()
    results['total_unique_audience'] = tot_uniq

    uic  = ml.groupby('username')['influencerusername'].nunique()
    core = uic[uic >= threshold]
    results['total_core_users'] = core.count()
    results['core_percentage'] = results['total_core_users'] / tot_uniq * 100
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

    summary = []
    for inf, aud in inf2core.items():
        subset = im[im['influencerusername']==inf]
        summary.append({
            'influencerusername': inf,
            'median_est_view_count': subset['view_count_est'].median() if len(subset) else np.nan,
            'user_reach': len(subset),
            'num_posts': len(subset)
        })
    results['df_influencers'] = pd.DataFrame(summary).sort_values('user_reach', ascending=False)

    # marginal gains
    sorted_inf = sorted(inf2core.items(), key=lambda x: len(x[1]), reverse=True)
    mg, cum = [], set()
    for inf, aud in sorted_inf:
        new = len(aud - cum)
        mg.append((inf, new))
        cum |= aud
    results['marginal_gains'] = mg

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

# Overview
st.header("Overview Metrics")
c1,c2,c3,c4 = st.columns(4)
aud_est = int(results['total_unique_audience']/results['median_ratio'])
core_est= int(results['total_core_users']*results['median_ratio'])
c1.metric("Unique Audience", results['total_unique_audience'])
c2.metric("Audience Est.", aud_est)
c3.metric("Core Users Est.", core_est)
c4.metric("Core %", f"{results['core_percentage']:.2f}%")

# Coverage
st.header("Coverage Metrics")
st.write("• 50% coverage → 3 influencers")
st.write("• 100% coverage → 5 influencers")

# ----------------------------
# Campaign Planner — form + exclusivity on submit
# ----------------------------
st.header("Campaign Planner")
st.subheader("Influencer Network Table")
st.caption("Select Include/Exclude, then click **Update Selections**.")

# init session-state table
if 'campaign_df' not in st.session_state:
    df0 = results['df_influencers'].copy()
    mg_df = pd.DataFrame(results['marginal_gains'], columns=['influencerusername','marginal_users_added'])
    df0 = df0.merge(mg_df, on='influencerusername', how='left')
    df0['core_users_reached'] = (df0['user_reach'] * results['median_ratio']).astype(int)
    df0['Include in Network']     = False
    df0['Exclude from Analysis']  = False
    st.session_state['campaign_df'] = df0

editable_cols = [
    'Include in Network','Exclude from Analysis',
    'influencerusername','median_est_view_count',
    'core_users_reached','marginal_users_added'
]
renamed_cols = {
    'influencerusername':'Influencer',
    'median_est_view_count':'Median Views',
    'core_users_reached':'Core Users Reached',
    'marginal_users_added':'Marginal Core Users Added'
}

with st.form("campaign_selection"):
    edited = st.data_editor(
        st.session_state['campaign_df'][editable_cols].rename(columns=renamed_cols),
        use_container_width=True,
        num_rows="fixed",
        disabled=['Influencer','Median Views','Core Users Reached','Marginal Core Users Added'],
        key="editor_table"
    )
    submitted = st.form_submit_button("Update Selections")

if submitted:
    # map back original names
    real = edited.rename(columns={v:k for k,v in renamed_cols.items()})
    # enforce "not both checked" rule
    real['Exclude from Analysis'] = real.apply(
        lambda r: False if (r['Include in Network'] and r['Exclude from Analysis']) else r['Exclude from Analysis'], axis=1
    )
    real['Include in Network'] = real.apply(
        lambda r: False if (r['Include in Network'] and r['Exclude from Analysis']) else r['Include in Network'], axis=1
    )
    for col in ['Include in Network','Exclude from Analysis']:
        st.session_state['campaign_df'][col] = real[col]
    st.success("Selections updated!")

if st.button("Reset Influencer Selections"):
    st.session_state['campaign_df']['Include in Network']    = False
    st.session_state['campaign_df']['Exclude from Analysis'] = False

# ----------------------------
# Campaign Metrics
# ----------------------------
st.subheader("Campaign Metrics")
df_sel = st.session_state['campaign_df']
sel    = df_sel[df_sel['Include in Network'] & ~df_sel['Exclude from Analysis']]

imp   = int(sel['median_est_view_count'].sum())
reach = int(sel['user_reach'].sum())
coreR = int(sel['core_users_reached'].sum())

m1,m2,m3,m4 = st.columns(4)
m1.metric("Total Impressions",    f"{imp:,}")
m2.metric("Estimated Reach",      f"{reach:,}")
m3.metric("Core Impressions (~30%)", f"{int(imp*0.3):,}")
m4.metric("Core Reach",           f"{coreR:,}")

# ----------------------------
# PyVis Network (unchanged)
# ----------------------------
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True)
net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=250, spring_strength=0.001)

reach_map  = ml_with_username.groupby('influencerusername')['username'].nunique()
total_aud  = reach_map.sum()
top5       = reach_map.sort_values(ascending=False).head(5).index.tolist()

for inf in sampled_pairs['influencerusername'].unique():
    r   = reach_map.get(inf,0)
    sz  = 20 + 2*math.sqrt(r)
    pct = (r/total_aud)*100
    title = f"{inf}: {r} users ({pct:.1f}%)"
    if inf in top5:
        net.add_node(inf, label=inf, title=title, shape="circularImage",
                     image=f"{inf}.jpg", size=sz, color="red")
    else:
        net.add_node(inf, label=inf, title=title, size=sz, color="gray")

aud = (sampled_pairs.groupby('username')['influencerusername']
       .apply(lambda x: frozenset(x))
       .reset_index(name='influencers_set'))
aud = aud.groupby('influencers_set')['username'].apply(list).reset_index(name='audience_list')
aud['group_id'] = aud.index.map(lambda i: f"Group_{i}")

for _, row in aud.iterrows():
    size = 5 + 10 * math.log(len(row['audience_list'])+1)
    net.add_node(row['group_id'], label=str(len(row['audience_list'])),
                 title=f"Size: {len(row['audience_list'])}",
                 size=size, color="lightblue")

for _, row in aud.iterrows():
    w = 1 + math.log(len(row['audience_list'])+1)
    for inf in row['influencers_set']:
        clr = "rgba(255,0,0,0.5)" if inf in top5 else "rgba(128,128,128,0.5)"
        net.add_edge(row['group_id'], inf, width=w, color=clr)

graph_html = net.generate_html(notebook=True)
components.html(graph_html, height=850, scrolling=True)
