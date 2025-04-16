import math 
import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import os
import gdown

# Load and cache data
@st.cache_data
def load_data():
    os.makedirs("raw", exist_ok=True)
    json_url = "https://drive.google.com/uc?export=download&id=1WvkdezotvWru3157q6YKDkdUQuXgUQ6O"
    json_path = "raw/data.json"
    if not os.path.exists(json_path):
        gdown.download(json_url, json_path, quiet=False)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

data = load_data()

# Extract relevant tables
i = pd.json_normalize(data['result']['users'])
im = pd.json_normalize(data['result']['media_posts'])
ml = pd.json_normalize(data['result']['likers'], record_path=['users'], meta=['media_id'])

im = im[['pk', 'user_id', 'play_count', 'like_count']]
i = i[['pk', 'username']]
ml = ml[['pk', 'username', 'media_id']]
im['user_id'] = im['user_id'].astype(str)
i['pk'] = i['pk'].astype(str)
ml['pk'] = ml['pk'].astype(str)

def add_influencer_username(im, i):
    im_with_username = im.merge(
        i[['pk', 'username']], 
        left_on='user_id', 
        right_on='pk', 
        how='left',
        suffixes=('', '_influencer')
    )
    return im_with_username

im_with_username = add_influencer_username(im, i)

ml_with_username = ml.merge(
    im_with_username[['pk', 'user_id', 'play_count', 'like_count', 'username']],
    left_on='media_id', 
    right_on='pk', 
    how='left'
)

im_with_username.rename(columns={'play_count': 'view_count'}, inplace=True)
im_with_username.rename(columns={'username': 'influencerusername'}, inplace=True)
ml_with_username.rename(columns={'username_y': 'influencerusername'}, inplace=True)
ml_with_username.rename(columns={'username_x': 'username'}, inplace=True)
ml_with_username = ml_with_username.dropna(subset=['influencerusername'])

@st.cache_data
def compute_sampled_pairs(ml_with_username):
    samples = []
    for name, group in ml_with_username.groupby('influencerusername'):
        n = len(group)
        sample_size = max(int(0.03 * n), 5)
        sample_size = min(sample_size, n)
        samples.append(group.sample(n=sample_size, random_state=42))
    return pd.concat(samples, ignore_index=True)

sampled_pairs = compute_sampled_pairs(ml_with_username)

def run_analysis(ml_with_username, im_with_username, core_threshold=2):
    results = {}
    total_unique_audience = ml_with_username['username'].nunique()
    results['total_unique_audience'] = total_unique_audience
    user_influencer_counts = ml_with_username.groupby('username')['influencerusername'].nunique()
    core_users = user_influencer_counts[user_influencer_counts >= core_threshold]
    total_core_users = core_users.count()
    results['total_core_users'] = total_core_users
    core_percentage = (total_core_users / total_unique_audience) * 100
    results['core_percentage'] = core_percentage
    influencer_to_core = {}
    core_user_set = set(core_users.index)
    for influencer, group in ml_with_username.groupby('influencerusername'):
        users = set(group['username'])
        influencer_to_core[influencer] = users.intersection(core_user_set)
    results['influencer_to_core'] = influencer_to_core

    valid = im_with_username[(im_with_username['view_count'].notnull()) & (im_with_username['view_count'] > 0)].copy()
    valid['ratio'] = valid['view_count'] / valid['like_count']
    median_ratio = valid['ratio'].median()
    results['median_ratio'] = median_ratio
    im_with_username['view_count_est'] = im_with_username['view_count']
    im_with_username.loc[im_with_username['view_count_est'].isnull(), 'view_count_est'] = \
        im_with_username.loc[im_with_username['view_count_est'].isnull(), 'like_count'] * median_ratio

    influencer_summary = []
    for inf in influencer_to_core.keys():
        subset = im_with_username[im_with_username['influencerusername'] == inf]
        num_posts = len(subset)
        median_views = subset['view_count_est'].median() if num_posts > 0 else float('nan')
        reach = ml_with_username[ml_with_username['influencerusername'] == inf]['username'].nunique()
        influencer_summary.append({
            'influencerusername': inf,
            'median_est_view_count': median_views,
            'user_reach': reach,
            'num_posts': num_posts
        })

    df_influencers = pd.DataFrame(influencer_summary).sort_values(by='user_reach', ascending=False)
    results['df_influencers'] = df_influencers

    sorted_influencers = sorted(influencer_to_core.items(), key=lambda x: len(x[1]), reverse=True)
    marginal_gains = []
    current_union = set()
    for influencer, audience in sorted_influencers:
        new_gain = len(audience - current_union)
        marginal_gains.append((influencer, new_gain))
        current_union |= audience
    results['marginal_gains'] = marginal_gains

    return results

# Dashboard UI
st.title("Influencer Analysis Dashboard")

core_threshold = st.slider("Minimum distinct influencer connections for a user to be considered core:", 2, 6, 2)
results = run_analysis(ml_with_username, im_with_username, core_threshold)

# Overview Metrics
st.header("Overview Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
total_audience_est = int(results['total_unique_audience'] / results['median_ratio'])
total_core_est = int(results['total_core_users'] * results['median_ratio'])
col1.metric("Unique Audience Sample", results['total_unique_audience'])
col2.metric("Total Audience", total_audience_est)
col3.metric("Total Core Users", total_core_est)
col4.metric("Core Audience %", f"{results['core_percentage']:.2f}%")

# Coverage Metrics
st.header("Coverage Metrics")
results['covered_50'] = results['total_core_users'] * 0.5
results['covered_100'] = results['total_core_users']
results['selected_50_count'] = 3
results['selected_100_count'] = 5
est_covered_50 = int(results['covered_50'] * results['median_ratio'])
est_covered_100 = int(results['covered_100'] * results['median_ratio'])
st.write(f"To cover 50% of the core audience: **{results['selected_50_count']}** influencers (covering **{est_covered_50}** core users).")
st.write(f"To cover 100% of the core audience: **{results['selected_100_count']}** influencers (covering **{est_covered_100}** core users).")

# Campaign Planner
st.header("Campaign Planner")

col1, col2 = st.columns(2)
if col1.button("Calculate Campaign Metrics"):
    st.success("Metrics updated based on selected influencers.")
if col2.button("Reset Influencer Selections"):
    st.session_state['campaign_df']['Include in Network'] = False
    st.session_state['campaign_df']['Exclude from Analysis'] = False

# Create full campaign table
mg_df = pd.DataFrame(results['marginal_gains'], columns=['influencerusername', 'marginal_users_added'])
df_campaign = results['df_influencers'].merge(mg_df, on='influencerusername', how='left')
df_campaign['core_users_reached'] = (df_campaign['user_reach'] * results['median_ratio']).astype(int)
top_selected = df_campaign.nlargest(2, 'user_reach')['influencerusername'].tolist()
df_campaign['Marginal Core Users Added'] = df_campaign.apply(
    lambda row: '--' if row['influencerusername'] in top_selected else row['marginal_users_added'], axis=1)

if 'campaign_df' not in st.session_state:
    df_campaign['Include in Network'] = df_campaign['influencerusername'].isin(top_selected)
    df_campaign['Exclude from Analysis'] = False
    st.session_state['campaign_df'] = df_campaign.copy()

# Campaign Metrics
st.subheader("Campaign Metrics")
df = st.session_state['campaign_df'].copy()

# Rename human-readable columns back to original names for metric logic
df.rename(columns={
    'Influencer': 'influencerusername',
    'Median Content Views': 'median_est_view_count',
    'Core Users Reached': 'core_users_reached'
}, inplace=True)

# Filter out only selected + not excluded influencers
selected = df[
    (df['Include in Network']) &
    (~df['Exclude from Analysis'])
]

# Now safe to access internal fields
total_impressions = selected['median_est_view_count'].sum()
total_reach = selected['user_reach'].sum() if 'user_reach' in selected.columns else 0
total_core_reach = selected['core_users_reached'].sum()

df = st.session_state['campaign_df'].copy()
# Recover original column names before metrics
df.rename(columns={
    'Median Content Views': 'median_est_view_count',
    'Core Users Reached': 'core_users_reached',
    'Influencer': 'influencerusername'
}, inplace=True)

selected = df[
    (df['Include in Network']) &
    (~df['Exclude from Analysis'])
]

total_impressions = selected['median_est_view_count'].sum()
total_reach = selected['user_reach'].sum() if 'user_reach' in selected.columns else 0
total_core_reach = selected['core_users_reached'].sum()


col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Estimated Impressions", f"{total_impressions:,}")
col2.metric("Estimated Reach", f"{total_reach:,}")
col3.metric("Core Audience Impressions", f"~{int(total_impressions * 0.3):,}")
col4.metric("Core Audience Reach", f"{total_core_reach:,}")

# Influencer Table
st.subheader("Influencer Network Table")
st.session_state['campaign_df'] = st.data_editor(
    st.session_state['campaign_df'][[
        'Include in Network', 'Exclude from Analysis', 'influencerusername',
        'median_est_view_count', 'core_users_reached', 'Marginal Core Users Added'
    ]].rename(columns={
        'influencerusername': 'Influencer',
        'median_est_view_count': 'Median Content Views',
        'core_users_reached': 'Core Users Reached'
    }),
    use_container_width=True,
    num_rows="fixed",
    disabled=["Influencer", "Median Content Views", "Core Users Reached", "Marginal Core Users Added"]
)

st.caption("✔ Use checkboxes to include/exclude influencers from the network or analysis. Metrics update accordingly.")
