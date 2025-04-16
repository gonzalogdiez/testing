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

# ================================ 
# Create PyVis Network for Dashboard
# ================================
# (This is separate from the analysis; we embed this graph in the dashboard.)
net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True)
net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=250, spring_strength=0.001)

# Use the previously computed top influencers and influencer_reach
influencer_reach = ml_with_username.groupby('influencerusername')['username'].nunique()
total_audience = len(set(ml_with_username['username']))
influencer_reach_sorted = influencer_reach.sort_values(ascending=False)
top_influencers = influencer_reach_sorted.head(5).index.tolist()

# Create influencer nodes with a square-root scaling and profile images for top influencers.
base_influencer_size = 20
scaling_factor_influencer = 2
sampled_influencer_nodes = sampled_pairs['influencerusername'].unique()
for influencer in sampled_influencer_nodes:
    reach = influencer_reach.get(influencer, 0)
    node_size = base_influencer_size + scaling_factor_influencer * math.sqrt(reach)
    pct = (reach / total_audience) * 100
    label_text = f"{influencer}\n{pct:.2f}%"
    title_text = f"Influencer: {influencer}\nReach: {reach} ({pct:.2f}%)"
    if influencer in top_influencers:
        node_color = "red"
        net.add_node(
            influencer,
            label=label_text,
            title=title_text,
            color=node_color,
            size=node_size,
            shape="circularImage",
            image=f"{influencer}.jpg"
        )
    else:
        node_color = "gray"
        net.add_node(
            influencer,
            label=label_text,
            title=title_text,
            color=node_color,
            size=node_size
        )

# Group Audience Nodes by Behavior (using sampled_pairs)
audience_behavior = sampled_pairs.groupby('username')['influencerusername'].apply(lambda x: frozenset(x)).reset_index(name='influencers_set')
audience_groups = audience_behavior.groupby('influencers_set')['username'].apply(list).reset_index(name='audience_list')
audience_groups['group_id'] = audience_groups.index.map(lambda i: f"Group_{i}")
# Convert frozenset to string for display
audience_groups['influencers_set'] = audience_groups['influencers_set'].apply(lambda x: ", ".join(sorted(list(x))) if isinstance(x, (set, frozenset)) else x)

for idx, row in audience_groups.iterrows():
    group_id = row['group_id']
    influencers_set = row['influencers_set']
    audience_list = row['audience_list']
    group_size = len(audience_list)
    node_size = 5 + 10 * math.log(group_size + 1)
    title_text = f"Audience Group (Size: {group_size})\nInfluencers: {influencers_set}"
    net.add_node(
        group_id,
        label=str(group_size),
        title=title_text,
        color="lightblue",
        size=node_size
    )

# Add edges from audience groups to influencers
base_edge_thickness = 1
scaling_edge = 1
for idx, row in audience_groups.iterrows():
    group_id = row['group_id']
    group_size = len(row['audience_list'])
    edge_thickness = base_edge_thickness + scaling_edge * math.log(group_size + 1)
    influencers_str = row['influencers_set']
    influencers_set = set(influencers_str.split(", ")) if influencers_str != "" else set()
    for influencer in influencers_set:
        if influencer in top_influencers:
            edge_color = "rgba(255,0,0,0.5)"
        else:
            edge_color = "rgba(128,128,128,0.5)"
        net.add_edge(group_id, influencer, color=edge_color, width=edge_thickness)

# Instead of net.show(), we generate the HTML and embed it in the Streamlit app.
graph_html = net.generate_html(notebook=True)


st.header("Interactive PyVis Graph")
components.html(graph_html, height=800, scrolling=False)

