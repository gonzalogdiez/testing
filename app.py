import math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import gdown
import os

# ================================
# Data Loading & Preprocessing
# ================================

# Define URLs (using the file IDs from your shared links)
im_url = "https://drive.google.com/uc?export=download&id=1F9Wfb-six5W4ZvkwYRVOtdTzZ8CVkYUc"
i_url  = "https://drive.google.com/uc?export=download&id=1Fbk6H6jqO6b3VHQ1SjLdWLEQo47JQvPu"
ml_url = "https://drive.google.com/uc?export=download&id=1FhrqVbUc3CQeKHqbbBq2xp6o4024jm7u"

# Optionally, check if the file exists locally before downloading.
if not os.path.exists("raw/influencer_media.csv"):
    gdown.download(im_url, "raw/influencer_media.csv", quiet=False)
if not os.path.exists("raw/influencers.csv"):
    gdown.download(i_url, "raw/influencers.csv", quiet=False)
if not os.path.exists("raw/media_likers.csv"):
    gdown.download(ml_url, "raw/media_likers.csv", quiet=False)

# Now load the CSVs from the local paths
im = pd.read_csv("raw/influencer_media.csv")
i = pd.read_csv("raw/influencers.csv")
ml = pd.read_csv("raw/media_likers.csv")


# Print some basic info (to console)
max_follower_count = i['follower_count'].max()
print(f"Maximum follower count: {max_follower_count}")
max_follower_row = i.loc[i['follower_count'].idxmax()]
username_max_follower = max_follower_row['username']
print(f"Username with the maximum follower count: {username_max_follower}")
total_follower_count = i['follower_count'].sum()
print(f"Total follower count: {total_follower_count}")

# Function to add influencerusername to im
def add_influencer_username(im, i):
    im_with_username = im.merge(i[['pk', 'username']], left_on='user_id', right_on='pk', how='left')
    im_with_username.rename(columns={'username': 'influencerusername'}, inplace=True)
    return im_with_username

im_with_username = add_influencer_username(im, i)
ml_with_username = ml.merge(im_with_username[['pk_x','user_id', 'influencerusername']], 
                            left_on='media_id', right_on='pk_x', how='left')

# ================================
# Utility Function for Histogram (Optional)
# ================================
def count_user_interactions_per_influencer(ml_with_username):
    user_interaction_counts = (
        ml_with_username.groupby('username')['influencerusername']
        .nunique()
        .reset_index()
        .rename(columns={'username': 'username', 'influencerusername': 'unique_influencers_count'})
    )
    return user_interaction_counts

# Optional: Plot a histogram (this will open in an external window)
s_hist = count_user_interactions_per_influencer(ml_with_username)
plt.figure(figsize=(10, 6))
plt.hist(s_hist['unique_influencers_count'], bins=range(1, 91), edgecolor='black', alpha=0.7)
plt.title('Distribution of Unique Influencer Interactions per User', fontsize=16)
plt.xlabel('Number of Unique Influencers', fontsize=14)
plt.ylabel('Number of Users', fontsize=14)
plt.xticks(range(1, 91, 5))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ================================
# Create Sampled Pairs for Audience Grouping
# ================================
def sample_connections(df):
    """
    For a given influencer's connections, sample 3% (rounded down) but at least 5 rows.
    """
    n = len(df)
    sample_size = max(int(0.03 * n), 5)
    sample_size = min(sample_size, n)
    return df.sample(n=sample_size, random_state=42)

sampled_pairs = ml_with_username.groupby('influencerusername').apply(sample_connections).reset_index(drop=True)

# ================================
# Analysis Function: run_analysis()
# ================================
def run_analysis(ml_with_username, im_with_username):
    results = {}
    
    # Q1: Total Unique Audience
    total_unique_audience = ml_with_username['username'].nunique()
    results["total_unique_audience"] = total_unique_audience

    # Q2 & Q3: Define CORE users (>=2 distinct influencers)
    user_influencer_counts = ml_with_username.groupby('username')['influencerusername'].nunique()
    core_users = user_influencer_counts[user_influencer_counts >= 2]
    total_core_users = core_users.count()
    results["total_core_users"] = total_core_users

    # Q4: Core audience as % of total audience
    core_percentage = (total_core_users / total_unique_audience) * 100
    results["core_percentage"] = core_percentage

    # Q5 & Q6: Build dictionary of core users per influencer
    influencer_to_core = {}
    core_user_set = set(core_users.index)
    for influencer, group in ml_with_username.groupby('influencerusername'):
        users = set(group['username'])
        influencer_to_core[influencer] = users.intersection(core_user_set)
    results["influencer_to_core"] = influencer_to_core

    # Greedy algorithm for coverage:
    def influencers_to_cover_core(target_percentage, inf_to_core):
        target = int((target_percentage / 100.0) * total_core_users)
        selected = set()
        chosen_influencers = []
        remaining = inf_to_core.copy()
        while len(selected) < target and remaining:
            best = None
            best_new = 0
            for influencer, audience in remaining.items():
                new_users = audience - selected
                if len(new_users) > best_new:
                    best_new = len(new_users)
                    best = influencer
            if best is None:
                break
            chosen_influencers.append(best)
            selected |= influencer_to_core[best]
            del remaining[best]
        return chosen_influencers, len(selected)
    
    selected_50, covered_50 = influencers_to_cover_core(50, influencer_to_core)
    selected_100, covered_100 = influencers_to_cover_core(100, influencer_to_core)
    results["selected_50_count"] = len(selected_50)
    results["covered_50"] = covered_50
    results["selected_100_count"] = len(selected_100)
    results["covered_100"] = covered_100

    # Q7: Estimate cost using view counts
    valid = im_with_username[(im_with_username['view_count'].notnull()) & (im_with_username['view_count'] > 0)].copy()
    valid['ratio'] = valid['view_count'] / valid['like_count']
    median_ratio = valid['ratio'].median()
    results["median_ratio"] = median_ratio

    im_with_username['view_count_est'] = im_with_username['view_count']
    im_with_username.loc[im_with_username['view_count_est'].isnull(), 'view_count_est'] = \
        im_with_username.loc[im_with_username['view_count_est'].isnull(), 'like_count'] * median_ratio

    influencer_cost = {}
    influencer_summary = []
    for inf in influencer_to_core.keys():
        subset = im_with_username[im_with_username['influencerusername'] == inf]
        num_posts = len(subset)
        mean_views = subset['view_count_est'].mean() if num_posts > 0 else float('nan')
        median_views = subset['view_count_est'].median() if num_posts > 0 else float('nan')
        cost_median = median_views * 0.10 if not pd.isnull(median_views) else float('nan')
        influencer_cost[inf] = cost_median
        reach = ml_with_username[ml_with_username['influencerusername'] == inf]['username'].nunique()
        influencer_summary.append({
            'influencerusername': inf,
            'num_posts': num_posts,
            'mean_est_view_count': mean_views,
            'median_est_view_count': median_views,
            'cost': cost_median,
            'user_reach': reach
        })
    results["influencer_cost"] = influencer_cost
    df_influencers = pd.DataFrame(influencer_summary).sort_values(by='user_reach', ascending=False)
    results["df_influencers"] = df_influencers

    # Greedy algorithm for cheapest mix using estimated cost.
    def greedy_cheapest_influencers(target_percentage, inf_to_core, inf_cost):
        target = int((target_percentage / 100.0) * total_core_users)
        selected = set()
        chosen = []
        remaining = inf_to_core.copy()
        while len(selected) < target and remaining:
            best = None
            best_ratio = float('inf')
            for influencer, audience in remaining.items():
                new_users = audience - selected
                if len(new_users) > 0:
                    ratio = inf_cost[influencer] / len(new_users)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best = influencer
            if best is None:
                break
            chosen.append(best)
            selected |= influencer_to_core[best]
            del remaining[best]
        total_cost = sum(inf_cost[inf] for inf in chosen)
        return chosen, len(selected), total_cost
    
    cheapest_50, covered_50_cost, cost_50 = greedy_cheapest_influencers(50, influencer_to_core, influencer_cost)
    cheapest_100, covered_100_cost, cost_100 = greedy_cheapest_influencers(100, influencer_to_core, influencer_cost)
    results["cheapest_50_count"] = len(cheapest_50)
    results["cheapest_50_cost"] = cost_50
    results["cheapest_100_count"] = len(cheapest_100)
    results["cheapest_100_cost"] = cost_100

    # Q8: Marginal gains
    sorted_influencers = sorted(influencer_to_core.items(), key=lambda x: len(x[1]), reverse=True)
    marginal_gains = []
    current_union = set()
    for influencer, audience in sorted_influencers:
        new_gain = len(audience - current_union)
        marginal_gains.append((influencer, new_gain))
        current_union |= audience
    results["marginal_gains"] = marginal_gains

    # Q10: Frequency analysis for >=3 engagements
    engagement_counts = ml_with_username.groupby(['username', 'influencerusername']).size().reset_index(name='count')
    freq_df = engagement_counts[engagement_counts['count'] >= 3]
    total_unique_freq = freq_df['username'].nunique()
    results["total_unique_freq"] = total_unique_freq
    user_freq_influencer = freq_df.groupby('username')['influencerusername'].nunique()
    core_freq = user_freq_influencer[user_freq_influencer >= 2]
    total_core_freq = core_freq.count()
    results["total_core_freq"] = total_core_freq
    core_freq_percentage = (total_core_freq / total_unique_freq) * 100 if total_unique_freq > 0 else 0
    results["core_freq_percentage"] = core_freq_percentage

    # Q11: Natural audience clusters (by exact set of influencers engaged, using the sample)
    audience_behavior = sampled_pairs.groupby('username')['influencerusername'].apply(lambda x: frozenset(x)).reset_index(name='influencers_set')
    audience_groups = audience_behavior.groupby('influencers_set')['username'].apply(list).reset_index(name='audience_list')
    audience_groups['group_id'] = audience_groups.index.map(lambda i: f"Group_{i}")
    results["audience_groups"] = audience_groups
    results["num_clusters"] = audience_groups.shape[0]

    return results

# Run the analysis and store output in 'results'
results = run_analysis(ml_with_username, im_with_username)

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

# ================================
# Streamlit Dashboard
# ================================
st.title("Influencer Analysis Dashboard")

st.header("Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Unique Audience", results["total_unique_audience"])
col2.metric("Total CORE Users", results["total_core_users"])
col3.metric("Core Audience %", f"{results['core_percentage']:.2f}%")

st.header("Coverage Metrics")
st.write(f"To cover 50% of the core audience: **{results['selected_50_count']}** influencers (covering **{results['covered_50']}** core users).")
st.write(f"To cover 100% of the core audience: **{results['selected_100_count']}** influencers (covering **{results['covered_100']}** core users).")

st.header("Cost Estimation")
st.write(f"Median ratio (view_count/like_count): **{results['median_ratio']:.2%}**")
st.write(f"Cheapest mix to cover 50% of core audience: **{results['cheapest_50_count']}** influencers with total cost **${results['cheapest_50_cost']:.2f}**.")
st.write(f"Cheapest mix to cover 100% of core audience: **{results['cheapest_100_count']}** influencers with total cost **${results['cheapest_100_cost']:.2f}**.")

st.header("Marginal Gains")
mg_df = pd.DataFrame(results["marginal_gains"], columns=["Influencer", "New Core Users Added"])
st.dataframe(mg_df)

st.header("Influencer Summary")
st.dataframe(results["df_influencers"])

st.header("Frequency Analysis (>=3 engagements)")
st.write(f"Total unique audience with frequency >= 3 per influencer: **{results['total_unique_freq']}**")
st.write(f"CORE unique users (freq>=3 with at least 2 influencers): **{results['total_core_freq']}**")
st.write(f"Core audience (freq>=3) is **{results['core_freq_percentage']:.2f}%** of the total audience with freq>=3.")

st.header("Natural Audience Clusters")
st.write(f"Number of natural audience clusters: **{results['num_clusters']}**")
st.dataframe(results["audience_groups"])

st.header("Interactive PyVis Graph")
components.html(graph_html, height=800, scrolling=False)
