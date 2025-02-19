import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import gdown

# ================================
# Data Loading & Preprocessing (Cached)
# ================================
@st.cache_data
def load_data():
    os.makedirs("raw", exist_ok=True)
    im_url = "https://drive.google.com/uc?export=download&id=1F9Wfb-six5W4ZvkwYRVOtdTzZ8CVkYUc"
    i_url  = "https://drive.google.com/uc?export=download&id=1Fbk6H6jqO6b3VHQ1SjLdWLEQo47JQvPu"
    ml_url = "https://drive.google.com/uc?export=download&id=1FhrqVbUc3CQeKHqbbBq2xp6o4024jm7u"

    # Download files if they don't exist locally
    if not os.path.exists("raw/influencer_media.csv"):
        gdown.download(im_url, "raw/influencer_media.csv", quiet=False)
    if not os.path.exists("raw/influencers.csv"):
        gdown.download(i_url, "raw/influencers.csv", quiet=False)
    if not os.path.exists("raw/media_likers.csv"):
        gdown.download(ml_url, "raw/media_likers.csv", quiet=False)

    # Load CSVs
    im = pd.read_csv("raw/influencer_media.csv")
    i = pd.read_csv("raw/influencers.csv")
    ml = pd.read_csv("raw/media_likers.csv")

    # Subset the DataFrames to only include necessary columns
    ml = ml[['pk', 'username', 'media_id']]
    im = im[['pk', 'id', 'media_type', 'code', 'user', 'comment_count', 'has_liked',
             'like_count', 'top_likers', 'reshare_count', 'usertags', 'play_count',
             'user_id', 'fb_like_count', 'view_count']]
    i = i[['pk', 'username', 'full_name', 'media_count', 'follower_count', 'following_count', 'account_type']]

    # Delete CSV files to free disk space
    os.remove("raw/influencer_media.csv")
    os.remove("raw/influencers.csv")
    os.remove("raw/media_likers.csv")

    # Print some basic info (to console)
    max_follower_count = i['follower_count'].max()
    print(f"Maximum follower count: {max_follower_count}")
    max_follower_row = i.loc[i['follower_count'].idxmax()]
    print(f"Username with the maximum follower count: {max_follower_row['username']}")
    total_follower_count = i['follower_count'].sum()
    print(f"Total follower count: {total_follower_count}")

    # -------------------------------
    # Utility: Add influencerusername to im
    # -------------------------------
    def add_influencer_username(im, i):
        im_with_username = im.merge(i[['pk', 'username']], 
                                    left_on='user_id', right_on='pk', how='left')
        im_with_username.rename(columns={'username': 'influencerusername'}, inplace=True)
        return im_with_username
    
    im_with_username = add_influencer_username(im, i)
    ml_with_username = ml.merge(im_with_username[['pk_x', 'user_id', 'influencerusername']],
                                left_on='media_id', right_on='pk_x', how='left')
    return ml, im, i, ml_with_username

@st.cache_data
def compute_sampled_pairs(ml_with_username):
    def sample_connections(df):
        n = len(df)
        sample_size = max(int(0.03 * n), 5)
        sample_size = min(sample_size, n)
        return df.sample(n=sample_size, random_state=42)
    sampled_pairs = ml_with_username.groupby('influencerusername').apply(sample_connections).reset_index(drop=True)
    return sampled_pairs

# Load data only once
ml, im, i, ml_with_username = load_data()
sampled_pairs = compute_sampled_pairs(ml_with_username)

# -------------------------------
# (Optional) Plot histogram of unique influencer interactions per user
# -------------------------------
def count_user_interactions_per_influencer(ml_with_username):
    user_interaction_counts = (
        ml_with_username.groupby('username')['influencerusername']
        .nunique()
        .reset_index()
        .rename(columns={'influencerusername': 'unique_influencers_count'})
    )
    return user_interaction_counts

s_hist = count_user_interactions_per_influencer(ml_with_username)
plt.figure(figsize=(10, 6))
plt.hist(s_hist['unique_influencers_count'], bins=range(1, 91), edgecolor='black', alpha=0.7)
plt.title('Distribution of Unique Influencer Interactions per User', fontsize=16)
plt.xlabel('Number of Unique Influencers', fontsize=14)
plt.ylabel('Number of Users', fontsize=14)
plt.xticks(range(1, 91, 5))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# -------------------------------
# Analysis Function: run_analysis()
# -------------------------------
def run_analysis(ml_with_username, im_with_username, core_threshold=2):
    results = {}
    
    # Q1: Total Unique Audience
    total_unique_audience = ml_with_username['username'].nunique()
    results["total_unique_audience"] = total_unique_audience

    # Q2 & Q3: Define CORE users (>= core_threshold distinct influencers)
    user_influencer_counts = ml_with_username.groupby('username')['influencerusername'].nunique()
    core_users = user_influencer_counts[user_influencer_counts >= core_threshold]
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
