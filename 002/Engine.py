# streamlit_app.py

import os
import streamlit as st
import mysql.connector
from mysql.connector import Error

# ─── MySQL Connection ─────────────────────────────────────────────────────────
def get_mysql_connection():
    # Try reading from Streamlit secrets.toml first
    cfg = st.secrets.get("mysql", {})

    host     = cfg.get("host")     or os.getenv("MYSQL_HOST", os.getenv("DB_HOST", "db"))
    port     = cfg.get("port")     or int(os.getenv("MYSQL_PORT", 3306))
    user     = cfg.get("user")     or os.getenv("MYSQL_USER", os.getenv("DB_USER"))
    password = cfg.get("password") or os.getenv("MYSQL_PASSWORD", os.getenv("DB_PASS"))
    database = cfg.get("database") or os.getenv("MYSQL_DATABASE", "instagram_db")

    try:
        return mysql.connector.connect(
            host     = host,
            port     = port,
            user     = user,
            password = password,
            database = database
        )
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# ─── Load hashtags with frequency (top N) ────────────────────────────────────
def load_hashtag_frequencies(limit: int = 10):
    """
    Returns a list of dicts: {id, hashtag, freq}
    Top {limit} hashtags by frequency.
    """
    conn = get_mysql_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(f"""
            SELECT
                h.id,
                h.hashtag,
                COUNT(uh.user_id) AS freq
            FROM instagram_hashtags AS h
            LEFT JOIN instagram_user_hashtags AS uh
                ON h.id = uh.hashtag_id
            GROUP BY h.id, h.hashtag
            ORDER BY freq DESC
            LIMIT %s
        """, (limit,))
        return cursor.fetchall()
    except Error as e:
        st.error(f"Error querying hashtags: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# ─── Load users for selected hashtags (up to limit) ─────────────────────────
def load_users_for_hashtags(hashtag_ids: list[int], limit: int = 50):
    """
    Given a list of hashtag_id, returns
    a list of dicts: {user_id, username}
    of users who used any of those hashtags.
    """
    if not hashtag_ids:
        return []

    conn = get_mysql_connection()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        placeholders = ",".join(["%s"] * len(hashtag_ids))
        sql = f"""
            SELECT DISTINCT
                u.id       AS user_id,
                u.username
            FROM instagram_users AS u
            JOIN instagram_user_hashtags AS uh
                ON u.id = uh.user_id
            WHERE uh.hashtag_id IN ({placeholders})
            LIMIT %s
        """
        params = hashtag_ids + [limit]
        cursor.execute(sql, params)
        return cursor.fetchall()
    except Error as e:
        st.error(f"Error querying users: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# ─── Streamlit App UI ────────────────────────────────────────────────────────
st.title("From Hashtags to Instagram Users")

# Sidebar controls for limits
top_n = st.sidebar.number_input(
    "Number of hashtags to load", min_value=1, max_value=100, value=10, step=1
)
user_limit = st.sidebar.number_input(
    "Maximum users to display", min_value=1, max_value=500, value=50, step=10
)

# Step 1: Display hashtag list
st.header("Step 1: Select Hashtags")
hashtags = load_hashtag_frequencies(limit=top_n)

if not hashtags:
    st.warning("Unable to load hashtags.")
    st.stop()

selected_tags = st.multiselect(
    "Select one or more hashtags:",
    options=hashtags,
    format_func=lambda x: f"{x['hashtag']} ({x['freq']})"
)

# Step 2: Fetch users when button clicked
if st.button("Get Users"):
    hashtag_ids = [h['id'] for h in selected_tags]
    users = load_users_for_hashtags(hashtag_ids, limit=user_limit)

    if users:
        st.header("Users Who Used Those Hashtags")
        st.table(users)
    else:
        st.info("No users found for the selected hashtags.")
