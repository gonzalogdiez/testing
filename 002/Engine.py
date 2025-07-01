# streamlit_app.py

import os
import streamlit as st
import mysql.connector
from mysql.connector import Error

# ─── Conexión a MySQL ────────────────────────────────────────────────────────
def get_mysql_connection():
    try:
        return mysql.connector.connect(
            host     = os.getenv("MYSQL_HOST", os.getenv("DB_HOST", "db")),
            port     = int(os.getenv("MYSQL_PORT", 3306)),
            user     = os.getenv("MYSQL_USER", os.getenv("DB_USER")),
            password = os.getenv("MYSQL_PASSWORD", os.getenv("DB_PASS")),
            database = os.getenv("MYSQL_DATABASE", "instagram_db")
        )
    except Error as e:
        st.error(f"Error conectando a MySQL: {e}")
        return None

# ─── Carga de hashtags con frecuencia ───────────────────────────────────────
def load_hashtag_frequencies():
    """
    Devuelve una lista de dicts: {id, hashtag, freq}
    donde freq es la cantidad de veces
    ese hashtag aparece en instagram_user_hashtags.
    """
    conn = get_mysql_connection()
    if not conn:
        return []
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
              h.id,
              h.hashtag,
              COUNT(uh.user_id) AS freq
            FROM instagram_hashtags AS h
            LEFT JOIN instagram_user_hashtags AS uh
              ON h.id = uh.hashtag_id
            GROUP BY h.id, h.hashtag
            ORDER BY freq DESC
        """)
        return cursor.fetchall()
    except Error as e:
        st.error(f"Error al consultar hashtags: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# ─── Carga de usuarios para los hashtags seleccionados ──────────────────────
def load_users_for_hashtags(hashtag_ids: list[int]):
    """
    Recibe una lista de hashtag_id y devuelve
    una lista de dicts: {user_id, username}
    con los usuarios que mencionaron alguno de esos hashtags.
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
              u.id    AS user_id,
              u.username
            FROM instagram_users AS u
            JOIN instagram_user_hashtags AS uh
              ON u.id = uh.user_id
            WHERE uh.hashtag_id IN ({placeholders})
        """
        cursor.execute(sql, hashtag_ids)
        return cursor.fetchall()
    except Error as e:
        st.error(f"Error al consultar usuarios: {e}")
        return []
    finally:
        cursor.close()
        conn.close()

# ─── Interfaz Streamlit ────────────────────────────────────────────────────
st.title("De Hashtags a Usuarios en Instagram")

# Paso 1: mostrar lista de hashtags con frecuencia
st.header("Paso 1: Selecciona hashtags")
hashtags = load_hashtag_frequencies()

if not hashtags:
    st.warning("No se pudieron cargar los hashtags.")
    st.stop()

selected_tags = st.multiselect(
    "Elige uno o más hashtags:",
    options=hashtags,
    format_func=lambda x: f"{x['hashtag']} ({x['freq']})"
)

# Botón para disparar la carga de usuarios
if st.button("Obtener usuarios"):
    hashtag_ids = [h['id'] for h in selected_tags]
    users = load_users_for_hashtags(hashtag_ids)

    if users:
        st.header("Usuarios que usaron esos hashtags")
        st.table(users)
    else:
        st.info("No se encontraron usuarios para los hashtags seleccionados.")
