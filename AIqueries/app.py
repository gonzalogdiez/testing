import streamlit as st
import pandas as pd
from openai import OpenAI

st.set_page_config(page_title="Excel Q&A", layout="centered")
st.title("Ask Questions About Our Data")

st.markdown(
    """
    <div style="font-size:0.9em;color:#555;">
    **Please note:** This application is powered by a curated sample of 100 credit-union records for demonstration and testing purposes only. Results may differ when connected to the full dataset.
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Load API key from secrets
api_key = st.secrets["OPENAI_API_KEY"]
client  = OpenAI(api_key=api_key)

# 2. Load the Excel file from disk
df = pd.read_excel("AIqueries/data/data.xlsx", engine="openpyxl")
st.write("Preview of your data:")
st.dataframe(df.head())

# 3. Ask question
question = st.text_input("Chat with our data")
if not question:
    st.stop()

# 4. System prompt
system_prompt = """
You are a data analyst. The user has a pandas DataFrame 'df' loaded from an Excel file.
Answer the user's question in clear, natural language, and include a brief (≤5-row) table or summary of the relevant data.
Always provide the source of what you used to answer. 
If the data cannot answer the question, say so clearly. Do not return any code.
"""

# 5. Build prompt and call OpenAI
data_csv   = df.to_csv(index=False)
user_prompt = f"DataFrame CSV:\n{data_csv}\n\nUser question: {question}"

with st.spinner("Generating answer..."):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   # or gpt-4-turbo if you have access
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0
    )
    answer = response.choices[0].message.content

# 6. Show answer
st.subheader("Answer")
st.write(answer)
