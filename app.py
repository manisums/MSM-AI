import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

# =========================
# CONFIG
# =========================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

KPI_FILE_PATH = r"C:\Users\WH327AH\OneDrive - EY\Desktop\MSM AI\Excel.xlsx"

llm_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

st.set_page_config(page_title="MSM ESG Gap Engine", layout="wide")
st.title("MSM ESG KPI Gap Analysis Engine")

# =========================
# EMBEDDINGS
# =========================
def embed_1536(text):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=[text]
    ).data[0].embedding

def embed_3072(text):
    return llm_client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    ).data[0].embedding

# =========================
# COLLECTION CONFIG
# =========================
COLLECTION_CONFIG = {
    "esg_regulations": 1536,
    "esrs_e1": 1536,
    "client_bor": 3072,
}

# =========================
# RETRIEVE CONTEXT
# =========================
def retrieve_regulatory_context(query):

    texts = []

    for col, dim in COLLECTION_CONFIG.items():

        qvec = embed_3072(query) if dim == 3072 else embed_1536(query)

        results = qdrant_client.query_points(
            collection_name=col,
            query=qvec,
            limit=6
        )

        for point in results.points:
            txt = (
                point.payload.get("text")
                or point.payload.get("document")
                or ""
            )
            if txt:
                texts.append(f"[{col}] {txt}")

    return "\n\n".join(texts)

# =========================
# GAP ANALYSIS
# =========================
SYSTEM_PROMPT = """
You are an ESG regulatory and carbon accounting expert.
Use only regulatory context to determine feasibility.
Return JSON only.
"""

def run_gap_analysis(kpi_id, kpi_name, reg_text, schema_text):

    USER_PROMPT = f"""
KPI ID:
{kpi_id}

KPI NAME:
{kpi_name}

REGULATORY CONTEXT:
{reg_text}

CLIENT SCHEMA:
{schema_text}

TASK:
1. Identify required fields.
2. Match with schema.
3. Decide:
   FULLY_CALCULABLE / PARTIALLY_CALCULABLE / NOT_CALCULABLE
4. List missing fields.
5. Give reasoning.

Return JSON:

{{
  "feasibility": "",
  "required_fields": [],
  "available_fields": [],
  "missing_fields": [],
  "reasoning": ""
}}
"""

    resp = llm_client.chat.completions.create(
        model="gpt-5",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return resp.choices[0].message.content

# =========================
# READ KPI FILE AUTOMATICALLY
# =========================
if not os.path.exists(KPI_FILE_PATH):
    st.error("KPI Excel file not found at given path.")
    st.stop()

kpi_df = pd.read_excel(KPI_FILE_PATH)

st.success(f"KPI file loaded from: {KPI_FILE_PATH}")

# =========================
# KPI DROPDOWN
# =========================
selected_kpi = st.selectbox(
    "Select KPI for Gap Analysis",
    kpi_df["KPI Name"]
)

selected_row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]
selected_kpi_id = selected_row["KPI ID"]

st.info(f"Selected KPI ID: {selected_kpi_id}")

# =========================
# SHOW UPLOAD OPTIONS ONLY AFTER KPI SELECTED
# =========================
st.subheader("Upload Supporting Files")

schema_file = st.file_uploader("Upload Schema Excel", type=["xlsx"])
data_file = st.file_uploader("Upload Sample Data Excel (optional)", type=["xlsx"])

# =========================
# RUN ANALYSIS
# =========================
if schema_file and st.button("Run Gap Analysis"):

    schema_df = pd.read_excel(schema_file)

    schema_text = "\n".join(
        schema_df.astype(str).agg(" | ".join, axis=1)
    )

    with st.spinner("Retrieving regulatory context..."):
        reg_text = retrieve_regulatory_context(selected_kpi)

    with st.spinner("Running AI Analysis..."):
        raw_output = run_gap_analysis(
            selected_kpi_id,
            selected_kpi,
            reg_text,
            schema_text
        )

    try:
        result = json.loads(raw_output)
    except:
        st.error("Model did not return valid JSON")
        st.write(raw_output)
        st.stop()

    # Fill columns
    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Feasiblity"
    ] = result["feasibility"]

    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Column names which are available"
    ] = ", ".join(result["available_fields"])

    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Column name which are required more"
    ] = ", ".join(result["missing_fields"])

    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Reason"
    ] = result["reasoning"]

    st.success("Gap Analysis Completed")

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df)

    # Save updated Excel back
    updated_path = r"C:\Users\WH327AH\OneDrive - EY\Desktop\MSM AI\Excel_Updated.xlsx"
    kpi_df.to_excel(updated_path, index=False)

    st.success(f"Updated file saved to: {updated_path}")
