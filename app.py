import streamlit as st
import pandas as pd
import json
from io import BytesIO
import os

from openai import OpenAI
from qdrant_client import QdrantClient

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="MSM ESG Gap Engine", layout="wide")
st.title("MSM ESG KPI Gap Analysis Engine")

# =====================================================
# SECRETS
# =====================================================
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

# =====================================================
# CLIENTS
# =====================================================
openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# =====================================================
# COLLECTIONS (ALL 1536)
# =====================================================
COLLECTIONS = [
    "client_bor",
    "esg_regulations",
    "esrs_e1",
    "msm_dataverse_model"
]

# =====================================================
# EMBEDDING
# =====================================================
def embed(text: str):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =====================================================
# RETRIEVE CONTEXT + AUDIT TRACE
# =====================================================
def retrieve_context_with_audit(query):
    context = []
    audit = []

    vector = embed(query)

    for collection in COLLECTIONS:
        results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=4
        )

        for p in results.points:
            text = p.payload.get("content")
            if not text:
                continue

            context.append(text)

            audit.append({
                "collection": collection,
                "doc_type": p.payload.get("doc_type"),
                "source": p.payload.get("source"),
                "page": p.payload.get("page")
            })

    return "\n\n".join(context), audit

# =====================================================
# GAP ANALYSIS PROMPT
# =====================================================
SYSTEM_PROMPT = """
You are an ESG regulatory expert.
Use only provided context.
Return STRICT JSON.
"""

def run_gap_analysis(kpi_id, kpi_name, context, schema_cols, data_cols):

    USER_PROMPT = f"""
KPI ID: {kpi_id}
KPI NAME: {kpi_name}

REGULATORY CONTEXT:
{context}

SCHEMA COLUMNS:
{schema_cols}

DATA COLUMNS:
{data_cols}

Return JSON:
{{
  "feasibility": "FULLY_CALCULABLE | PARTIALLY_CALCULABLE | NOT_CALCULABLE",
  "required_fields": [],
  "available_fields": [],
  "missing_fields": [],
  "reasoning": ""
}}
"""

    res = openai_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return json.loads(res.choices[0].message.content)

# =====================================================
# AUDIT SCORE
# =====================================================
def calculate_audit_score(result, audit):
    score = 0

    if result["feasibility"] == "FULLY_CALCULABLE":
        score += 40
    elif result["feasibility"] == "PARTIALLY_CALCULABLE":
        score += 20

    score += min(len(result["available_fields"]) * 5, 30)
    score += min(len(audit) * 5, 30)

    return min(score, 100)

# =====================================================
# FILE PICKERS (FROM REPO)
# =====================================================
BASE_PATH = "data"
excel_files = [f for f in os.listdir(BASE_PATH) if f.endswith(".xlsx")]

st.sidebar.header("Input Files")

kpi_file = st.sidebar.selectbox("KPI Master", excel_files)
schema_file = st.sidebar.selectbox("Schema File", excel_files)
data_file = st.sidebar.selectbox("Sample Data", ["None"] + excel_files)

# =====================================================
# LOAD FILES
# =====================================================
kpi_df = pd.read_excel(os.path.join(BASE_PATH, kpi_file))
schema_df = pd.read_excel(os.path.join(BASE_PATH, schema_file))
data_df = pd.read_excel(os.path.join(BASE_PATH, data_file)) if data_file != "None" else None

# =====================================================
# KPI SELECTION
# =====================================================
selected_kpi = st.selectbox("Select KPI", kpi_df["KPI Name"])
row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]

# =====================================================
# RUN
# =====================================================
if st.button("Run Gap Analysis"):

    with st.spinner("Retrieving regulatory context..."):
        context, audit = retrieve_context_with_audit(selected_kpi)

    schema_cols = schema_df["column_name"].astype(str).tolist()
    data_cols = data_df.columns.tolist() if data_df is not None else []

    with st.spinner("Running analysis..."):
        result = run_gap_analysis(
            row["KPI ID"],
            selected_kpi,
            context,
            schema_cols,
            data_cols
        )

    audit_score = calculate_audit_score(result, audit)

    # Update KPI table
    kpi_df.loc[row.name, "Feasibility"] = result["feasibility"]
    kpi_df.loc[row.name, "Available Columns"] = ", ".join(result["available_fields"])
    kpi_df.loc[row.name, "Missing Columns"] = ", ".join(result["missing_fields"])
    kpi_df.loc[row.name, "Audit Score"] = audit_score
    kpi_df.loc[row.name, "Reason"] = result["reasoning"]

    st.success("Analysis complete")

    st.subheader("Audit Trace")
    st.dataframe(pd.DataFrame(audit))

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    buffer = BytesIO()
    kpi_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "Download Updated KPI File",
        buffer,
        "kpi_gap_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
