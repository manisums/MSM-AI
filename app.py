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
st.set_page_config(page_title="MSM ESG KPI Gap Analysis Engine", layout="wide")
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
# COLLECTIONS (ALL 1536-dim)
# =====================================================
COLLECTIONS = [
    "client_bor",
    "esg_regulations",
    "esrs_e1",
    "msm_dataverse_model"
]

# =====================================================
# EMBEDDING (SINGLE MODEL)
# =====================================================
def embed(text: str):
    return openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =====================================================
# RETRIEVE CONTEXT + AUDIT TRACE
# =====================================================
def retrieve_context_with_audit(query: str):
    context_chunks = []
    audit_rows = []

    vector = embed(query)

    for collection in COLLECTIONS:
        results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=4
        )

        for p in results.points:
            content = p.payload.get("content")
            if not content:
                continue

            context_chunks.append(content)

            audit_rows.append({
                "collection": collection,
                "doc_type": p.payload.get("doc_type"),
                "source": p.payload.get("source"),
                "page": p.payload.get("page")
            })

    return "\n\n".join(context_chunks), audit_rows

# =====================================================
# GAP ANALYSIS (LLM)
# =====================================================
SYSTEM_PROMPT = """
You are an ESG regulatory and sustainability reporting expert.
Use ONLY the provided regulatory context.
Return STRICT valid JSON only.
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

TASK:
1. Identify required fields from context
2. Match with schema & data
3. Decide feasibility
4. List available and missing fields

Return JSON:
{{
  "feasibility": "FULLY_CALCULABLE | PARTIALLY_CALCULABLE | NOT_CALCULABLE",
  "required_fields": [],
  "available_fields": [],
  "missing_fields": [],
  "reasoning": ""
}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return json.loads(response.choices[0].message.content)

# =====================================================
# AUDIT SCORE (DETERMINISTIC)
# =====================================================
def calculate_audit_score(result, audit_rows):
    score = 0

    if result["feasibility"] == "FULLY_CALCULABLE":
        score += 40
    elif result["feasibility"] == "PARTIALLY_CALCULABLE":
        score += 20

    score += min(len(result["available_fields"]) * 5, 30)
    score += min(len(audit_rows) * 5, 30)

    return min(score, 100)

# =====================================================
# FILE PICKERS (DROPDOWN FROM REPO)
# =====================================================
BASE_PATH = "data"

if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)

excel_files = [
    f for f in os.listdir(BASE_PATH)
    if f.lower().endswith(".xlsx")
]

if not excel_files:
    st.error("No Excel files found in /data folder. Please add files to the repo.")
    st.stop()

st.sidebar.header("Input Files")

kpi_file = st.sidebar.selectbox("KPI Master File", excel_files)
schema_file = st.sidebar.selectbox("Schema File", excel_files)
data_file = st.sidebar.selectbox("Sample Data (optional)", ["None"] + excel_files)

# =====================================================
# LOAD FILES
# =====================================================
kpi_df = pd.read_excel(os.path.join(BASE_PATH, kpi_file))
schema_df = pd.read_excel(os.path.join(BASE_PATH, schema_file))

data_df = None
if data_file != "None":
    data_df = pd.read_excel(os.path.join(BASE_PATH, data_file))

# =====================================================
# KPI SELECTION
# =====================================================
selected_kpi = st.selectbox("Select KPI", kpi_df["KPI Name"])
kpi_row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]

# =====================================================
# RUN ANALYSIS
# =====================================================
if st.button("Run Gap Analysis"):

    with st.spinner("Retrieving regulatory context..."):
        context, audit_rows = retrieve_context_with_audit(selected_kpi)

    schema_cols = schema_df.iloc[:, 0].astype(str).tolist()
    data_cols = data_df.columns.tolist() if data_df is not None else []

    with st.spinner("Running AI gap analysis..."):
        result = run_gap_analysis(
            kpi_row["KPI ID"],
            selected_kpi,
            context,
            schema_cols,
            data_cols
        )

    audit_score = calculate_audit_score(result, audit_rows)

    # =================================================
    # UPDATE KPI TABLE
    # =================================================
    kpi_df.loc[kpi_row.name, "Feasibility"] = result["feasibility"]
    kpi_df.loc[kpi_row.name, "Available Columns"] = ", ".join(result["available_fields"])
    kpi_df.loc[kpi_row.name, "Missing Columns"] = ", ".join(result["missing_fields"])
    kpi_df.loc[kpi_row.name, "Audit Score"] = audit_score
    kpi_df.loc[kpi_row.name, "Reason"] = result["reasoning"]

    st.success("Gap analysis completed")

    # =================================================
    # OUTPUTS
    # =================================================
    st.subheader("Audit Trace (from Qdrant)")
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True)

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    buffer = BytesIO()
    kpi_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "Download Updated KPI Excel",
        buffer,
        "kpi_gap_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
