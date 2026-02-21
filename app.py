import os
import json
import datetime
from io import BytesIO

import streamlit as st
import pandas as pd

from openai import OpenAI
from qdrant_client import QdrantClient

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="MSM ESG Gap Engine",
    layout="wide"
)

st.title("MSM ESG KPI Gap Analysis Engine (Audit + Evidence Ready)")

# =========================
# SECRETS
# =========================
try:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Secrets not found. Configure Streamlit secrets.")
    st.stop()

# =========================
# CLIENTS
# =========================
llm_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# =========================
# DIRECTORIES
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_DIR = os.path.join(BASE_DIR, "audit_logs")
os.makedirs(AUDIT_DIR, exist_ok=True)

def list_excel_files(directory):
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(".xlsx")
    ])

excel_files = list_excel_files(BASE_DIR)
if not excel_files:
    st.error("No Excel files found in project directory.")
    st.stop()

# =========================
# EMBEDDINGS
# =========================
def embed_1536(text):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def embed_3072(text):
    return llm_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    ).data[0].embedding

# =========================
# COLLECTION CONFIG
# =========================
COLLECTION_CONFIG = {
    "esg_regulations": 1536,
    "esrs_e1": 1536,
    "client_bor": 3072
}

# =========================
# RETRIEVE REGULATORY CONTEXT + AUDIT TRACE
# =========================
def retrieve_regulatory_context(query):

    texts = []
    evidence = []

    for collection, dim in COLLECTION_CONFIG.items():

        vector = embed_3072(query) if dim == 3072 else embed_1536(query)

        results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=6
        )

        for point in results.points:
            payload = point.payload or {}

            text = payload.get("text") or payload.get("document") or ""
            source = payload.get("source", "unknown")
            page = payload.get("page", "NA")

            if text:
                texts.append(f"[{collection}] {text}")

                evidence.append({
                    "collection": collection,
                    "source": source,
                    "page": page,
                    "snippet": text[:300]
                })

    return "\n\n".join(texts), evidence

# =========================
# GAP ANALYSIS PROMPT
# =========================
SYSTEM_PROMPT = """
You are an ESG regulatory and carbon accounting expert.
Use only regulatory context, schema and sample data.
Return strictly valid JSON only.
"""

def run_gap_analysis(kpi_id, kpi_name, reg_text, schema_text, sample_text):

    USER_PROMPT = f"""
KPI ID:
{kpi_id}

KPI NAME:
{kpi_name}

REGULATORY CONTEXT:
{reg_text}

CLIENT SCHEMA:
{schema_text}

SAMPLE DATA:
{sample_text}

TASK:
1. Identify required fields from regulation.
2. Match against schema.
3. Consider whether sample data is populated.
4. Decide feasibility:
   FULLY_CALCULABLE / PARTIALLY_CALCULABLE / NOT_CALCULABLE
5. List required, available and missing fields.
6. Provide audit-ready reasoning.

Return JSON:
{{
  "feasibility": "",
  "required_fields": [],
  "available_fields": [],
  "missing_fields": [],
  "reasoning": ""
}}
"""

    response = llm_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return response.choices[0].message.content

# =========================
# NORMALIZE FIELDS (CRITICAL FIX)
# =========================
def normalize_fields(result, schema_columns):

    available = []
    missing = []

    for field in result.get("required_fields", []):
        if field.lower().strip() in schema_columns:
            available.append(field)
        else:
            missing.append(field)

    result["available_fields"] = available
    result["missing_fields"] = missing

    return result

# =========================
# AUDIT SCORE
# =========================
def calculate_audit_score(result):

    total = len(result["required_fields"])
    available = len(result["available_fields"])

    if total == 0:
        return 0

    coverage = available / total

    if result["feasibility"] == "FULLY_CALCULABLE":
        base = 85
    elif result["feasibility"] == "PARTIALLY_CALCULABLE":
        base = 55
    else:
        base = 25

    return min(100, base + int(coverage * 15))

# =========================
# SAVE AUDIT TRAIL
# =========================
def save_audit_trail(record):
    path = os.path.join(AUDIT_DIR, f"{record['kpi_id']}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

# =========================
# FILE SELECTION
# =========================
st.subheader("Select Input Files")

kpi_file = st.selectbox("KPI Master Excel", excel_files)
schema_file = st.selectbox("Schema Excel", excel_files)
sample_file = st.selectbox("Sample Data Excel", excel_files)

kpi_df = pd.read_excel(os.path.join(BASE_DIR, kpi_file))
schema_df = pd.read_excel(os.path.join(BASE_DIR, schema_file))
sample_df = pd.read_excel(os.path.join(BASE_DIR, sample_file))

schema_columns = [c.lower().strip() for c in schema_df.columns]

schema_text = "\n".join(
    schema_df.astype(str).agg(" | ".join, axis=1)
)

sample_text = (
    sample_df.head(50)
    .astype(str)
    .agg(" | ".join, axis=1)
    .str.cat(sep="\n")
)

# =========================
# KPI SELECTION
# =========================
st.subheader("Select KPIs")

selected_kpis = st.multiselect(
    "Choose KPIs for analysis",
    kpi_df["KPI Name"]
)

if not selected_kpis:
    st.warning("Select at least one KPI.")
    st.stop()

# =========================
# RUN BATCH ANALYSIS
# =========================
if st.button("Run Batch Gap Analysis"):

    progress = st.progress(0)
    total = len(selected_kpis)

    for idx, kpi_name in enumerate(selected_kpis, start=1):

        row = kpi_df[kpi_df["KPI Name"] == kpi_name].iloc[0]
        kpi_id = row["KPI ID"]

        reg_text, evidence = retrieve_regulatory_context(kpi_name)

        raw = run_gap_analysis(
            kpi_id,
            kpi_name,
            reg_text,
            schema_text,
            sample_text
        )

        result = json.loads(raw)
        result = normalize_fields(result, schema_columns)
        score = calculate_audit_score(result)

        # Update KPI master
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Feasiblity"] = result["feasibility"]
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Audit Score"] = score
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Reason"] = result["reasoning"]
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Column names which are available"] = ", ".join(result["available_fields"])
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Column name which are required more"] = ", ".join(result["missing_fields"])

        # Audit trail
        audit_record = {
            "kpi_id": kpi_id,
            "kpi_name": kpi_name,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "feasibility": result["feasibility"],
            "audit_score": score,
            "required_fields": result["required_fields"],
            "available_fields": result["available_fields"],
            "missing_fields": result["missing_fields"],
            "schema_file": schema_file,
            "sample_file": sample_file,
            "audit_trace": evidence,
            "reasoning": result["reasoning"],
            "llm_model": "gpt-4.1",
            "embedding_models": [
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]
        }

        save_audit_trail(audit_record)

        with st.expander(f"Audit Trace – {kpi_name}"):
            st.json(audit_record["audit_trace"])

        progress.progress(idx / total)

    st.success("Batch Gap Analysis Completed")

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    # Download updated KPI master
    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="Download Updated KPI Excel",
        data=output,
        file_name="KPI_Master_Updated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
