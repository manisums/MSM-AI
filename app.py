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

st.title("MSM ESG KPI Gap Analysis Engine (Audit Ready)")

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
# FILE DISCOVERY
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
# RETRIEVE CONTEXT
# =========================
def retrieve_regulatory_context(query):

    texts = []
    used_collections = set()

    for collection, dim in COLLECTION_CONFIG.items():

        vector = embed_3072(query) if dim == 3072 else embed_1536(query)

        results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=6
        )

        for point in results.points:
            txt = (
                point.payload.get("text")
                or point.payload.get("document")
                or ""
            )
            if txt:
                used_collections.add(collection)
                texts.append(f"[{collection}] {txt}")

    return "\n\n".join(texts), list(used_collections)

# =========================
# GAP ANALYSIS PROMPT
# =========================
SYSTEM_PROMPT = """
You are an ESG regulatory and carbon accounting expert.
Use only regulatory context to determine feasibility.
Return strictly valid JSON only.
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
# AUDIT SCORE
# =========================
def calculate_audit_score(result):

    total = len(result["required_fields"])
    available = len(result["available_fields"])

    if total == 0:
        return 0

    coverage = available / total

    if result["feasibility"] == "FULLY_CALCULABLE":
        base = 80
    elif result["feasibility"] == "PARTIALLY_CALCULABLE":
        base = 50
    else:
        base = 20

    return min(100, base + int(coverage * 20))

# =========================
# AUDIT SAVE
# =========================
def save_audit_trail(record):
    path = os.path.join(AUDIT_DIR, f"{record['kpi_id']}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

# =========================
# FILE SELECTION UI
# =========================
st.subheader("Select Input Files")

kpi_file = st.selectbox("Select KPI Master Excel", excel_files)
schema_file = st.selectbox("Select Schema Excel", excel_files)

kpi_df = pd.read_excel(os.path.join(BASE_DIR, kpi_file))
schema_df = pd.read_excel(os.path.join(BASE_DIR, schema_file))

schema_text = "\n".join(
    schema_df.astype(str).agg(" | ".join, axis=1)
)

# =========================
# KPI MULTI SELECTION
# =========================
st.subheader("Select KPIs for Gap Analysis")

selected_kpis = st.multiselect(
    "Choose one or more KPIs",
    kpi_df["KPI Name"]
)

if not selected_kpis:
    st.warning("Select at least one KPI.")
    st.stop()

# =========================
# RUN BATCH
# =========================
if st.button("Run Batch Gap Analysis"):

    progress = st.progress(0)
    total = len(selected_kpis)

    for idx, kpi_name in enumerate(selected_kpis, start=1):

        row = kpi_df[kpi_df["KPI Name"] == kpi_name].iloc[0]
        kpi_id = row["KPI ID"]

        reg_text, used_collections = retrieve_regulatory_context(kpi_name)

        raw = run_gap_analysis(
            kpi_id,
            kpi_name,
            reg_text,
            schema_text
        )

        result = json.loads(raw)
        score = calculate_audit_score(result)

        # Update KPI Master
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Feasiblity"] = result["feasibility"]
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Audit Score"] = score
        kpi_df.loc[kpi_df["KPI Name"] == kpi_name, "Reason"] = result["reasoning"]

        # Audit Trail
        audit_record = {
            "kpi_id": kpi_id,
            "kpi_name": kpi_name,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "feasibility": result["feasibility"],
            "audit_score": score,
            "required_fields": result["required_fields"],
            "available_fields": result["available_fields"],
            "missing_fields": result["missing_fields"],
            "regulatory_collections": used_collections,
            "evidence_snippets": reg_text.split("\n\n")[:5],
            "reasoning": result["reasoning"],
            "llm_model": "gpt-4.1",
            "embedding_models": [
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]
        }

        save_audit_trail(audit_record)

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
