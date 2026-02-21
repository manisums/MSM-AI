import streamlit as st
import pandas as pd
import json
import os
import datetime
from io import BytesIO

from openai import OpenAI
from qdrant_client import QdrantClient

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="MSM ESG Gap Engine", layout="wide")
st.title("MSM ESG KPI Gap Analysis Engine (Final – Audit Ready)")

# =========================
# SECRETS
# =========================
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

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
# FILE DISCOVERY (NO UPLOADS)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def list_excel_files():
    return sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".xlsx")])

excel_files = list_excel_files()
if not excel_files:
    st.error("No Excel files found in project directory.")
    st.stop()

# =========================
# FILE SELECTION
# =========================
st.subheader("Select Input Files")

kpi_file = st.selectbox("Select KPI Master", excel_files)
bor_file = st.selectbox("Select BOR / Schema File", excel_files)

kpi_df = pd.read_excel(os.path.join(BASE_DIR, kpi_file))
bor_df = pd.read_excel(os.path.join(BASE_DIR, bor_file))

bor_columns = [c.lower().strip() for c in bor_df.columns]

# =========================
# ENSURE OUTPUT COLUMNS EXIST (CRITICAL)
# =========================
OUTPUT_COLUMNS = [
    "Feasiblity",
    "Available Columns",
    "Required More Columns",
    "Audit Score",
    "Traceability",
    "Reason"
]

for col in OUTPUT_COLUMNS:
    if col not in kpi_df.columns:
        kpi_df[col] = ""

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
# VECTOR COLLECTIONS
# =========================
COLLECTION_CONFIG = {
    "esg_regulations": 1536,
    "esrs_e1": 1536,
    "client_bor": 3072
}

# =========================
# RETRIEVE REGULATORY CONTEXT + TRACEABILITY
# =========================
def retrieve_regulatory_context(kpi_name):

    evidence = []
    text_blocks = []

    for collection, dim in COLLECTION_CONFIG.items():
        vector = embed_3072(kpi_name) if dim == 3072 else embed_1536(kpi_name)

        results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=5
        )

        for point in results.points:
            payload = point.payload or {}
            text = payload.get("text") or payload.get("document") or ""
            source = payload.get("source", "unknown")
            page = payload.get("page", "NA")

            if text:
                text_blocks.append(text)
                evidence.append(f"{source} (page {page})")

    return "\n".join(text_blocks), list(set(evidence))

# =========================
# LLM – REQUIRED FIELDS ONLY
# =========================
SYSTEM_PROMPT = """
You are an ESG regulatory expert.
Identify ONLY the REQUIRED data fields needed to calculate the KPI.
Do not check availability.
Return valid JSON only.
"""

def get_required_fields(kpi_id, kpi_name, reg_text):

    prompt = f"""
KPI ID: {kpi_id}
KPI NAME: {kpi_name}

REGULATORY CONTEXT:
{reg_text}

Return JSON:
{{
  "required_fields": [],
  "reasoning": ""
}}
"""

    response = llm_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(response.choices[0].message.content)

# =========================
# AUDIT SCORE
# =========================
def calculate_audit_score(required, available):
    if not required:
        return 0
    return int((len(available) / len(required)) * 100)

# =========================
# KPI SELECTION
# =========================
st.subheader("Select KPI")
selected_kpi = st.selectbox("Choose KPI", kpi_df["KPI Name"])

row_idx = kpi_df[kpi_df["KPI Name"] == selected_kpi].index[0]
kpi_id = kpi_df.at[row_idx, "KPI ID"]

# =========================
# RUN ANALYSIS
# =========================
if st.button("Run Gap Analysis"):

    reg_text, trace = retrieve_regulatory_context(selected_kpi)
    result = get_required_fields(kpi_id, selected_kpi, reg_text)

    required = [f.lower().strip() for f in result["required_fields"]]
    available = [f for f in required if f in bor_columns]
    missing = [f for f in required if f not in bor_columns]

    feasibility = (
        "FULLY_CALCULABLE" if not missing else
        "PARTIALLY_CALCULABLE" if available else
        "NOT_CALCULABLE"
    )

    audit_score = calculate_audit_score(required, available)

    # =========================
    # WRITE RESULTS (NO NULLS)
    # =========================
    kpi_df.at[row_idx, "Feasiblity"] = feasibility
    kpi_df.at[row_idx, "Available Columns"] = ", ".join(available)
    kpi_df.at[row_idx, "Required More Columns"] = ", ".join(missing)
    kpi_df.at[row_idx, "Audit Score"] = audit_score
    kpi_df.at[row_idx, "Traceability"] = "; ".join(trace)
    kpi_df.at[row_idx, "Reason"] = result["reasoning"]

    # =========================
    # SAVE AUDIT LOG
    # =========================
    os.makedirs("audit_logs", exist_ok=True)

    audit_record = {
        "kpi_id": kpi_id,
        "kpi_name": selected_kpi,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "required_fields": required,
        "available_fields": available,
        "missing_fields": missing,
        "feasibility": feasibility,
        "audit_score": audit_score,
        "traceability": trace
    }

    with open(f"audit_logs/{kpi_id}.json", "w") as f:
        json.dump(audit_record, f, indent=2)

    st.success("Gap Analysis Completed")

    st.subheader("Result")
    st.dataframe(kpi_df.loc[[row_idx]], use_container_width=True)

    # =========================
    # DOWNLOAD
    # =========================
    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Updated KPI Excel",
        output,
        "KPI_Master_Updated.xlsx"
    )
