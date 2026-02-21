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
st.title("MSM ESG KPI Gap Analysis Engine (Audit Safe)")

# =========================
# SECRETS
# =========================
try:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Secrets not found.")
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
# RETRIEVE REGULATORY CONTEXT + TRACE
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
# GAP ANALYSIS (LLM DOES ONLY REQUIRED FIELDS)
# =========================
SYSTEM_PROMPT = """
You are an ESG regulatory expert.
Identify REQUIRED data fields only.
Return valid JSON only.
"""

def run_gap_analysis(kpi_id, kpi_name, reg_text):

    USER_PROMPT = f"""
KPI ID: {kpi_id}
KPI NAME: {kpi_name}

REGULATORY CONTEXT:
{reg_text}

TASK:
List only the REQUIRED fields to calculate this KPI.

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
            {"role": "user", "content": USER_PROMPT}
        ]
    )

    return response.choices[0].message.content

# =========================
# KPI FILE UPLOAD
# =========================
st.subheader("Upload KPI Master File")

kpi_file = st.file_uploader("Upload KPI Excel", type=["xlsx"])
if not kpi_file:
    st.stop()

kpi_df = pd.read_excel(kpi_file)

# =========================
# KPI SELECTION
# =========================
selected_kpi = st.selectbox("Select KPI", kpi_df["KPI Name"])
row_idx = kpi_df[kpi_df["KPI Name"] == selected_kpi].index[0]
kpi_id = kpi_df.at[row_idx, "KPI ID"]

# =========================
# SUPPORTING FILES
# =========================
schema_file = st.file_uploader("Upload Schema Excel", type=["xlsx"])

if not schema_file:
    st.stop()

schema_df = pd.read_excel(schema_file)
schema_columns = [c.lower().strip() for c in schema_df.columns]

# =========================
# RUN ANALYSIS
# =========================
if st.button("Run Gap Analysis"):

    reg_text, evidence = retrieve_regulatory_context(selected_kpi)
    raw = run_gap_analysis(kpi_id, selected_kpi, reg_text)

    result = json.loads(raw)
    required = [f.lower().strip() for f in result["required_fields"]]

    available = [f for f in required if f in schema_columns]
    missing = [f for f in required if f not in schema_columns]

    feasibility = (
        "FULLY_CALCULABLE" if not missing else
        "PARTIALLY_CALCULABLE" if available else
        "NOT_CALCULABLE"
    )

    # SAFE ASSIGNMENTS (NO NULLS)
    kpi_df.at[row_idx, "Feasiblity"] = feasibility
    kpi_df.at[row_idx, "Column names which are available"] = ", ".join(available)
    kpi_df.at[row_idx, "Column name which are required more"] = ", ".join(missing)
    kpi_df.at[row_idx, "Reason"] = result["reasoning"]

    # =========================
    # AUDIT LOG
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
        "audit_trace": evidence
    }

    with open(f"audit_logs/{kpi_id}.json", "w") as f:
        json.dump(audit_record, f, indent=2)

    st.success("Gap Analysis Completed")
    st.dataframe(kpi_df, use_container_width=True)

    with st.expander("Audit Trace"):
        st.json(evidence)

    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Updated KPI Excel",
        output,
        "Excel_Updated.xlsx"
    )
