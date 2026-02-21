import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO

from openai import OpenAI
from qdrant_client import QdrantClient

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="MSM ESG Gap Engine",
    layout="wide"
)

st.title("MSM ESG KPI Gap Analysis Engine")

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
# EMBEDDINGS
# =========================
def embed_1536(text: str):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def embed_3072(text: str):
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
def retrieve_regulatory_context(query: str):

    texts = []
    audit_trace = []

    for collection, dim in COLLECTION_CONFIG.items():

        query_vector = embed_3072(query) if dim == 3072 else embed_1536(query)

        results = qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=6
        )

        for point in results.points:
            payload = point.payload or {}

            # ✅ actual regulation text
            text = payload.get("text") or payload.get("document") or ""

            if text:
                texts.append(f"[{collection}] {text}")

                audit_trace.append({
                    "collection": collection,
                    "text": text[:500]   # first 500 chars for audit
                })

    return "\n\n".join(texts), audit_trace

# =========================
# GAP ANALYSIS PROMPT (UNCHANGED)
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
# FILE DISCOVERY (DROPDOWNS)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

excel_files = sorted([
    f for f in os.listdir(BASE_DIR)
    if f.lower().endswith(".xlsx")
])

if not excel_files:
    st.error("No Excel files found in repository.")
    st.stop()

# =========================
# KPI MASTER
# =========================
st.subheader("Select KPI Master File")

kpi_file_name = st.selectbox("Select KPI Excel", excel_files)
kpi_df = pd.read_excel(os.path.join(BASE_DIR, kpi_file_name))

# Ensure audit columns exist
for col in ["Audit Score", "Traceability"]:
    if col not in kpi_df.columns:
        kpi_df[col] = ""

# =========================
# KPI SELECTION
# =========================
selected_kpi = st.selectbox(
    "Select KPI for Gap Analysis",
    kpi_df["KPI Name"]
)

selected_row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]
selected_kpi_id = selected_row["KPI ID"]

# =========================
# SCHEMA + SAMPLE DATA
# =========================
st.subheader("Select Supporting Files")

schema_file_name = st.selectbox("Select Schema Excel", excel_files)
sample_file_name = st.selectbox(
    "Select Sample Data Excel (optional)",
    ["None"] + excel_files
)

schema_df = pd.read_excel(os.path.join(BASE_DIR, schema_file_name))

schema_text = "\n".join(
    schema_df.astype(str).agg(" | ".join, axis=1)
)

# sample data is loaded only for completeness / audit
if sample_file_name != "None":
    sample_df = pd.read_excel(os.path.join(BASE_DIR, sample_file_name))
else:
    sample_df = None

# =========================
# RUN ANALYSIS
# =========================
if st.button("Run Gap Analysis"):

    with st.spinner("Retrieving regulatory context..."):
        reg_text, audit_trace = retrieve_regulatory_context(selected_kpi)

    with st.spinner("Running AI Gap Analysis..."):
        raw_output = run_gap_analysis(
            selected_kpi_id,
            selected_kpi,
            reg_text,
            schema_text
        )

    result = json.loads(raw_output)

    # -------------------------
    # AUDIT SCORE
    # -------------------------
    required = result["required_fields"]
    available = result["available_fields"]

    audit_score = int((len(available) / len(required)) * 100) if required else 0

    # -------------------------
    # TRACEABILITY (COLLECTION NAMES)
    # -------------------------
    trace_refs = sorted(set([e["collection"] for e in audit_trace]))

    # =========================
    # UPDATE KPI TABLE
    # =========================
    kpi_df.loc[kpi_df["KPI Name"] == selected_kpi, "Feasiblity"] = result["feasibility"]
    kpi_df.loc[kpi_df["KPI Name"] == selected_kpi, "Column names which are available"] = ", ".join(available)
    kpi_df.loc[kpi_df["KPI Name"] == selected_kpi, "Column name which are required more"] = ", ".join(result["missing_fields"])
    kpi_df.loc[kpi_df["KPI Name"] == selected_kpi, "Reason"] = result["reasoning"]
    kpi_df.loc[kpi_df["KPI Name"] == selected_kpi, "Audit Score"] = audit_score
    kpi_df.loc[kpi_df["KPI Name"] == selected_kpi, "Traceability"] = "; ".join(trace_refs)

    st.success("Gap Analysis Completed")

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    with st.expander("Audit Evidence (Vector DB Text Used)"):
        st.json(audit_trace)

    # =========================
    # DOWNLOAD UPDATED FILE
    # =========================
    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="Download Updated KPI Excel",
        data=output,
        file_name="Excel_Updated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
