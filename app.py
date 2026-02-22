import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO
from uuid import uuid4

from openai import OpenAI
from qdrant_client import QdrantClient

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="MSM ESG KPI Gap Analysis Engine",
    layout="wide"
)
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
llm_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# =====================================================
# COLLECTION ROLES
# =====================================================
REGULATORY_COLLECTIONS = [
    "esrs_e1",
    "esg_regulations"
]

SUPPORTING_COLLECTIONS = [
    "client_bor",
    "msm_dataverse_model"
]

# =====================================================
# EMBEDDING (ALL 1536)
# =====================================================
def embed(text: str):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =====================================================
# UTILS
# =====================================================
def normalize(col):
    return col.strip().lower().replace(" ", "_")

# =====================================================
# REGULATORY CONTEXT + AUDIT
# =====================================================
def retrieve_regulatory_context(kpi_name: str):
    vector = embed(kpi_name)

    context = []
    audit = []

    for collection in REGULATORY_COLLECTIONS:
        results = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=4
        )

        for p in results.points:
            payload = p.payload

            text = payload.get("content")
            if not text:
                continue

            context.append(text)

            audit.append({
                "collection": collection,
                "doc_type": payload.get("doc_type"),
                "source": payload.get("source"),
                "page": payload.get("page")
            })

    return "\n\n".join(context), audit

# =====================================================
# GAP ANALYSIS PROMPT
# =====================================================
SYSTEM_PROMPT = """
You are an ESG regulatory expert.
Use ONLY the regulatory context provided.
Return STRICT JSON only.
"""

def run_gap_analysis(kpi_id, kpi_name, regulatory_text):
    USER_PROMPT = f"""
KPI ID: {kpi_id}
KPI NAME: {kpi_name}

REGULATORY CONTEXT:
{regulatory_text}

TASK:
1. Identify REQUIRED data fields needed to calculate this KPI
2. Decide feasibility:
   FULLY_CALCULABLE / PARTIALLY_CALCULABLE / NOT_CALCULABLE
3. Explain reasoning

Return JSON ONLY:

{{
  "feasibility": "",
  "required_fields": [],
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

# =====================================================
# LOAD FILES FROM REPO
# =====================================================
BASE_PATH = "."

excel_files = [f for f in os.listdir(BASE_PATH) if f.endswith(".xlsx")]

kpi_file = st.selectbox(
    "Select KPI Master File",
    excel_files
)

schema_file = st.selectbox(
    "Select Schema File",
    excel_files
)

data_file = st.selectbox(
    "Select Sample Data File",
    excel_files
)

kpi_df = pd.read_excel(kpi_file)
schema_df = pd.read_excel(schema_file)
data_df = pd.read_excel(data_file)

# =====================================================
# KPI SELECTION
# =====================================================
selected_kpi = st.selectbox(
    "Select KPI",
    kpi_df["KPI Name"]
)

kpi_row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]
kpi_id = kpi_row["KPI ID"]

# =====================================================
# RUN ANALYSIS
# =====================================================
if st.button("Run Gap Analysis"):

    # ---- Available Columns (deterministic) ----
    schema_cols = schema_df.iloc[:, 0].astype(str).tolist()
    data_cols = data_df.columns.astype(str).tolist()

    schema_norm = {normalize(c): c for c in schema_cols}
    data_norm = {normalize(c): c for c in data_cols}

    available_norm = set(schema_norm) | set(data_norm)
    available_cols = sorted(
        schema_norm.get(c, data_norm.get(c)) for c in available_norm
    )

    # ---- Regulatory context ----
    regulatory_text, audit_trace = retrieve_regulatory_context(selected_kpi)

    # ---- LLM analysis ----
    raw = run_gap_analysis(kpi_id, selected_kpi, regulatory_text)

    try:
        result = json.loads(raw)
    except Exception:
        st.error("Invalid LLM output")
        st.code(raw)
        st.stop()

    # ---- Missing columns (deterministic) ----
    required_norm = [normalize(c) for c in result["required_fields"]]

    missing_cols = [
        c for c in result["required_fields"]
        if normalize(c) not in available_norm
    ]

    # ---- Audit score ----
    if len(result["required_fields"]) == 0:
        audit_score = 0
    else:
        audit_score = round(
            100 * (1 - len(missing_cols) / len(result["required_fields"])),
            1
        )

    # =================================================
    # UPDATE KPI TABLE
    # =================================================
    idx = kpi_df[kpi_df["KPI Name"] == selected_kpi].index[0]

    kpi_df.loc[idx, "Feasibility"] = result["feasibility"]
    kpi_df.loc[idx, "Column names which are available"] = ", ".join(available_cols)
    kpi_df.loc[idx, "Column names which are required more"] = ", ".join(missing_cols)
    kpi_df.loc[idx, "Audit Score"] = audit_score
    kpi_df.loc[idx, "Reason"] = result["reasoning"]

    # =================================================
    # OUTPUTS
    # =================================================
    st.subheader("Audit Trace (Vector DB Evidence)")
    st.dataframe(pd.DataFrame(audit_trace), use_container_width=True)

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    # ---- Download ----
    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Updated KPI File",
        data=output,
        file_name="KPI_Gap_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
