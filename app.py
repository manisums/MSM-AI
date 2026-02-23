import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO

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
llm_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, check_compatibility=False)

# =====================================================
# CONSTANTS
# =====================================================
REGULATORY_COLLECTIONS = ["esrs_e1", "esg_regulations"]

# =====================================================
# EMBEDDING (1536)
# =====================================================
def embed(text: str):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =====================================================
# NORMALIZER
# =====================================================
def norm(col):
    return col.strip().lower().replace(" ", "_")

# =====================================================
# RETRIEVE REGULATORY CONTEXT
# =====================================================
def retrieve_regulatory_context(kpi_name):
    vector = embed(kpi_name)
    context, audit = [], []

    for col in REGULATORY_COLLECTIONS:
        results = qdrant_client.query_points(col, vector, limit=4)
        for p in results.points:
            payload = p.payload or {}
            text = payload.get("content") or payload.get("text") or payload.get("document")
            if text:
                context.append(text)
                audit.append({
                    "collection": col,
                    "source": payload.get("source"),
                    "page": payload.get("page")
                })

    return "\n\n".join(context), audit

# =====================================================
# MSM CONTEXT
# =====================================================
def retrieve_msm_context(kpi_name):
    vector = embed(kpi_name)
    results = qdrant_client.query_points("msm_dataverse_model", vector, limit=6)

    context, audit = [], []
    for p in results.points:
        payload = p.payload or {}
        if payload.get("content"):
            context.append(payload["content"])
            audit.append({
                "collection": "msm_dataverse_model",
                "source": payload.get("source"),
                "page": None
            })

    return "\n\n".join(context), audit

# =====================================================
# BOR CONTEXT (FOR COMPUTED KPI)
# =====================================================
def retrieve_bor_context(kpi_name):
    vector = embed(kpi_name)
    results = qdrant_client.query_points("client_bor", vector, limit=4)

    context, audit = [], []
    for p in results.points:
        payload = p.payload or {}
        if payload.get("content"):
            context.append(payload["content"])
            audit.append({
                "collection": "client_bor",
                "source": payload.get("source"),
                "page": payload.get("page")
            })

    return "\n\n".join(context), audit

# =====================================================
# REGULATORY GAP ANALYSIS
# =====================================================
def run_regulatory_gap(kpi_id, kpi_name, reg_text):
    prompt = f"""
KPI ID: {kpi_id}
KPI Name: {kpi_name}

REGULATORY CONTEXT:
{reg_text}

TASK:
1. Identify required fields
2. Decide feasibility

Return JSON:
{{
  "feasibility": "",
  "required_fields": [],
  "reasoning": ""
}}
"""
    res = llm_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(res.choices[0].message.content)

# =====================================================
# MSM GAP
# =====================================================
def run_msm_gap(kpi_id, kpi_name, msm_text):
    prompt = f"""
KPI ID: {kpi_id}
KPI Name: {kpi_name}

MSM DATA MODEL:
{msm_text}

Return JSON:
{{
  "feasibility": "",
  "available_fields": [],
  "missing_fields": [],
  "reasoning": ""
}}
"""
    res = llm_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(res.choices[0].message.content)

# =====================================================
# KPI COMPUTATION (EMISSIONS)
# =====================================================
def compute_emission(data_df):
    numeric_cols = data_df.select_dtypes(include="number")
    return round(numeric_cols.sum().sum(), 2) if not numeric_cols.empty else None

# =====================================================
# FILE SELECTION (ORDER FIXED)
# =====================================================
files = [f for f in os.listdir(".") if f.endswith(".xlsx")]

default_kpi = files.index("excel.xlsx") if "excel.xlsx" in files else 0
kpi_file = st.selectbox("Select KPI Master File", files, index=default_kpi)

kpi_df = pd.read_excel(kpi_file)
selected_kpi = st.selectbox("Select KPI", kpi_df["KPI Name"])
row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]
kpi_id = row["KPI ID"]

schema_file = st.selectbox("Select Schema File", files)
data_file = st.selectbox("Select Sample Data File", files)

schema_df = pd.read_excel(schema_file)
data_df = pd.read_excel(data_file)

# =====================================================
# ENSURE OUTPUT COLUMNS
# =====================================================
for c in ["Feasibility","Available Columns","Missing Columns","Mapping to MSM",
          "KPI Value","Calculation Source","Audit Score","Reason"]:
    if c not in kpi_df.columns:
        kpi_df[c] = ""

# =====================================================
# RUN GAP ANALYSIS
# =====================================================
if st.button("Run Gap Analysis"):

    reg_text, reg_audit = retrieve_regulatory_context(selected_kpi)
    reg_result = run_regulatory_gap(kpi_id, selected_kpi, reg_text)

    schema_cols = schema_df.iloc[:,0].astype(str).tolist()
    data_cols = list(map(str, data_df.columns))
    available = {norm(c): c for c in schema_cols + data_cols}

    missing = [c for c in reg_result["required_fields"] if norm(c) not in available]

    idx = kpi_df[kpi_df["KPI Name"] == selected_kpi].index[0]

    kpi_df.loc[idx,"Feasibility"] = reg_result["feasibility"]
    kpi_df.loc[idx,"Available Columns"] = ", ".join(available.values())
    kpi_df.loc[idx,"Missing Columns"] = ", ".join(missing)
    kpi_df.loc[idx,"Audit Score"] = round(100*(1-len(missing)/max(len(reg_result["required_fields"]),1)),1)
    kpi_df.loc[idx,"Reason"] = reg_result["reasoning"]

    st.subheader("Audit Traceability")
    st.dataframe(pd.DataFrame(reg_audit), use_container_width=True)

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    # =================================================
    # MSM QUESTION (ONLY AFTER ABOVE)
    # =================================================
    st.subheader("MSM Mapping Decision")
    map_msm = st.radio("Should this KPI be mapped to MSM?", ["Yes","No"], horizontal=True)

    if map_msm == "Yes":
        msm_text, msm_audit = retrieve_msm_context(selected_kpi)
        msm_res = run_msm_gap(kpi_id, selected_kpi, msm_text)

        kpi_df.loc[idx,"Mapping to MSM"] = "Yes"
        kpi_df.loc[idx,"Feasibility"] = msm_res["feasibility"]
        kpi_df.loc[idx,"Available Columns"] = ", ".join(msm_res["available_fields"])
        kpi_df.loc[idx,"Missing Columns"] = ", ".join(msm_res["missing_fields"])
        kpi_df.loc[idx,"Calculation Source"] = "MSM Dataverse"
        kpi_df.loc[idx,"KPI Value"] = ""

        st.subheader("MSM Audit Trace")
        st.dataframe(pd.DataFrame(msm_audit), use_container_width=True)

    else:
        kpi_value = compute_emission(data_df)
        _, bor_audit = retrieve_bor_context(selected_kpi)

        kpi_df.loc[idx,"Mapping to MSM"] = "No"
        kpi_df.loc[idx,"KPI Value"] = kpi_value
        kpi_df.loc[idx,"Calculation Source"] = "Derived (Non-MSM)"

        st.subheader("BoR Audit Trace")
        st.dataframe(pd.DataFrame(bor_audit), use_container_width=True)

    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download KPI Analysis",
        data=output,
        file_name="KPI_Gap_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
