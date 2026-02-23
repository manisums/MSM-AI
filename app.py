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

llm_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False
)

# =====================================================
# SAFE JSON PARSER (NO MORE CRASHES)
# =====================================================
def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass
        st.error("Model did not return valid JSON")
        st.code(text)
        st.stop()

# =====================================================
# EMBEDDING (1536 – text-embedding-3-small)
# =====================================================
def embed(text: str):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =====================================================
# FIND ALL EXCEL FILES (RECURSIVE FIX)
# =====================================================
def find_excel_files(base_path="."):
    excel_files = []
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.lower().endswith(".xlsx"):
                excel_files.append(os.path.join(root, f))
    return sorted(excel_files)

excel_files = find_excel_files(".")

if not excel_files:
    st.error("No Excel files found in the repository")
    st.stop()

# =====================================================
# KPI MASTER FILE (DEFAULT = Excel.xlsx)
# =====================================================
default_kpi_idx = 0
for i, f in enumerate(excel_files):
    if os.path.basename(f).lower() == "excel.xlsx":
        default_kpi_idx = i
        break

kpi_master_file = st.selectbox(
    "Select KPI Master File",
    excel_files,
    index=default_kpi_idx
)

kpi_df = pd.read_excel(kpi_master_file)

# =====================================================
# CLEAN & STANDARDIZE KPI MASTER COLUMNS
# =====================================================
COLUMN_MAP = {
    "Feasiblity": "Feasibility",
    "Column names which are available": "Available Columns",
    "Column name which are required more": "Missing Columns"
}

kpi_df.rename(columns=COLUMN_MAP, inplace=True)

FINAL_COLUMNS = [
    "KPI ID",
    "KPI Name",
    "Feasibility",
    "Available Columns",
    "Missing Columns",
    "Audit Score",
    "Reason"
]

for col in FINAL_COLUMNS:
    if col not in kpi_df.columns:
        kpi_df[col] = ""

kpi_df = kpi_df[FINAL_COLUMNS]

# =====================================================
# KPI SELECTION
# =====================================================
selected_kpi = st.selectbox("Select KPI", kpi_df["KPI Name"])
kpi_row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]
kpi_id = kpi_row["KPI ID"]

# =====================================================
# SCHEMA & SAMPLE DATA FILES
# =====================================================
schema_file = st.selectbox(
    "Select Schema File",
    [f for f in excel_files if "schema" in os.path.basename(f).lower()]
)

data_file = st.selectbox(
    "Select Sample Data File",
    [f for f in excel_files if "data" in os.path.basename(f).lower()]
)

schema_df = pd.read_excel(schema_file)
data_df = pd.read_excel(data_file)

schema_columns = set(schema_df.iloc[:, 0].astype(str).str.lower())
data_columns = set(data_df.columns.astype(str).str.lower())

available_columns = sorted(schema_columns & data_columns)

# =====================================================
# RETRIEVE REGULATORY CONTEXT (ONLY REQUIRED COLLECTIONS)
# =====================================================
def retrieve_regulatory_context(query: str):
    audit_sources = set()
    texts = []

    for collection in ["esg_regulations", "esrs_e1"]:
        res = qdrant_client.query_points(
            collection_name=collection,
            query=embed(query),
            limit=5
        )

        for p in res.points:
            content = p.payload.get("content", "")
            if content:
                texts.append(content)
                audit_sources.add(collection)

    return "\n\n".join(texts), list(audit_sources)

# =====================================================
# REGULATORY GAP ANALYSIS
# =====================================================
def run_regulatory_gap(kpi_id, kpi_name, reg_text):

    prompt = f"""
KPI ID: {kpi_id}
KPI Name: {kpi_name}

REGULATORY CONTEXT:
{reg_text}

Return ONLY valid JSON.

{{
  "feasibility": "",
  "required_fields": [],
  "reasoning": ""
}}
"""

    res = llm_client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}]
    )

    return safe_json_parse(res.choices[0].message.content)

# =====================================================
# RUN ANALYSIS
# =====================================================
if st.button("Run Gap Analysis"):

    with st.spinner("Retrieving regulatory context..."):
        reg_text, audit_sources = retrieve_regulatory_context(selected_kpi)

    with st.spinner("Running regulatory gap analysis..."):
        reg_result = run_regulatory_gap(kpi_id, selected_kpi, reg_text)

    required_fields = [f.lower() for f in reg_result["required_fields"]]
    missing_fields = sorted(set(required_fields) - set(available_columns))

    audit_score = int(
        (len(required_fields) - len(missing_fields)) /
        max(len(required_fields), 1) * 100
    )

    idx = kpi_df["KPI Name"] == selected_kpi
    kpi_df.loc[idx, "Feasibility"] = reg_result["feasibility"]
    kpi_df.loc[idx, "Available Columns"] = ", ".join(available_columns)
    kpi_df.loc[idx, "Missing Columns"] = ", ".join(missing_fields)
    kpi_df.loc[idx, "Audit Score"] = audit_score
    kpi_df.loc[idx, "Reason"] = reg_result["reasoning"]

    st.success("Gap analysis completed")

    st.subheader("Audit Traceability")
    st.write("Sources used:", ", ".join(audit_sources))

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        "Download Updated KPI File",
        output,
        "KPI_Gap_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
