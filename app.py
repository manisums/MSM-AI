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
try:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Secrets not found. Please configure secrets.")
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
# HELPER: LIST EXCEL FILES
# =========================
def list_excel_files(folder="."):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(".xlsx")
    ]

excel_files = list_excel_files()

if not excel_files:
    st.error("No Excel files found in application directory.")
    st.stop()

# =========================
# EMBEDDING (1536 ONLY)
# =========================
def embed_1536(text: str):
    return llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# =========================
# COLLECTIONS (ONLY REQUIRED)
# =========================
COLLECTIONS = [
    "esg_regulations",
    "esrs_e1"
]

# =========================
# RETRIEVE REGULATORY CONTEXT
# =========================
def retrieve_regulatory_context(query: str) -> str:

    texts = []
    query_vector = embed_1536(query)

    for collection in COLLECTIONS:

        response = qdrant_client.http.search_api.search_points(
            collection_name=collection,
            search_request={
                "vector": query_vector,
                "limit": 6
            }
        )

        for point in response.result:
            txt = (
                point.payload.get("text")
                or point.payload.get("document")
                or ""
            )
            if txt:
                texts.append(f"[{collection}] {txt}")

    return "\n\n".join(texts)

# =========================
# GAP ANALYSIS PROMPTS
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
# KPI FILE SELECTION
# =========================
st.subheader("Select KPI Master File")

kpi_file_name = st.selectbox(
    "Select KPI Excel",
    excel_files
)

kpi_df = pd.read_excel(kpi_file_name)
st.success(f"KPI file loaded: {kpi_file_name}")

# =========================
# KPI SELECTION
# =========================
selected_kpi = st.selectbox(
    "Select KPI for Gap Analysis",
    kpi_df["KPI Name"]
)

selected_row = kpi_df[kpi_df["KPI Name"] == selected_kpi].iloc[0]
selected_kpi_id = selected_row["KPI ID"]

st.info(f"Selected KPI ID: {selected_kpi_id}")

# =========================
# SUPPORTING FILE SELECTION
# =========================
st.subheader("Select Supporting Files")

schema_file_name = st.selectbox(
    "Select Schema Excel",
    excel_files
)

data_file_name = st.selectbox(
    "Select Sample Data Excel (optional)",
    ["None"] + excel_files
)

# =========================
# RUN ANALYSIS
# =========================
if st.button("Run Gap Analysis"):

    schema_df = pd.read_excel(schema_file_name)

    schema_text = "\n".join(
        schema_df.astype(str).agg(" | ".join, axis=1)
    )

    # Sample data intentionally preserved but not used
    if data_file_name != "None":
        _ = pd.read_excel(data_file_name)

    with st.spinner("Retrieving regulatory context..."):
        reg_text = retrieve_regulatory_context(selected_kpi)

    with st.spinner("Running AI Gap Analysis..."):
        raw_output = run_gap_analysis(
            selected_kpi_id,
            selected_kpi,
            reg_text,
            schema_text
        )

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        st.error("Model did not return valid JSON.")
        st.code(raw_output)
        st.stop()

    # =========================
    # UPDATE KPI TABLE
    # =========================
    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Feasiblity"
    ] = result["feasibility"]

    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Column names which are available"
    ] = ", ".join(result["available_fields"])

    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Column name which are required more"
    ] = ", ".join(result["missing_fields"])

    kpi_df.loc[
        kpi_df["KPI Name"] == selected_kpi,
        "Reason"
    ] = result["reasoning"]

    st.success("Gap Analysis Completed")

    st.subheader("Updated KPI Table")
    st.dataframe(kpi_df, use_container_width=True)

    output = BytesIO()
    kpi_df.to_excel(output, index=False)
    output.seek(0)

    st.download_button(
        label="Download Updated KPI Excel",
        data=output,
        file_name="Excel_Updated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
