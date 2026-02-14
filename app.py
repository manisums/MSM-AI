import streamlit as st
import pandas as pd
import json
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
# SECRETS (STREAMLIT CLOUD)
# =========================
try:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("Secrets not found. Please configure secrets in Streamlit Cloud.")
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
# EMBEDDING FUNCTIONS
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
# RETRIEVE REGULATORY CONTEXT
# =========================
def retrieve_regulatory_context(query: str) -> str:

    texts = []

    for collection, dim in COLLECTION_CONFIG.items():

        query_vector = (
            embed_3072(query) if dim == 3072 else embed_1536(query)
        )

        results = qdrant_client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=6
        )

        for point in results.points:
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
        model="gpt-5",
        temperature=0.1,
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

kpi_file = st.file_uploader(
    "Upload KPI Excel",
    type=["xlsx"]
)

if not kpi_file:
    st.warning("Please upload KPI Excel file to proceed.")
    st.stop()

kpi_df = pd.read_excel(kpi_file)
st.success("KPI file loaded successfully")

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
# SUPPORTING FILE UPLOADS
# =========================
st.subheader("Upload Supporting Files")

schema_file = st.file_uploader(
    "Upload Schema Excel",
    type=["xlsx"]
)

data_file = st.file_uploader(
    "Upload Sample Data Excel (optional)",
    type=["xlsx"]
)

# =========================
# RUN ANALYSIS
# =========================
if schema_file and st.button("Run Gap Analysis"):

    schema_df = pd.read_excel(schema_file)

    schema_text = "\n".join(
        schema_df.astype(str).agg(" | ".join, axis=1)
    )

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
