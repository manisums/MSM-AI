def safe_json_parse(text: str):
    """
    Safely parse JSON from LLM output
    Prevents Streamlit crashes
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring
        start = text.find("{")
        end = text.rfind("}") + 1

        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end])
            except Exception:
                pass

        # Final fallback
        st.error("LLM did not return valid JSON.")
        st.code(text)
        st.stop()
