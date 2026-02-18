import streamlit as st
from vectorstore import create_vectorstore
from query import analyze_resume
from chatbot import chat_with_resume
import tempfile

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("AI Resume Analyzer")

st.subheader("Upload Resume")
uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if uploaded_file:
    if (
        "file_name" not in st.session_state
        or uploaded_file.name != st.session_state.file_name
    ):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        st.session_state.vectorstore = create_vectorstore(temp_path)
        st.session_state.file_name = uploaded_file.name
        st.session_state.chat_history = []
        st.success(f"Resume processed: {uploaded_file.name}")

if "file_name" in st.session_state:
    st.info(f"Current resume: {st.session_state.file_name}")

jd_text = st.text_area("Enter Job Description")

if st.button("Analyze Resume"):
    if "vectorstore" in st.session_state:
        if jd_text.strip():
            result = analyze_resume(st.session_state.vectorstore, jd_text)
            st.session_state.analysis_result = result
            st.session_state.jd_text = jd_text
        else:
            st.warning("Please enter a job description")
    else:
        st.warning("Please upload a resume first")

if "analysis_result" in st.session_state:
    st.subheader("Analysis Result")
    st.write(st.session_state.analysis_result)

with st.sidebar:
    st.header("Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        st.chat_message(message["role"]).write(message["content"])

    if prompt := st.chat_input("Ask your question"):
        if "vectorstore" in st.session_state:
            analysis = st.session_state.get("analysis_result", "")
            jd = st.session_state.get("jd_text", "")

            answer = chat_with_resume(
                st.session_state.vectorstore,
                prompt,
                [m["content"] for m in st.session_state.chat_history],
                analysis,
                jd
            )

            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()
        else:
            st.warning("Please upload a resume first")