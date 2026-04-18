from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from backend.agent import study_coach_agent
from backend.database import student_repo
from backend.model_service import llm_service
from backend.rag_service import rag_service

st.set_page_config(page_title="AI Study Buddy", layout="wide")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "student_id" not in st.session_state:
    st.session_state.student_id = "student-1"
if "topic" not in st.session_state:
    st.session_state.topic = "General"

st.title("🎓 AI Study Buddy — Streamlit")

with st.sidebar:
    st.header("Session")
    st.session_state.student_id = st.text_input("Student ID", value=st.session_state.student_id)
    st.session_state.topic = st.text_input("Active Topic", value=st.session_state.topic)
    st.caption(f"Model: {llm_service.model_name}")
    st.caption(f"Fallback Mode: {llm_service.fallback_mode}")

chat_tab, upload_tab, tools_tab, dashboard_tab, coach_tab = st.tabs(
    ["Chat", "Upload & Index", "Study Tools", "Dashboard", "AI Coach"]
)

with chat_tab:
    st.subheader("Chat")
    message = st.text_area("Ask a question", height=120)
    if st.button("Send", key="chat_send"):
        with st.spinner("Thinking..."):
            context = rag_service.retrieve(
                query=message or st.session_state.topic,
                student_id=st.session_state.student_id,
                top_k=4,
            )
            answer = llm_service.generate(task="chat", user_input=message, context=context)
            st.session_state.chat_history.append({"user": message, "assistant": answer})

    for pair in st.session_state.chat_history[::-1]:
        st.markdown(f"**You:** {pair['user']}")
        st.markdown(f"**Assistant:** {pair['assistant']}")
        st.divider()

with upload_tab:
    st.subheader("Upload Notes / PDF")
    uploads = st.file_uploader("Upload files", accept_multiple_files=True, type=["txt", "md", "pdf"])
    if st.button("Index Files", key="index_files") and uploads:
        docs = [(item.name, item.getvalue()) for item in uploads]
        with st.spinner("Indexing into ChromaDB..."):
            chunks, docs_count = rag_service.index_documents(student_id=st.session_state.student_id, documents=docs)
        st.success(f"Indexed {chunks} chunks from {docs_count} documents")

    query = st.text_input("Test retrieval query", value=st.session_state.topic)
    if st.button("Retrieve", key="retrieve"):
        chunks = rag_service.retrieve(query=query, student_id=st.session_state.student_id, top_k=4)
        if not chunks:
            st.warning("No chunks found. Upload and index notes first.")
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}:** {chunk}")


def feature_prompt(feature: str, topic: str) -> str:
    mapping = {
        "Summary": "Create a concise study summary",
        "Flashcards": "Generate 10 flashcards in Q/A format",
        "Quiz": "Generate 5 MCQs with options and answer key",
        "Mind Map": "Generate an indented markdown mind map",
    }
    return mapping[feature] + f" for {topic}"


with tools_tab:
    st.subheader("Study Tools")
    feature = st.selectbox("Tool", ["Summary", "Flashcards", "Quiz", "Mind Map"])
    query = st.text_input("Topic / Query", value=st.session_state.topic, key="tool_query")

    if st.button("Generate", key="feature_generate"):
        with st.spinner("Generating..."):
            chunks = rag_service.retrieve(query=query, student_id=st.session_state.student_id, top_k=4)
            output = llm_service.generate(task=feature_prompt(feature, query), user_input=query, context=chunks)
        st.markdown(output)

        if feature == "Quiz":
            score = st.slider("Log your score (%)", min_value=0, max_value=100, value=70)
            if st.button("Save Quiz Score"):
                student_repo.record_quiz_attempt(
                    student_id=st.session_state.student_id,
                    topic=query,
                    score=float(score),
                    metadata=json.dumps({"source": "streamlit_quiz"}),
                )
                st.success("Quiz attempt saved")

with dashboard_tab:
    st.subheader("Performance Dashboard")
    student_id = st.session_state.student_id
    topic_scores = student_repo.get_topic_scores(student_id)
    trend = student_repo.get_trend(student_id)
    recent = student_repo.get_recent_attempts(student_id, limit=10)
    weak_topics = study_coach_agent.get_weak_topics(student_id)

    if trend:
        st.line_chart(pd.DataFrame(trend).set_index("date")["avg_score"])
    else:
        st.info("No trend data yet.")

    if topic_scores:
        st.bar_chart(pd.DataFrame(topic_scores).set_index("topic")["avg_score"])
    else:
        st.info("No topic scores yet.")

    st.markdown("**Recent Attempts**")
    st.dataframe(pd.DataFrame(recent) if recent else pd.DataFrame(columns=["topic", "score", "attempted_at"]))
    st.markdown("**Weak Topics**")
    st.write(weak_topics if weak_topics else "No weak topics detected")

with coach_tab:
    st.subheader("AI Coach")
    suggestion = study_coach_agent.recommend(st.session_state.student_id)
    st.info(suggestion.recommendation)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Accept Suggestion"):
            student_repo.record_feedback(
                student_id=st.session_state.student_id,
                topic=suggestion.topic,
                suggestion_type=suggestion.suggestion_type,
                accepted=True,
            )
            output = study_coach_agent.run_tool(suggestion)
            st.success("Suggestion accepted and executed")
            st.markdown(output)

    with col2:
        if st.button("Decline Suggestion"):
            student_repo.record_feedback(
                student_id=st.session_state.student_id,
                topic=suggestion.topic,
                suggestion_type=suggestion.suggestion_type,
                accepted=False,
            )
            st.warning("Suggestion declined and logged")
