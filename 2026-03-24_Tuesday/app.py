import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from main import get_response

# Page setup
st.set_page_config(
    page_title="Web Intelligence Assistant",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("Settings")

    model_name = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile"]
    )

    temp = st.slider(
        "Temperature",
        0.0, 1.0, 0.3
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("Web Intelligence Assistant")
st.caption("AI-powered real-time web search using Tavily + Groq")

# Show chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.write(msg.content)

# Input
if prompt := st.chat_input("Ask anything..."):

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = get_response(
                st.session_state.messages,
                model_name,
                temp
            )

            st.write(answer)