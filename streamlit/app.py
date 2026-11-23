"""Streamlit interface for the local AI Voice Assistant."""

from uuid import uuid4

import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder

from api import send_chat, send_voice

st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid4().hex

with st.sidebar:
    st.title("🤖 AI Voice Assistant")
    st.caption("Conversation session")
    st.code(st.session_state.session_id, language=None)

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = uuid4().hex
        st.rerun()

    if st.button("🗑 Clear Visible Chat", use_container_width=True):
        # This only clears the current browser view. Persistent history remains
        # in SQLite and can be deleted through the conversation API.
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.toggle("Dark Mode", value=True)
    st.divider()
    st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"],
        disabled=True,
        help="Document upload is available through the backend API; frontend integration is not implemented yet.",
    )

st.title("🤖 AI Voice Assistant")
st.caption("Local LLM chat, document retrieval and voice processing")

chat_tab, voice_tab = st.tabs(["💬 Chat", "🎤 Voice"])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Type your message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 AI is thinking..."):
                try:
                    result = send_chat(st.session_state.session_id, prompt)
                    answer = result.get("response", "")
                    if not answer:
                        raise RuntimeError("The assistant returned an empty response.")
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    st.error(f"Unable to contact the assistant: {exc}")
        st.rerun()

with voice_tab:
    st.write("### 🎤 Record a message")
    audio_bytes = audio_recorder(
        text="",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:
        with st.spinner("🤖 Transcribing and generating a response..."):
            try:
                result = send_voice(st.session_state.session_id, audio_bytes)
                st.success("Speech processed successfully")
                st.write(f"**You:** {result.get('transcription', '')}")
                st.write(f"**Assistant:** {result.get('response', '')}")
                audio_content = result.get("audio_bytes")
                if audio_content:
                    st.audio(audio_content, format="audio/mp3")
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                st.error(f"Unable to process the voice message: {exc}")