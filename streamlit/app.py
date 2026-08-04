import tempfile

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

with st.sidebar:

    st.title("🤖 AI Voice Assistant")

    session_id = st.text_input(
        "Session ID",
        value="tarun",
    )

    st.divider()

    if st.button("🆕 New Chat"):

        st.session_state.messages = []

        st.rerun()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.toggle(
        "Dark Mode",
        value=True,
    )

    st.divider()

    st.file_uploader(
        "📄 Upload PDF",
        type=["pdf"],
        disabled=True,
    )

st.title("🤖 AI Voice Assistant")

chat_tab, voice_tab = st.tabs(
    [
        "💬 Chat",
        "🎤 Voice",
    ]
)

with chat_tab:

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    prompt = st.chat_input(
        "Type your message..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.spinner("🤖 AI is thinking..."):


            result = send_chat(
                session_id,
                prompt,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response"],
            }
        )

        st.rerun()

with voice_tab:

    st.write("### 🎤 Click below and speak")

    audio_bytes = audio_recorder(
        text="",
        icon_name="microphone",
        icon_size="2x",
    )

    if audio_bytes:

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as f:

            f.write(audio_bytes)

            temp_path = f.name

        with open(
            temp_path,
            "rb",
        ) as audio:

            with st.spinner("🤖 AI is processing your voice..."):

                result = send_voice(
                    session_id,
                    audio,
                )

        st.success("Speech Recognized")

        st.write(
            f"**You:** {result['transcription']}"
        )

        st.write(
            f"**Assistant:** {result['response']}"
        )

        st.audio(
            result["audio_url"],
        )