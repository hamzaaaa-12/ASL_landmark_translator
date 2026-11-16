
import streamlit as st

from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from streamlit_autorefresh import st_autorefresh

from live_predict_streamlit import ASLVideoProcessor


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Real-Time ASL Translator",
    page_icon="",
    layout="wide"
)


# ============================================================
# 2. PAGE DESIGN
# ============================================================

st.title("Real-Time ASL Alphabet Translator")

st.write(
    "Translate ASL alphabet hand gestures into text "
    "using MediaPipe and KNN."
)

st.divider()


# ============================================================
# 3. WEBRTC CONFIGURATION
# ============================================================

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)


# ============================================================
# 4. WEBCAM
# ============================================================

st.subheader("Live Camera")

ctx = webrtc_streamer(
    key="asl-translator",
    video_processor_factory=ASLVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 20, "max": 20}
        },
        "audio": False
    },
    async_processing=True
)


# ============================================================
# 5. LIVE TEXT AND STATUS
# ============================================================

st.divider()

st.subheader("Translation Output")

if ctx.state.playing:

    # Refresh the UI so it can display updated processor state
    st_autorefresh(
        interval=500,
        key="asl-refresh"
    )

    processor = ctx.video_processor

    if processor is not None:

        state = processor.get_state()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Current Prediction",
                value=state["prediction"]
            )

        with col2:
            st.metric(
                label="Status",
                value=state["status"]
            )

        st.markdown("### Recognized Text")

        st.text_area(
            "Your translated sentence",
            value=state["sentence"],
            height=120,
            disabled=True
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "Delete Last Character",
                use_container_width=True
            ):
                processor.delete_last()
                st.rerun()

        with col2:
            if st.button(
                "Clear Text",
                use_container_width=True
            ):
                processor.clear_text()
                st.rerun()

else:

    st.info(
        "Start the camera to begin recognizing ASL gestures."
    )


# ============================================================
# 6. GESTURE GUIDE
# ============================================================

st.divider()

st.subheader("How to Use")

st.markdown(
    """
    - **Add Letter:** Hold an ASL letter stable for 1.5 seconds.
    - **Add Space:** Hold Thumb + Index Finger Pinch for 1 second.
    - **Delete:** Perform a Left Swipe.
    - **Clear Text:** Use the Clear Text button.
    """
)
