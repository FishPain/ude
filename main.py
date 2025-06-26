import streamlit as st
from PIL import Image
from io import BytesIO
import base64

from graph import build_graph
from langchain_openai import ChatOpenAI


def encode_pil_image(pil_img: Image.Image) -> str:
    buffered = BytesIO()
    pil_img.convert("RGB").save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# --- Streamlit UI ---
st.set_page_config(page_title="BUBU Career Chaos", layout="centered")
st.title("BUBU Career Chaos - Job Application Evaluator")

st.subheader("Step 1: Capture an image")
photo = st.camera_input("Take a photo of the job and skill cards")

if photo:
    image = Image.open(photo)
    st.image(image, caption="Captured Image")

    if st.button("Send to Model"):
        st.info("Processing image... Please wait ⏳")

        try:
            model = ChatOpenAI(model="gpt-4o")
            base64_image = encode_pil_image(image)

            # LangGraph Inference
            graph = build_graph()
            state = graph.invoke({"model": model, "base64_image": base64_image})

            # Output
            is_hired = state.get("is_hired", False)
            result = state.get("scored_skills", {})

            st.subheader("🔍 Evaluation Result")
            st.markdown(f"### 💼 Verdict: {'✅ Hired' if is_hired else '❌ Not Hired'}")

            with st.expander("📊 Skill Breakdown"):
                for skill in result.get("evaluated_skills", []):
                    st.markdown(
                        f"**{skill['skill']}** — Relevance: `{skill['relevance']}/10`"
                    )
                    st.caption(skill["reason"])

        except Exception as e:
            st.error(f"❌ Error during processing:\n{str(e)}")
