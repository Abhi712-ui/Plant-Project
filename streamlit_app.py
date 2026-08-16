import io
import json
from pathlib import Path
import streamlit as st
from PIL import Image
from app.inference import load_model, load_class_names, predict

st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")

@st.cache_resource
def get_model():
    class_names = load_class_names()
    model = load_model(len(class_names))
    return model, class_names

model, class_names = get_model()

st.title("Plant Disease Classifier")
st.write(
    "Upload a photograph of a plant and the model will identify the crop species and and diseases present"
    "\nTrained on the New Plant Diseases dataset with 98.7 percent accuracy"
)
uploaded_file = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
        
    with col2:
        with st.spinner("Analyzing..."):
            results = predict(model, class_names, image, top_k=3)
            
        st.subheader("Prediction")
        top = results[0]
        label = top["class"].replace("___", " — ").replace("_", " ")
        st.markdown(f"### {label}")
        st.progress(top["confidence"])
        st.caption(f"Confidence: {top['confidence']:.1%}")
        
        st.subheader("Other possibilities")
        for r in results[1:]:
            alt_label = r["class"].replace("___", " — ").replace("_", " ")
            st.write(f"{alt_label} — {r['confidence']:.1%}")
    
st.divider()
st.caption(
    "Model: ResNet18 fine-tuned via transfer learning"
)
