import streamlit as st
from PIL import Image
import logging
import warnings
from openai import OpenAI
from transformers import pipeline

# Suppress specific warnings
warnings.filterwarnings("ignore")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client for text generation
try:
    client = OpenAI(
        base_url="https://api-inference.huggingface.co/v1/",
        api_key="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"
    )
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    client = None

# Load image captioning model
try:
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    logger.info("Image captioning model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load image captioning model: {e}")
    image_to_text = None

# Streamlit App
st.set_page_config(page_title="AI Chat & Captioning App", layout="centered", page_icon="✨")

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✨ AI-Powered Chat & Image Captioning App")
st.markdown("---")

# Tabs for different functionalities
tabs = st.tabs(["📝 Text Generation", "🖼️ Image Captioning"])

# Text Generation Tab
with tabs[0]:
    st.header("📝 Generate Text")
    st.markdown("Enter your prompt below and let the AI create something amazing!")

    text_input = st.text_area("Enter your text prompt:", placeholder="Type something here...")
    generate_btn = st.button("Generate")

    if generate_btn:
        if client and text_input.strip():
            with st.spinner("Generating text..."):
                try:
                    messages = [{"role": "user", "content": text_input}]
                    completion = client.chat.completions.create(
                        model="microsoft/Phi-3.5-mini-instruct",
                        messages=messages,
                        max_tokens=4069
                    )
                    generated_content = completion.choices[0].message.content

                    if not generated_content.endswith(('.', '!', '?')):
                        generated_content = generated_content.rsplit('.', 1)[0] + '.'

                    st.success("✨ Generated Text:")
                    st.write(generated_content)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("Please enter a valid text prompt.")

# Image Captioning Tab
with tabs[1]:
    st.header("🖼️ Generate Captions for Images")
    st.markdown("Upload an image, and the AI will generate a caption for it!")

    uploaded_image = st.file_uploader("Upload your image:", type=["jpeg", "jpg", "png", "bmp", "gif"])
    generate_caption_btn = st.button("Generate Caption")

    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if generate_caption_btn:
        if uploaded_image and image_to_text:
            with st.spinner("Generating caption..."):
                try:
                    image = Image.open(uploaded_image)

                    if image.format not in ["JPEG", "PNG", "BMP", "GIF"]:
                        st.error("Unsupported image format. Please upload JPEG, PNG, BMP, or GIF.")
                    else:
                        result = image_to_text(image)
                        caption = result[0]['generated_text'] if result else "No caption could be generated."
                        st.success("✨ Generated Caption:")
                        st.write(caption.capitalize())
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.error("No image uploaded or the captioning model is unavailable.")
