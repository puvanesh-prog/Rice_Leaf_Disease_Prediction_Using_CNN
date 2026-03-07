import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="Rice Doctor AI", layout="centered")

# 1. Load the Model (Unga .h5 file name inga correct-ah kudunga)
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model('rice_leaf_disease_final.keras', compile=False)
    return model

model = load_my_model()

# 2. Disease Labels & Solutions (Intha order-ah notebook classes-ku mathi check pannunga)
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

remedies = {
    'Bacterial leaf blight': "Advice: Use balanced fertilizer. Spray Copper oxychloride (2.5 g/L).",
    'Brown spot': "Advice: Improve soil fertility. Apply Mancozeb (2.0 g/L) if infection is severe.",
    'Leaf smut': "Advice: Use certified seeds. Spray Propiconazole (1 ml/L) during tillering stage."
}

# 3. UI Design
st.title("🌾 Rice Leaf Disease Detector")
st.write("Upload a rice leaf photo, and our AI will diagnose the problem.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image Display
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)
    
    st.write("🔄 Diagnosing...")

    # 4. Preprocessing (Notebook-la panna adhe steps)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)

    # 5. Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Confidence score calculation
    
    result = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    # 6. Result Display
    st.success(f"### Result: {result}")
    st.info(f"**Confidence Level:** {confidence:.2f}%")
    
    # Remedies Section
    st.subheader("💡 Prevention & Cure:")
    st.write(remedies[result])

st.markdown("---")
st.caption("Developed with ❤️ for Farmers")
