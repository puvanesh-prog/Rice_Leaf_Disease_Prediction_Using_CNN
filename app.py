import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Model Load (Make sure the filename matches exactly)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('rice_leaf_disease_final.keras', compile=False)

model = load_my_model()

# 2. Class Names & Remedies
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy']
remedies = {
    'Bacterial Leaf Blight': "Use resistant varieties; apply balanced fertilizer.",
    'Brown Spot': "Improve soil fertility; use certified seeds.",
    'Healthy': "Keep it up! Your crop is in good condition."
}

st.title("🌿 Rice Doctor AI")
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3. Image Display
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Leaf Image', use_container_width=True)
    st.write("🔄 Diagnosing...")

    # 4. Preprocessing
    img = image_display.resize((224, 224))
    img_array = np.array(img)
    
    # Normalization (Training-la 1/255 panni irundha mattum)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0
        
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension

    # 5. Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Softmax for confidence
    
    result = class_names[np.argmax(predictions)]
    confidence = 100 * np.max(tf.nn.softmax(predictions[0]))

    # 6. Result Display
    st.success(f"### Result: {result}")
    st.info(f"**Confidence Level:** {confidence:.2f}%")

    # 7. Remedies Section
    st.subheader("💡 Prevention & Cure:")
    st.write(remedies[result])
