import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Model Load
@st.cache_resource
def load_my_model():
    # Make sure the filename matches your GitHub file exactly
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
    image_display = Image.open(uploaded_file).convert('RGB') # Convert to RGB is MUST
    st.image(image_display, caption='Uploaded Leaf Image', use_container_width=True)
    st.write("🔄 Diagnosing...")

    # 4. Preprocessing
    img = image_display.resize((224, 224))
    img_array = np.array(img)
    
    # Batch dimension add panrom (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Normalization
    img_array = img_array.astype('float32') / 255.0

    # 5. Prediction
    predictions = model.predict(img_array)
    
    # Result Processing
    result_idx = np.argmax(predictions[0])
    result = class_names[result_idx]
    
    # Confidence Calculation
    score = tf.nn.softmax(predictions[0])
    confidence = 100 * np.max(score)

    # 6. Result Display
    st.success(f"### Result: {result}")
    st.info(f"**Confidence Level:** {confidence:.2f}%")

    # 7. Remedies Section
    st.subheader("💡 Prevention & Cure:")
    st.write(remedies[result])
