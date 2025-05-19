import streamlit as st
import joblib
import json
import os
from PIL import Image
import numpy as np
import pandas as pd

# === Label Mapping ===
label_map = {0: 'Normal', 1: 'Viral Pneumonia', 2: 'Covid'}
inv_label_map = {v: k for k, v in label_map.items()}

# === Paths ===
MODEL_ROOT = r"C:\Users\moham\Desktop\Machine Learning\Project\Models"

# === Sidebar ===
st.sidebar.title("XRay Covid19 Detect 🦠🤖")
st.sidebar.markdown("""
Welcome to **XRay Covid19 Detect** – a diagnostic AI tool for classifying chest X-ray images into one of three categories:

- 🟢 **Normal**
- 🟡 **Viral Pneumonia**
- 🔴 **COVID-19**
""")

# === Main ===
# Step 1: Select model first, default value is empty
model_dirs = [d for d in os.listdir(MODEL_ROOT) if os.path.isdir(os.path.join(MODEL_ROOT, d))]
model_name = st.sidebar.selectbox("Select a model", [""] + model_dirs)  # Empty option at the start

if model_name:
    model_folder = os.path.join(MODEL_ROOT, model_name)

    # Step 2: Load model evaluation results
    report_path = os.path.join(model_folder, f"{model_name}_report.json")
    cm_path = os.path.join(model_folder, f"{model_name}_confusion_matrix.png")
    lc_path = os.path.join(model_folder, f"{model_name}_learning_curve.png")
    loss_path = os.path.join(model_folder, f"{model_name}_loss_curve.png")

    # Display evaluation results if they exist
    if os.path.exists(report_path):
        with open(report_path) as f:
            report_data = json.load(f)

        st.subheader("Classification Report", anchor="classification-report")
        st.markdown(f"**Accuracy:** {report_data['accuracy']:.2f}", unsafe_allow_html=True)

        class_report = report_data['classification_report']
        rows = []

        for label, metrics in class_report.items():
            if isinstance(metrics, dict):
                # Try to map class index to label
                label_name = label_map.get(int(label), label) if label.isdigit() else label
                rows.append({
                    "Class": label_name,
                    "Precision": round(metrics.get("precision", 0), 2),
                    "Recall": round(metrics.get("recall", 0), 2),
                    "F1-Score": round(metrics.get("f1-score", 0), 2),
                    "Support": int(metrics.get("support", 0))
                })

        df_report = pd.DataFrame(rows)
        st.dataframe(df_report.set_index("Class"))

    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix", anchor="confusion-matrix")
        st.image(cm_path)

    if os.path.exists(lc_path):
        st.subheader("Learning Curve", anchor="learning-curve")
        st.image(lc_path)

    if os.path.exists(loss_path):
        st.subheader("Loss Curve", anchor="loss-curve")
        st.image(loss_path)

    # Add a horizontal line after the last curve and then increase the font size for the next content
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 32px;'>Prediction</h2>", unsafe_allow_html=True)

    # Step 3: Allow user to upload an image for prediction
    image_file = st.file_uploader("Upload an X-ray image", type=["png", "jpg", "jpeg"])

    if image_file:
        # Show the uploaded image
        st.image(image_file, caption="Uploaded Image", width=250)
        img = Image.open(image_file).convert("L")  # convert to grayscale
        img = img.resize((64, 64))
        img_array = np.array(img).reshape(1, -1)

        # Load the model and make predictions
        model_path = os.path.join(model_folder, f"{model_name}.pkl")
        model = joblib.load(model_path)
        pred_class_idx = model.predict(img_array)[0]
        pred_class_label = label_map.get(pred_class_idx, f"Unknown ({pred_class_idx})")

        # Display the prediction result with appropriate color and font size
        if pred_class_label == 'Normal':
            st.markdown(f"<span style='color: green; font-size: 30px;'>Predicted Class: **{pred_class_label}**</span>", unsafe_allow_html=True)
        elif pred_class_label == 'Viral Pneumonia':
            st.markdown(f"<span style='color: yellow; font-size: 30px;'>Predicted Class: **{pred_class_label}**</span>", unsafe_allow_html=True)
        elif pred_class_label == 'Covid':
            st.markdown(f"<span style='color: red; font-size: 30px;'>Predicted Class: **{pred_class_label}**</span>", unsafe_allow_html=True)

