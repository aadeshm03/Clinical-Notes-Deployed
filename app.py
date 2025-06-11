import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Label map
label_map = {"fracture": 0, "osteomalacia": 1, ...}
id_to_label = {v: k for k, v in label_map.items()}

# Streamlit UI
st.title("🩺 Clinical Note Diagnosis Predictor")
note = st.text_area("Enter clinical note text:")

if st.button("Predict"):
    inputs = tokenizer(note, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = logits.argmax(dim=1).item()
        pred_label = id_to_label[pred_id]
    st.success(f"Predicted Diagnosis: **{pred_label}**")
