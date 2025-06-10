import streamlit as st
import joblib
import numpy as np
import os

# Load model and encoders
model = joblib.load("disease_model.pkl")
le = joblib.load("label_encoder.pkl")
symptoms_list = joblib.load("symptoms_list.pkl")

# Disease description
disease_info = {
    "COVID-19": "A viral infection causing fever, cough, fatigue, and breathing issues.",
    "Flu": "An infectious illness with fever, body aches, and sore throat.",
    "Dengue": "Mosquito-borne fever with rash, pain, and weakness.",
    "Common Cold": "A mild viral infection causing cough and runny nose.",
    "Typhoid": "A bacterial infection via contaminated food/water.",
    "Malaria": "Mosquito-borne disease with chills, fever, and sweating."
}

# Precautions and tablets
disease_precautions = {
    "COVID-19": {
        "precautions": ["Isolate yourself", "Wear a mask", "Drink warm water"],
        "tablets": ["Paracetamol", "Vitamin C", "Zinc supplement"]
    },
    "Flu": {
        "precautions": ["Stay hydrated", "Rest", "Avoid contact with others"],
        "tablets": ["Paracetamol", "Cough syrup", "Antiviral (if prescribed)"]
    },
    "Dengue": {
        "precautions": ["Avoid mosquito bites", "Use mosquito repellent"],
        "tablets": ["Paracetamol (no aspirin!)", "ORS", "Doctor visit mandatory"]
    },
    "Common Cold": {
        "precautions": ["Rest", "Drink warm fluids"],
        "tablets": ["Paracetamol", "Decongestant", "Cough drops"]
    },
    "Typhoid": {
        "precautions": ["Boil water", "Eat hygienic food"],
        "tablets": ["Antibiotics", "ORS", "Paracetamol"]
    },
    "Malaria": {
        "precautions": ["Sleep under mosquito nets", "Remove stagnant water"],
        "tablets": ["Antimalarial drugs", "Paracetamol", "Fluids"]
    }
}

# Awareness images (if using locally)
image_map = {
    "COVID-19": "images/covid.png",
    "Flu": "images/flu.png",
    "Dengue": "images/dengue.png",
    "Common Cold": "images/cold.png",
    "Typhoid": "images/typhoid.png",
    "Malaria": "images/malaria.png"
}

# Streamlit UI
st.set_page_config(page_title="AI Disease Predictor", layout="centered")
st.title("🩺 AI Disease Predictor")

# Symptom input
selected_symptoms = st.multiselect("Select your symptoms:", symptoms_list)
input_data = [1 if symptom in selected_symptoms else 0 for symptom in symptoms_list]

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        prediction = model.predict([input_data])[0]
        predicted_disease = le.inverse_transform([prediction])[0]

        st.success(f"🧬 Predicted Disease: **{predicted_disease}**")

        # Disease description
        description = disease_info.get(predicted_disease, "No info available.")
        st.info(f"📝 About the Disease:\n{description}")

        # Precautions
        precautions = disease_precautions.get(predicted_disease, {}).get("precautions", [])
        if precautions:
            st.subheader("🛡️ Precautions")
            for item in precautions:
                st.markdown(f"- {item}")

        # Tablets
        tablets = disease_precautions.get(predicted_disease, {}).get("tablets", [])
        if tablets:
            st.subheader("💊 Suggested Tablets / Treatment")
            for tab in tablets:
                st.markdown(f"- {tab}")

        # Awareness image
        img_path = image_map.get(predicted_disease)
        if img_path and os.path.exists(img_path):
            st.image(img_path, use_container_width=True, caption=f"{predicted_disease} Awareness Image")
        else:
            st.warning("📷 Awareness image not found for this disease.")

        # Nearby hospitals
        st.markdown("📍 [Click here to find Nearby Hospitals](https://www.google.com/maps/search/hospitals+near+me)")

# FAQ section
st.markdown("---")
with st.expander("💬 Frequently Asked Questions"):
    st.markdown("""
**Q: Can I take antibiotics for a viral disease?**  
❌ No. Antibiotics are for bacterial infections, not viruses like COVID-19 or flu.

**Q: Should I take tablets without doctor advice?**  
❌ No. Always consult a healthcare professional before taking medication.

**Q: How can I prevent dengue or malaria?**  
✅ Use mosquito nets, repellents, and remove stagnant water around you.

**Q: I have mild fever and sore throat. What should I do?**  
🤒 You may have a cold or flu. Rest, drink warm fluids, and monitor your symptoms.
""")
