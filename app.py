import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ========================= MODEL LOADING =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_disease_model.keras")

model = load_model()

# ========================= PREDICTION FUNCTION =========================
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    result_index = int(np.argmax(prediction))
    return result_index

# ========================= AI HEALTH BOT (Disease Info) =========================
def disease_info(disease_name):
    info = {
        "BA- cellulitis": "Bacteremia-associated (BA) cellulitis is a serious complication where a skin infection spreads to the bloodstream. While cellulitis is typically localized, systemic symptoms like fever, chills, and fatigue can indicate bacteremia, a potentially life-threatening condition. It is a rare complication, occurring in only about 5–10% of cases, but is more common in immunocompromised patients, those with diabetes, and the elderly. If left untreated, it can lead to further complications like sepsis, endocarditis, or osteomyelitis.",
        "BA-impetigo": "Impetigo is a highly contagious bacterial skin infection that most commonly affects infants and young children, often appearing around the nose, mouth, hands, and feet. The sores quickly rupture, ooze fluid, and develop a characteristic honey-colored crust. The infection is primarily caused by Staphylococcus aureus or Streptococcus pyogenes bacteria, which can enter the body through a cut, scrape, or insect bite. Treatment with antibiotics, either topical or oral, is used to clear the infection and limit its spread",
        "FU-athlete-foot": "Athlete's foot, or tinea pedis, is a contagious fungal infection that typically causes an itchy, scaly, and burning rash, most commonly between the toes. It thrives in warm, moist environments like sweaty shoes and socks, and can spread through contact with infected people or surfaces in public places like pools and locker rooms. The infection can also cause dry, cracked skin on the soles or blisters on the feet. Treatment usually involves over-the-counter or prescription antifungal creams, powders, or sprays.",
        "FU-nail-fungus": "Nail fungus, or onychomycosis, is a common infection that makes nails thick, discolored, and brittle. It is most often caused by a type of fungus called dermatophyte, and thrives in warm, moist environments like shoes and public showers. The infection can lead to misshapen, crumbling nails and may emit a foul odor. While generally not serious for healthy individuals, it can be persistent, difficult to treat, and poses a risk of further infection for those with diabetes or weakened immune systems.",
        "FU-ringworm": "Ringworm is a common, contagious fungal infection of the skin, not caused by a worm. It appears as a red, itchy, ring-shaped rash, but symptoms can vary and may include scaly, raised patches. It spreads through direct contact with an infected person, animal, or contaminated surfaces like clothing and towels, and is treated with antifungal medication.",
        "PA-cutaneous-larva-migrans": "Cutaneous larva migrans (CLM), or 'creeping eruption' is a parasitic skin infection caused by hookworm larvae. Humans are accidental hosts who become infected through direct skin contact with warm, moist soil or sand contaminated with animal feces. The larvae, most commonly from dog and cat hookworms, burrow into the skin but cannot penetrate past the outer layer. This migration causes intensely itchy, red, winding tracks on the skin, typically on the feet, legs, or buttocks.",
        "VI-chickenpox": "Chickenpox is a highly contagious viral infection caused by the varicella-zoster virus (VZV). It is characterized by an itchy rash of fluid-filled blisters that eventually scab over, accompanied by symptoms like fever and fatigue. While typically mild in children, it can cause serious complications in adults, pregnant women, and those with weakened immune systems.",
        "VI-shingles": "Shingles is a reactivation of the chickenpox virus causing painful rashes and nerve pain on one side of the body. VI, or the abducens nerve, is rarely affected by shingles, a viral infection caused by the varicella-zoster virus. This condition, a complication of herpes zoster ophthalmicus (HZO), can lead to abducens nerve palsy. The palsy causes weakness or paralysis of the lateral rectus muscle, leading to an impaired ability to move the eye outwards. Symptoms include horizontal double vision and the inability to abduct the affected eye. While recovery is common, the diplopia can persist for weeks or months after the initial rash has subsided.",
        "Actinic Keratosis" : "An actinic keratosis (ak-TIN-ik ker-uh-TOE-sis) is a rough, scaly patch on the skin that develops from years of sun exposure. It's often found on the face, lips, ears, forearms, scalp, neck or back of the hands,, increase the risk of developing skin cancer.",
        "Bowen Disease" : "Bowen disease is a pre-cancerous skin condition with a low risk of progressing to invasive squamous cell carcinoma (SCC), estimated at 3%–5%. Key risk factors for developing the condition include excessive sun exposure, fair skin, older age, and a weakened immune system. While the prognosis is generally excellent with treatment, the lesions can be progressive, and if they become invasive, one-third may potentially metastasize."
    }
    return info.get(disease_name, "Information not available for this disease.")

# ========================= APP INTERFACE =========================
st.sidebar.title("🩺 Skin Disease Recognition System")
app_mode = st.sidebar.radio(
    "Navigation",
    ["Home", "Disease Recognition", "AI Health Bot", "Categories", "Developers Group", "About Project"]
)

# ========================= HOME PAGE =========================
if app_mode == "Home":
    st.title("🏠 Welcome to the Skin Disease Prediction System")
    st.image("/content/sample_image.jpg", width='stretch')
    st.markdown("""
    ### 👋 Introduction
    Welcome to the **AI-powered Skin Disease Recognition System**.
    This platform helps you **detect skin diseases** using deep learning.

    🔍 Upload an image, get an instant prediction, and learn about the disease.
    """)

# ========================= DISEASE RECOGNITION PAGE =========================
elif app_mode == "Disease Recognition":
    st.title("🧬 Disease Recognition")
    test_image = st.file_uploader("📤 Upload an Image:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        image = Image.open(test_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width='stretch')

        if st.button("🔎 Predict Disease"):
            with st.spinner("Model analyzing image..."):
                result_index = model_prediction(test_image)
                class_name = [
                    'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
                    'FU-nail-fungus', 'FU-ringworm',
                    'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles',
                    'Actinic Keratosis', 'Bowen Disease'
                ]
                predicted_disease = class_name[result_index]
                st.success(f"🩸 The model predicts: **{predicted_disease}**")

                # Show disease info automatically
                st.subheader("💬 AI Health Bot Summary:")
                st.info(disease_info(predicted_disease))

# ========================= AI HEALTH BOT =========================
elif app_mode == "AI Health Bot":
    st.title("🤖 Diagnose AI")
    st.markdown("""
    Type the disease name below to learn about it.
    Example: *ringworm*, *cellulitis*, *shingles*, etc.
    """)
    query = st.text_input("Enter Disease Name:")
    if query:
        found = [key for key in [
            'BA- cellulitis', 'BA-impetigo', 'FU-athlete-foot',
            'FU-nail-fungus', 'FU-ringworm', 'PA-cutaneous-larva-migrans',
            'VI-chickenpox', 'VI-shingles',
            'Actinic Keratosis', 'Bowen Disease'
        ] if query.lower() in key.lower()]
        if found:
            st.success(disease_info(found[0]))
        else:
            st.warning("❌ Disease not found. Please check spelling.")

# ========================= CATEGORIES =========================
elif app_mode == "Categories":
    st.title("📂 Disease Categories")
    st.markdown("""
    Diseases are classified into:
    - 🦠 **Bacterial** (BA): *Cellulitis*, *Impetigo*
    - 🍄 **Fungal** (FU): *Athlete's Foot*, *Ringworm*, *Nail Fungus*
    - 🪱 **Parasitic** (PA): *Cutaneous Larva Migrans*
    - 🧫 **Viral** (VI): *Chickenpox*, *Shingles*
    """)

# ========================= DEVELOPERS GROUP =========================
elif app_mode == "Developers Group":
    st.title("👩‍💻 Developers Group")
    st.markdown("Meet the minds behind this project 👇")

    cols = st.columns(5)
    devs = [
        ("Dev 3", "ML Developer and Data Analysrt", "Prepared and augmented datasets."),
    ]
    for i, (name, role, desc) in enumerate(devs):
        with cols[i]:
            st.subheader(f"👤 {name}")
            st.write(f"**Role:** {role}")
            st.caption(desc)

# ========================= ABOUT PAGE =========================
elif app_mode == "About Project":
    st.title("ℹ️ About Project")
    st.markdown("""
    ### 🧠 Overview
    This system uses **Deep Learning (CNN)** to classify skin diseases from images.
    It helps early detection and awareness for common infections.

    **Dataset:** Augmented skin disease dataset
    **Framework:** TensorFlow + Streamlit
    **Goal:** Build accessible healthcare AI for all 🌍
    """)

