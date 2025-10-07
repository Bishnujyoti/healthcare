import streamlit as st
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --------------------------------------------
# 💡 Author Credit
"""
Health Symptom Checker
Author: BG 😏
Original Work © 2025 BG. All rights reserved.
Unauthorized copying or redistribution is prohibited.
"""
# --------------------------------------------

# --- Rule-based disease library ---
DISEASE_DB: Dict[str, Dict] = {
    "Common Cold": {
        "symptoms": ["sneezing", "runny nose", "stuffy nose", "sore throat", "mild cough", "mild fever"],
        "advice": "Rest, stay hydrated, use over-the-counter remedies for symptoms, avoid close contact with vulnerable people.",
        "notes": "Usually mild and improves in ~7–10 days."
    },
    "Influenza (Flu)": {
        "symptoms": ["fever", "chills", "body aches", "fatigue", "headache", "dry cough", "sore throat"],
        "advice": "Rest, fluids, consider antiviral treatment if within 48 hours and high-risk. See a clinician if symptoms worsen.",
        "notes": "Can be more severe than a common cold, especially in elderly or those with chronic illness."
    },
    "COVID-19 (possible)": {
        "symptoms": ["fever", "dry cough", "loss of taste", "loss of smell", "shortness of breath", "fatigue", "sore throat"],
        "advice": "Isolate until tested/cleared, consider testing, monitor breathing; seek medical attention if breathing difficulty develops.",
        "notes": "Follow local public health guidance for testing and isolation."
    },
    "Migraine": {
        "symptoms": ["headache", "throbbing headache", "nausea", "sensitivity to light", "sensitivity to sound"],
        "advice": "Rest in a dark quiet room, use prescribed migraine meds if you have them, see a doctor if new severe headaches occur.",
        "notes": "Sudden, very severe headache may require urgent evaluation."
    },
    "General Headache": {
        "symptoms": ["mild headache", "tension headache", "head pain", "stress headache"],
        "advice": "Hydrate, rest, manage stress. Use mild pain relievers if needed. See a doctor if headaches are frequent or severe.",
        "notes": "Most headaches are not serious, but persistent or sudden severe headaches should be evaluated."
    },
    "Muscle or Body Pain": {
        "symptoms": ["muscle pain", "body aches", "back pain", "neck pain", "joint pain"],
        "advice": "Rest, gentle stretching, over-the-counter pain relievers. If pain is persistent or severe, seek medical advice.",
        "notes": "Can be caused by strain, flu, or other conditions."
    },
    "Stomachache": {
        "symptoms": ["stomach pain", "abdominal pain", "bloating", "indigestion"],
        "advice": "Avoid heavy meals, rest, hydrate. Seek care if severe, persistent, or associated with vomiting blood or black stools.",
        "notes": "Common and often self-limited, but may signal more serious illness."
    },
    "Urinary Tract Infection (UTI)": {
        "symptoms": ["painful urination", "burning urination", "frequent urination", "lower abdominal pain", "cloudy urine"],
        "advice": "Drink fluids, see a clinician for urine testing and antibiotics if a UTI is suspected.",
        "notes": "Left untreated, can ascend and cause kidney infection."
    },
    "Gastroenteritis / Food Poisoning": {
        "symptoms": ["nausea", "vomiting", "diarrhea", "stomach cramps", "fever"],
        "advice": "Stay hydrated (oral rehydration), rest. Seek care if unable to keep fluids down, blood in stool, severe dehydration, or high fever.",
        "notes": "Often self-limited but can be serious in very young, elderly, or immunocompromised."
    },
    "Allergic Rhinitis (Allergy)": {
        "symptoms": ["sneezing", "itchy eyes", "runny nose", "watery eyes", "nasal congestion"],
        "advice": "Avoid triggers, consider antihistamines or nasal steroid sprays. See an allergist if persistent.",
        "notes": "Usually no fever."
    }
}

EMERGENCY_SYMPTOMS = [
    "trouble breathing", "shortness of breath", "difficulty breathing",
    "chest pain", "pressure in chest", "sudden confusion",
    "loss of consciousness", "unresponsiveness", "bluish lips", "bluish face",
    "severe bleeding", "severe head injury", "sudden severe weakness on one side"
]

# --- NLP-based matcher ---
def match_symptoms(user_symptoms: List[str], threshold: float = 0.4):
    """Match user symptoms to disease symptoms using cosine similarity."""
    all_known_symptoms = [s for d in DISEASE_DB.values() for s in d["symptoms"]]
    vectorizer = TfidfVectorizer().fit(all_known_symptoms + user_symptoms)

    matched = {}
    for disease, info in DISEASE_DB.items():
        score = 0
        matched_symptoms = []
        for us in user_symptoms:
            vec_user = vectorizer.transform([us])
            vec_known = vectorizer.transform(info["symptoms"])
            sims = cosine_similarity(vec_user, vec_known)
            max_sim = np.max(sims)
            if max_sim >= threshold:
                score += 1
                matched_symptoms.append(info["symptoms"][np.argmax(sims)])
        matched[disease] = {"score": score, "matched": matched_symptoms}
    return matched

def detect_emergency(user_symptoms: List[str]) -> List[str]:
    """Check if emergency symptoms are reported (substring match)."""
    found = []
    for s in user_symptoms:
        for em in EMERGENCY_SYMPTOMS:
            if em in s or s in em:
                found.append(em)
    return list(set(found))

# --- Streamlit App ---
try:
    st.set_page_config(page_title="Health Symptom Checker", page_icon="🩺", layout="wide")
except Exception:
    pass

# --- Sidebar Credit (persistent) ---
with st.sidebar:
    st.markdown("<h4>About this app</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:14px;'>This code was written by <b>BG</b> 😏</div>", unsafe_allow_html=True)
    st.markdown("---")

# --- Unified Styling ---
st.markdown(
    """
    <style>
        body, .main {
            background-color: #fdf6f9 !important;
            color: #2d2d2d !important;
        }
        .stTextArea textarea {
            background-color: #f0f9ff !important;
            color: #222 !important;
            border-radius: 12px;
            padding: 12px;
            font-size: 16px;
            border: 1px solid #d0e7ff;
        }
        .stButton button {
            background: linear-gradient(90deg, #a5b4fc, #c7d2fe);
            color: #222 !important;
            font-weight: 600;
            padding: 0.6em 1.2em;
            border-radius: 10px;
            border: none;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #93c5fd, #bfdbfe);
        }
        .result-card {
            background: #fff7ed;
            color: #2d2d2d !important;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 15px;
        }
        h3 {
            color: #374151;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Health Symptom Checker")
st.write("Enter your symptoms below (comma-separated). Example: `fever, sore throat, headache`")

user_input = st.text_area("Your symptoms:", height=100, placeholder="e.g., fever, headache, sore throat")

col1, col2 = st.columns([1,3])
with col1:
    check_btn = st.button("Check Symptoms")

if check_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one symptom.")
    else:
        user_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]

        # Emergency check
        emergencies = detect_emergency(user_symptoms)
        if emergencies:
            st.error("🚨 EMERGENCY WARNING: You reported symptom(s) that may require urgent care: " + ", ".join(emergencies))
            st.info("If someone has trouble breathing, severe chest pain, fainting, severe bleeding, or sudden confusion, call emergency services immediately.")

        # Match diseases
        results = match_symptoms(user_symptoms)
        sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

        if sorted_results[0][1]["score"] == 0:
            st.warning("❌ No strong match found for your symptoms.")
            st.info("Monitor your health, try self-care for mild symptoms, and seek medical advice if symptoms persist or worsen.")
        else:
            st.subheader("🔎 Possible Matches:")
            colors = ["#fff7ed", "#f0f9ff", "#fef9c3"]
            for i, (disease, info) in enumerate(sorted_results[:3]):
                if info["score"] > 0:
                    color = colors[i % len(colors)]
                    st.markdown(f"<div class='result-card' style='background:{color};'><h3>{disease}</h3>", unsafe_allow_html=True)
                    st.write(f"**Matched symptoms:** {', '.join(info['matched'])}")
                    st.write(f"**Advice:** {DISEASE_DB[disease]['advice']}")
                    if DISEASE_DB[disease].get('notes'):
                        st.caption(f"Note: {DISEASE_DB[disease]['notes']}")
                    st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.markdown(
            """<p style='font-size:14px; color:#555;'>⚠️ <b>Disclaimer:</b> This tool is for educational purposes only. It is <b>not a medical diagnosis</b>. Always consult a healthcare professional for medical concerns.</p>""",
            unsafe_allow_html=True
        )

        # --- 💡 Author credit shown after each search ---
        st.markdown(
            """
            <div style="
                margin-top:12px;
                padding:10px;
                border-radius:8px;
                text-align:center;
                background:linear-gradient(90deg, rgba(245,245,245,0.9), rgba(255,255,255,0.9));
                border:1px solid #eee;
            ">
                <strong>this code was written by BG 😏</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
