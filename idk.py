import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ----- Simulate realistic dataset -----
@st.cache_data
def create_dataset():
    np.random.seed(42)
    diseases = ['Hemophilia', 'Thalassemia', 'Fanconi Anemia', 'Sickle Cell Anemia', 'No Disease']
    data = []
    for _ in range(200):
        d = np.random.choice(diseases, p=[0.20, 0.20, 0.15, 0.20, 0.25])

        # Numeric test ranges and symptom probabilities per disease
        if d == 'Hemophilia':
            hb = np.random.normal(13, 1)
            wbc = np.random.normal(6, 1)
            platelet = np.random.normal(180, 40)
            mcv = np.random.normal(90, 5)
            retic = np.random.normal(2, 0.5)

            bruising = 1
            fatigue = np.random.choice([0, 1], p=[0.3, 0.7])
            joint_pain = np.random.choice([0, 1], p=[0.4, 0.6])
            pale_skin = np.random.choice([0, 1], p=[0.6, 0.4])
            short_breath = np.random.choice([0, 1], p=[0.7, 0.3])
            dizziness = np.random.choice([0, 1], p=[0.6, 0.4])
            cold_hands = np.random.choice([0, 1], p=[0.7, 0.3])
            infections = np.random.choice([0, 1], p=[0.9, 0.1])

        elif d == 'Thalassemia':
            hb = np.random.normal(8, 1.5)
            wbc = np.random.normal(7, 1)
            platelet = np.random.normal(260, 50)
            mcv = np.random.normal(70, 5)
            retic = np.random.normal(4, 1)

            bruising = 0
            fatigue = 1
            joint_pain = 0
            pale_skin = 1
            short_breath = 1
            dizziness = 1
            cold_hands = np.random.choice([0, 1], p=[0.5, 0.5])
            infections = 0

        elif d == 'Fanconi Anemia':
            hb = np.random.normal(7.5, 1)
            wbc = np.random.normal(3, 1)
            platelet = np.random.normal(100, 30)
            mcv = np.random.normal(85, 7)
            retic = np.random.normal(1.5, 0.5)

            bruising = 1
            fatigue = 1
            joint_pain = 1
            pale_skin = 1
            short_breath = 1
            dizziness = 1
            cold_hands = 1
            infections = 1

        elif d == 'Sickle Cell Anemia':
            hb = np.random.normal(8.5, 1)
            wbc = np.random.normal(10, 2)
            platelet = np.random.normal(300, 60)
            mcv = np.random.normal(85, 6)
            retic = np.random.normal(6, 1.5)

            bruising = 0
            fatigue = 1
            joint_pain = 1
            pale_skin = 1
            short_breath = 1
            dizziness = 1
            cold_hands = np.random.choice([0, 1], p=[0.3, 0.7])
            infections = 0

        else:
            hb = np.random.normal(14, 1)
            wbc = np.random.normal(7, 1.5)
            platelet = np.random.normal(250, 50)
            mcv = np.random.normal(90, 5)
            retic = np.random.normal(2, 0.5)

            bruising = 0
            fatigue = 0
            joint_pain = 0
            pale_skin = 0
            short_breath = 0
            dizziness = 0
            cold_hands = 0
            infections = 0

        hb = np.clip(hb, 4, 18)
        wbc = np.clip(wbc, 1, 20)
        platelet = np.clip(platelet, 20, 1000)
        mcv = np.clip(mcv, 50, 120)
        retic = np.clip(retic, 0.5, 10)

        data.append([hb, wbc, platelet, mcv, retic,
                     bruising, fatigue, joint_pain, pale_skin, short_breath, dizziness, cold_hands, infections, d])

    cols = ['Hb', 'WBC', 'Platelet', 'MCV', 'Reticulocyte',
            'Bruising', 'Fatigue', 'JointPain', 'PaleSkin', 'ShortBreath',
            'Dizziness', 'ColdHands', 'Infections', 'Disease']
    df = pd.DataFrame(data, columns=cols)
    return df

@st.cache_resource(show_spinner=False)
def train_model():
    df = create_dataset()
    X = df.drop('Disease', axis=1)
    y = df['Disease']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    return model, le, report

def welcome_page():
    st.markdown("<h1 style='text-align:center; color:#4B0082;'>Rare Blood Diseases AI Predictor</h1>", unsafe_allow_html=True)
    st.write("""
    Welcome! This app predicts rare blood diseases based on your blood test results and symptoms.
    
    **Created by Divyanu Mehta and Aripra Bhadani.**
    """)
    if st.button("Go to Disease Predictor"):
        st.session_state.page = "predictor"

def predictor_page():
    st.markdown("<h1 style='text-align:center; color:#006400;'>Disease Predictor</h1>", unsafe_allow_html=True)
    st.write("Enter your blood test results and select symptoms:")

    hb = st.number_input("Hemoglobin (g/dL)", 4.0, 18.0, 13.0, 0.1)
    wbc = st.number_input("White Blood Cell Count (10³/µL)", 1.0, 20.0, 7.0, 0.1)
    platelet = st.number_input("Platelet Count (10³/µL)", 20, 1000, 250, 1)
    mcv = st.number_input("Mean Corpuscular Volume (fL)", 50, 120, 90, 1)
    retic = st.number_input("Reticulocyte Count (%)", 0.5, 10.0, 2.0, 0.1)

    bruising = st.checkbox("Easy bruising or bleeding")
    fatigue = st.checkbox("Fatigue or weakness")
    joint_pain = st.checkbox("Joint pain or swelling")
    pale_skin = st.checkbox("Pale skin or pallor")
    short_breath = st.checkbox("Shortness of breath")
    dizziness = st.checkbox("Dizziness or lightheadedness")
    cold_hands = st.checkbox("Cold hands or feet")
    infections = st.checkbox("Frequent infections")

    input_data = np.array([[hb, wbc, platelet, mcv, retic,
                            int(bruising), int(fatigue), int(joint_pain), int(pale_skin), int(short_breath),
                            int(dizziness), int(cold_hands), int(infections)]])

    model, le, report = train_model()

    if st.button("Predict Disease"):
        pred_enc = model.predict(input_data)[0]
        pred = le.inverse_transform([pred_enc])[0]

        if pred == 'No Disease':
            st.success("No signs of the rare blood diseases considered here.")
        else:
            st.error(f"Predicted Disease: {pred}")

        # Warning notes for common conditions with specific names:
        warnings = []
        if hb < 12:
            warnings.append("Low Hemoglobin (possible anemia or iron deficiency). Please consult your doctor.")
        if wbc < 4 or wbc > 11:
            warnings.append("Abnormal White Blood Cell count (possible infection or immune disorder).")
        if platelet < 150 or platelet > 450:
            warnings.append("Abnormal Platelet count (possible thrombocytopenia or thrombocytosis).")
        if mcv < 80 or mcv > 100:
            warnings.append("Abnormal MCV (possible microcytic or macrocytic anemia).")
        if retic < 0.5 or retic > 2.5:
            warnings.append("Abnormal Reticulocyte count (possible bone marrow response issues or hemolysis).")

        if warnings:
            st.warning(
                "**Note:** " + " ".join(warnings) +
                "\n\nThis app predicts rare blood diseases but abnormal values may have many other causes. Always seek medical advice for proper diagnosis."
            )

    if st.button("Learn about Diseases"):
        st.session_state.page = "info"

    with st.expander("Model Accuracy & Details"):
        st.write("Model trained on simulated data — not for medical diagnosis!")
        accuracy = 100 * sum([report[cls]['precision'] for cls in report if cls in le.classes_]) / len(le.classes_)
        st.write(f"Approximate precision average on test data: {accuracy:.2f}%")
        st.write("Classification report:")
        st.json(report)

def info_page():
    st.markdown("<h1 style='text-align:center; color:#800000;'>Information About Diseases</h1>", unsafe_allow_html=True)
    st.write("""
    ### Hemophilia  
    A genetic disorder causing blood clotting problems leading to easy bleeding and bruising.

    ### Thalassemia  
    An inherited anemia with low hemoglobin and smaller red blood cells (low MCV).

    ### Fanconi Anemia  
    A rare genetic disorder causing bone marrow failure, low blood counts, and immune deficiencies.

    ### Sickle Cell Anemia  
    Genetic disorder causing abnormally shaped red blood cells, painful crises, and anemia.

    """)
    if st.button("Back to Welcome"):
        st.session_state.page = "welcome"

def main():
    if "page" not in st.session_state:
        st.session_state.page = "welcome"

    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "predictor":
        predictor_page()
    elif st.session_state.page == "info":
        info_page()

if __name__ == "__main__":
    main()



