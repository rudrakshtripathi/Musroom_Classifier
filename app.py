import streamlit as st
import pickle
import numpy as np
from readable_mappings import readable_feature_map

# Load model
with open("mushroom_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load categorical options
with open("categorical_options.pkl", "rb") as f:
    categorical_options = pickle.load(f)

# Streamlit page config
st.set_page_config(page_title="Mushroom Classifier 🍄", layout="wide")

# Title and description
st.markdown("<h1 style='text-align: center;'>🍄 Mushroom Edibility Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Know if your 🍄 is edible or poisonous before you take a bite 😅</h4>", unsafe_allow_html=True)
st.write("---")

# Sidebar input form
st.sidebar.title("🔍 Mushroom Features")
st.sidebar.info("Select each characteristic from the dropdowns below.")

input_data = {}

with st.sidebar.form("input_form"):
    for feature in categorical_options:
        if feature == "class":
            continue
        
        # Get readable options from mapping
        options = [(v, k) for k, v in readable_feature_map[feature].items()]
        display_names = [opt[0] for opt in options]

        selected_name = st.selectbox(f"{feature.replace('-', ' ').capitalize()}", display_names)
        
        # Get the actual letter code
        letter_code = dict(options)[selected_name]
        input_data[feature] = letter_code

    submitted = st.form_submit_button("Predict 🚀")

# Encoding function (no feature printout)
def encode_input(data_dict):
    encoded = []
    for feature, letter_code in data_dict.items():
        le = label_encoders.get(feature)
        if le is None:
            st.error(f"No label encoder found for feature: {feature}")
            return None
        try:
            encoded_value = le.transform([letter_code])[0]
            encoded.append(encoded_value)
        except Exception as e:
            st.error(f"Encoding error for feature {feature} with value {letter_code}: {e}")
            return None
    return np.array(encoded).reshape(1, -1)

# Prediction
if submitted:
    with st.spinner("Analyzing mushroom characteristics... 🧠"):
        try:
            input_encoded = encode_input(input_data)
            prediction = model.predict(input_encoded)[0]

            # Interpret prediction
            if prediction == 0 or prediction == 'e':
                st.markdown("""
                    <div style='background-color:#d4edda; padding:20px; border-radius:10px; text-align:center;'>
                        <h2 style='color:#155724;'>✅ This mushroom is <u>Edible</u>. Enjoy your meal!</h2>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background-color:#f8d7da; padding:20px; border-radius:10px; text-align:center;'>
                        <h2 style='color:#721c24;'>☠️ This mushroom is <u>Poisonous</u>. <b>Do NOT eat it!</b></h2>
                    </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")


st.write("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed using Streamlit | Machine Learning Project by G. Sai Sanjana</p>",
    unsafe_allow_html=True
)
