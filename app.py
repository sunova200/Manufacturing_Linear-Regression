import streamlit as st
import pickle
import pandas as pd

# Load the saved model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the Streamlit app title and description
st.set_page_config(page_title="Parts Per Hour Predictor", layout="wide")
st.title('Parts Per Hour Predictor 💡')
st.markdown("""
    Predict the number of parts produced per hour based on manufacturing parameters like Injection Temperature,
    Injection Pressure, Material Viscosity, and Mold Temperature. Adjust the sliders below and get the prediction.
""")

# Custom CSS for styling the app
st.markdown("""
    <style>
        /* Main layout styling */
        .stApp {
            background: linear-gradient(135deg, #ff6f61, #003366);
        }

        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #f0f4f7;
            border-radius: 15px;
            padding: 20px;
        }
        
        .css-1d391kg {
            background-color: #003366;
            color: white;
            border-radius: 10px;
            padding: 20px;
            font-size: 26px;
            font-weight: bold;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #ff6f61;
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 16px;
            transition: transform 0.3s ease;
        }

        .stButton>button:hover {
            transform: scale(1.05);
        }

        /* Title and header Styling */
        .css-1v3fvcr {
            font-size: 28px;
            font-weight: bold;
            color: #003366;
        }
        
        /* Input sliders and containers styling */
        .stSlider {
            margin-top: 20px;
            margin-bottom: 30px;
        }
        
        .stText {
            font-size: 18px;
            color: #003366;
            font-weight: bold;
        }
        
        .stRadio, .stCheckbox {
            font-size: 16px;
            color: #003366;
        }

        /* Prediction Box Styling */
        .stContainer {
            background-color: #f7f7f7;
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Create input widgets for the user
st.sidebar.header('Input Parameters')
def user_input_features():
    st.sidebar.markdown("### Adjust the sliders to input the manufacturing parameters:")
    injection_temp = st.sidebar.slider('Injection Temperature (°C)', 150.0, 300.0, 220.0)
    injection_pressure = st.sidebar.slider('Injection Pressure (bar)', 50.0, 250.0, 130.0)  # Updated range
    material_viscosity = st.sidebar.slider('Material Viscosity (cP)', 100.0, 600.0, 350.0)  # Updated range
    mold_temp = st.sidebar.slider('Mold Temperature (°C)', 30.0, 120.0, 80.0)  # New parameter

    # Return the user input as a DataFrame
    data = {
        'Injection_Temperature': injection_temp,
        'Injection_Pressure': injection_pressure,
        'Material_Viscosity': material_viscosity,
        'Mold_Temperature': mold_temp  # Include new parameter
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_input = user_input_features()

# Display the user inputs
st.subheader('User Input Parameters')
st.write(df_input)

# Add a separator for better readability
st.markdown("---")

# Align input with model training features
df_input = df_input.reindex(columns=model.feature_names_in_, fill_value=0)

# Debug info (optional)
# st.write("Expected features:", model.feature_names_in_)
# st.write("Input features after alignment:", df_input.columns.tolist())

# Make a prediction
prediction = model.predict(df_input)

# Display the prediction in a modern format inside a container
with st.container():
    st.markdown("""
        <div class="stContainer">
            <h3 style="color:#003366;">Predicted Parts Per Hour 🎯</h3>
            <h4 style="color:#ff6f61;">{:.2f} parts per hour</h4>
        </div>
    """.format(prediction[0]), unsafe_allow_html=True)

# Optional: Add some extra information or tips
st.markdown("""
    ### How this prediction works:
    - The model uses the features it was trained on (check your training script).
    - If you added **Mold Temperature** but the model wasn’t trained with it,
      the app will safely ignore it until you retrain the model with that feature.
""")

# Optional: Add a download button
st.download_button("Download Prediction Data", df_input.to_csv(index=False), "user_input_data.csv")
