import streamlit as st
import numpy as np
import pickle

# Load model + scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Human Activity Clustering App 🧍‍♂️")

st.write("Enter sensor values to predict activity cluster")

# Example: 561 features (HAR dataset)
# Simplified input (demo ke liye few inputs le rahe)
inputs = []

for i in range(10):  # demo ke liye 10 inputs (real me 561 hote hain)
    val = st.number_input(f'Feature {i+1}', value=0.0)
    inputs.append(val)

# Convert to numpy
input_array = np.array(inputs).reshape(1, -1)

# ⚠️ IMPORTANT (match features count)
# Agar 561 features hain, to dummy padding karenge
if input_array.shape[1] < scaler.n_features_in_:
    diff = scaler.n_features_in_ - input_array.shape[1]
    input_array = np.pad(input_array, ((0,0),(0,diff)), 'constant')

# Scale
input_scaled = scaler.transform(input_array)

# Predict
if st.button("Predict Cluster"):
    result = model.predict(input_scaled)
    st.success(f"Predicted Cluster: {result[0]}")