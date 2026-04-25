import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="shashwativatsa/prediction-model", filename="best_prediction_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI to Forecast Sales Revenue
st.title("Forecast Sales Revenue Application")
st.write("The Forecast Sales Revenue App is an internal tool for company staff that predicts future sales revenue based on historical data, industry trends, and the status of the current sales pipeline, to estimate weekly, monthly, quarterly, and annual sales totals.")
st.write("Kindly enter the product and store details to forecast the sales revenue.")

# Collect user input
Product_Weight           = st.number_input("Product Weight in gms", min_value=0.0, max_value=100000.0, step=1.0, value=100.0)
Product_Sugar_Content    = st.selectbox("Sugar Content in Product", ["No Sugar", "Low Sugar", "Reg", "Regular"])
Product_Allocated_Area   = st.number_input("Product Allocated Area", min_value=0.000, max_value=1.000, step=0.001, value=0.005)
Product_Type             = st.selectbox("Product Type", ["Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods", "Fruits and Vegitables", "Hard Drinks", "Health and Hygiene", "Household", "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"])
Product_MRP              = st.number_input("Product MRP", min_value=0.00, max_value=10000.00, step=1.0, value=50.00)
Store_Establishment_Year = st.selectbox("Store Establishment Year", ["1987", "1998", "1999", "2009"])
Store_Size               = st.selectbox("Store Size", ["High", "Medium", "Small"])
Store_Location_City_Type = st.selectbox("City Type (where store is located)", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type               = st.selectbox("Store Type", ["Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Weight': Product_Weight,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_Type': Product_Type,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.write(f"Based on the information provided, the forcasted sales revenue is likely to {prediction}.")
