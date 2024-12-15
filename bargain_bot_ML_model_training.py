import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Existing functions
def train_budget_model(historical_data):
    features = historical_data[['duration', 'resources', 'expertise']]
    target = historical_data['budget']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    st.write(f"Model Trained with RMSE: {rmse:.2f}")
    return model

def generate_historical_data(num_samples):
    durations = np.random.randint(1, 20, num_samples)
    resources = np.random.randint(1, 20, num_samples)
    expertise = np.random.randint(1, 20, num_samples)
    noise = np.random.normal(0, 100, num_samples)
    budgets = (durations ** 1.5 * 50) + (resources * 150) + (np.log1p(expertise) * 300) + noise
    return pd.DataFrame({'duration': durations, 'resources': resources, 'expertise': expertise, 'budget': budgets})

def forecast_budget_with_model(model, project_details):
    features = np.array([project_details['duration'], project_details['resources'], project_details['expertise']]).reshape(1, -1)
    return model.predict(features)[0]

def suggest_price(complexity_score, estimated_budget):
    weights = np.array([0.4, 0.6])
    inputs = np.array([complexity_score, estimated_budget])
    return np.dot(weights, inputs)

def adjust_price(current_price, feedback):
    adjustment = {'positive': 0.95, 'negative': 1.1, 'neutral': 1.0}
    return current_price * adjustment.get(feedback, 1.0)

# Streamlit app
def main():
    st.title("BargainBot")
    st.sidebar.header("Navigation")
    options = st.sidebar.selectbox("Choose an option:", ["Home", "Train Model", "Estimate Budget", "Suggest Price", "Adjust Price"])
    
    if options == "Home":
        st.write("Welcome to BargainBot! A smart tool for budget estimation and price suggestions.")
        st.write("Navigate through the options in the sidebar.")
    
    elif options == "Train Model":
        st.header("Train a Model")
        num_samples = st.number_input("Enter the number of data samples to generate:", min_value=100, max_value=5000, value=1000)
        if st.button("Generate Data and Train"):
            historical_data = generate_historical_data(num_samples)
            st.write("Generated Data:", historical_data.head())
            global model
            model = train_budget_model(historical_data)
    
    elif options == "Estimate Budget":
        st.header("Estimate Project Budget")
        duration = st.number_input("Enter project duration (weeks):", min_value=1, max_value=52, value=10)
        resources = st.number_input("Enter number of resources required:", min_value=1, max_value=100, value=5)
        expertise = st.number_input("Enter expertise level (1-10):", min_value=1, max_value=10, value=5)
        if st.button("Estimate Budget"):
            if 'model' in globals():
                project_details = {"duration": duration, "resources": resources, "expertise": expertise}
                budget = forecast_budget_with_model(model, project_details)
                st.write(f"Estimated Budget: ${budget:.2f}")
            else:
                st.error("Please train the model first!")
    
    elif options == "Suggest Price":
        st.header("Get a Price Suggestion")
        complexity_score = st.number_input("Enter the complexity score:", min_value=0.0, value=50.0)
        estimated_budget = st.number_input("Enter the estimated budget:", min_value=0.0, value=1000.0)
        if st.button("Suggest Price"):
            price = suggest_price(complexity_score, estimated_budget)
            st.write(f"Suggested Price: ${price:.2f}")
    
    elif options == "Adjust Price":
        st.header("Adjust Price Based on Feedback")
        current_price = st.number_input("Enter the current price:", min_value=0.0, value=1000.0)
        feedback = st.selectbox("Select feedback type:", ["positive", "negative", "neutral"])
        if st.button("Adjust Price"):
            adjusted_price = adjust_price(current_price, feedback)
            st.write(f"Adjusted Price: ${adjusted_price:.2f}")

if __name__ == "__main__":
    main()
