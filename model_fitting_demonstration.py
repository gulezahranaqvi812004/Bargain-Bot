import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import streamlit as st

# Generate historical data
def generate_historical_data(num_samples):
    durations = np.random.randint(1, 20, num_samples)
    resources = np.random.randint(1, 20, num_samples)
    expertise = np.random.randint(1, 20, num_samples)

    duration_factor = 1.5
    duration_scale = 50
    resource_scale = 150
    expertise_scale = 300

    noise = np.random.normal(0, 100, num_samples)
    budgets = (
        (durations ** duration_factor * duration_scale)
        + (resources * resource_scale)
        + (np.log1p(expertise) * expertise_scale)
        + noise
    )

    historical_data = pd.DataFrame({
        'duration': durations,
        'resources': resources,
        'expertise': expertise,
        'budget': budgets
    })
    return historical_data

# Custom Simple Linear Regression Class
class SimpleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept column
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept column
        return X_b @ self.coefficients

# Plot predicted vs actual values
def plot_with_polynomial_fit(y_test, predictions, title="Prediction vs Actual (Polynomial Fit)"):
    x = np.arange(len(y_test))

    poly_actual = Polynomial.fit(x, y_test, deg=3)
    poly_predicted = Polynomial.fit(x, predictions, deg=3)

    x_fit = np.linspace(0, len(y_test) - 1, 500)
    y_actual_fit = poly_actual(x_fit)
    y_predicted_fit = poly_predicted(x_fit)

    plt.figure(figsize=(12, 6))

    plt.scatter(x, y_test, color='blue', alpha=0.7, label='Actual Data Points')
    plt.plot(x_fit, y_actual_fit, color='green', linewidth=2, linestyle='dashed', label='Actual (Smoothed Curve)')
    plt.plot(x_fit, y_predicted_fit, color='red', linewidth=2, label='Predicted (Smoothed Curve)')

    plt.xlabel("Data Point Index")
    plt.ylabel("Budget")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt

# Train budget model
def train_budget_model(historical_data):
    features = historical_data[['duration', 'resources', 'expertise']].to_numpy()
    target = historical_data['budget'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = SimpleLinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    st.write(f"Custom Simple Linear Regression RMSE: {rmse}")
    plt = plot_with_polynomial_fit(y_test, predictions)
    st.pyplot(plt)

    return model

# Streamlit Interface
def chatbot_with_ml():
    st.title("BargainBot Chatbot with Machine Learning")

    menu = ["Generate Historical Data and Train Model", "Exit"]
    choice = st.sidebar.selectbox("Choose an Option", menu)

    if choice == "Generate Historical Data and Train Model":
        st.subheader("Training Budget Model")
        num_samples = st.slider("Number of Samples", min_value=100, max_value=5000, value=1000, step=100)
        historical_data = generate_historical_data(num_samples=num_samples)
        # st.write("Sample of Historical Data:")
        # st.write(historical_data.head())
        train_budget_model(historical_data)
    elif choice == "Exit":
        st.write("Exiting... Goodbye!")

if __name__ == "__main__":
    chatbot_with_ml()
