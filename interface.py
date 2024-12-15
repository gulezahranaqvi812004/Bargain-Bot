import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# Custom Linear Regression Implementation
class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


# Custom Random Forest Regressor Implementation (Simplified)
class CustomRandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return predictions.mean(axis=0)


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
    budgets = (durations ** duration_factor * duration_scale) + \
              (resources * resource_scale) + \
              (np.log1p(expertise) * expertise_scale) + noise

    historical_data = pd.DataFrame({
        'duration': durations,
        'resources': resources,
        'expertise': expertise,
        'budget': budgets
    })
    return historical_data


# Train the model
def train_budget_model(historical_data):
    features = historical_data[['duration', 'resources', 'expertise']].values
    target = historical_data['budget'].values

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    models = {
        'Custom Linear Regression': CustomLinearRegression(),
        'Custom Random Forest': CustomRandomForestRegressor(n_estimators=10, max_depth=5)
    }

    best_model = None
    best_rmse = float('inf')

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    return best_model


# Forecast budget using the model
def forecast_budget_with_model(model, project_details):
    features = np.array([project_details['duration'], project_details['resources'], project_details['expertise']]).reshape(1, -1)
    estimated_budget = model.predict(features)[0]
    return estimated_budget


# Main chatbot UI with Streamlit
def chatbot_with_ml():
    st.title("BargainBot")
    st.subheader("Negotiate and get best price according to you!")

    # Generate historical data and train the model
    historical_data = generate_historical_data(num_samples=1000)
    model = train_budget_model(historical_data)

    # Input section
    st.subheader("Welcome!")
    st.write("Please choose an option from below:")

    option = st.selectbox("Select an option", (
        "Ask about project types (FR-1)",
        "Calculate complexity score (FR-2)",
        "Estimate budget with ML (FR-3)",
        # "Get a price suggestion (FR-4)",
        "Adjust price based on feedback (FR-5)"
    ))

    # Handle each option
    if option == "Ask about project types (FR-1)":
        query = st.text_input("Enter your project query:")
        if query:
            st.write("BargainBot: We can help with your query!")

    elif option == "Calculate complexity score (FR-2)":
        duration = st.number_input("Enter project duration (weeks):", min_value=1, max_value=52)
        resources = st.number_input("Enter resources required (count):", min_value=1, max_value=100)
        expertise = st.number_input("Enter expertise level (1-10):", min_value=1, max_value=10)

        if st.button("Calculate Complexity Score"):
            complexity_score = (duration * 2) + (resources * 3) + (expertise * 5)
            st.write(f"BargainBot: Complexity Score is {complexity_score}")

    elif option == "Estimate budget with ML (FR-3)":
        duration = st.number_input("Enter project duration (weeks):", min_value=1, max_value=52)
        resources = st.number_input("Enter resources required (count):", min_value=1, max_value=100)
        expertise = st.number_input("Enter expertise level (1-10):", min_value=1, max_value=10)

        project_details = {"duration": duration, "resources": resources, "expertise": expertise}
        if st.button("Estimate Budget"):
            budget = forecast_budget_with_model(model, project_details)
            st.write(f"BargainBot: Estimated Budget is {budget}")

    # elif option == "Get a price suggestion (FR-4)":
    #     complexity_score = st.number_input("Enter the complexity score:", min_value=0)
    #     estimated_budget = st.number_input("Enter the estimated budget:", min_value=0.0)

    #     if st.button("Suggest Price"):
    #         weights = np.array([0.4, 0.6])
    #         inputs = np.array([complexity_score, estimated_budget])
    #         suggested_price = np.dot(weights, inputs)
    #         st.write(f"BargainBot: Suggested Price is {suggested_price}")

    elif option == "Adjust price based on feedback (FR-5)":
        current_price = st.number_input("Enter the current price:", min_value=0.0)
        feedback = st.selectbox("Enter client feedback", ("positive", "negative", "neutral"))

        if st.button("Adjust Price"):
            adjustment = {'positive': 0.95, 'negative': 1.1, 'neutral': 1.0}
            adjusted_price = current_price * adjustment.get(feedback, 1.0)
            st.write(f"BargainBot: Adjusted Price is {adjusted_price}")


# Run the chatbot interface
if __name__ == "__main__":
    chatbot_with_ml()
