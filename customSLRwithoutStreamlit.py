import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

def plot_with_polynomial_fit(y_test, predictions, title="Prediction vs Actual (Polynomial Fit)"):
    x = np.arange(len(y_test))  # Index for data points

    # Fit polynomial (degree 3 for smooth curves)
    poly_actual = Polynomial.fit(x, y_test, deg=3)
    poly_predicted = Polynomial.fit(x, predictions, deg=3)

    # Generate smooth lines
    x_fit = np.linspace(0, len(y_test) - 1, 500)  # Smooth x range
    y_actual_fit = poly_actual(x_fit)
    y_predicted_fit = poly_predicted(x_fit)

    plt.figure(figsize=(12, 6))

    # Scatter plot for actual data points
    plt.scatter(x, y_test, color='blue', alpha=0.7, label='Actual Data Points')

    # Smooth line for actual data
    plt.plot(x_fit, y_actual_fit, color='green', linewidth=2, linestyle='dashed', label='Actual (Smoothed Curve)')

    # Smooth line for predicted data
    plt.plot(x_fit, y_predicted_fit, color='red', linewidth=2, label='Predicted (Smoothed Curve)')

    # Add labels, title, and legend
    plt.xlabel("Data Point Index")
    plt.ylabel("Budget")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.show()

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
    budgets = (durations ** duration_factor * duration_scale) + (resources * resource_scale) + (np.log1p(expertise) * expertise_scale) + noise  

    historical_data = pd.DataFrame({
        'duration': durations,
        'resources': resources,
        'expertise': expertise,
        'budget': budgets
    })
    return historical_data
def generate_scattered_historical_data(num_samples):
    # Generate random inputs
    durations = np.random.randint(1, 20, num_samples)
    resources = np.random.randint(1, 20, num_samples)
    expertise = np.random.randint(1, 20, num_samples)
    
    # Define the nonlinear relationship with higher noise
    duration_factor = 1.5
    duration_scale = 50
    resource_scale = 150
    expertise_scale = 300
    
    # Increase noise level to scatter data
    noise = np.random.normal(0, 1000, num_samples)  # Higher noise level
    budgets = (durations ** duration_factor * duration_scale) + (resources * resource_scale) + (np.log1p(expertise) * expertise_scale) + noise *2 

    scattered_data = pd.DataFrame({
        'duration': durations,
        'resources': resources,
        'expertise': expertise,
        'budget': budgets
    })
    return scattered_data

# Custom Simple Linear Regression Class
class SimpleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a column of ones to X for the intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Calculate the coefficients using the normal equation
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        # Add a column of ones to X for the intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coefficients

# Function to plot predicted vs actual values
def plot_predicted_vs_actual(y_test, predictions, title="Predicted vs Actual Budget"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, predictions, alpha=0.7, color='blue', label='Predicted vs Actual')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2, label='Perfect Prediction Line')
    ax.set_xlabel('Actual Budget')
    ax.set_ylabel('Predicted Budget')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()

# Train budget model
def train_budget_model(historical_data):
    features = historical_data[['duration', 'resources', 'expertise']].to_numpy()
    target = historical_data['budget'].to_numpy()
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Initialize and train the custom linear regression model
    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Custom Simple Linear Regression RMSE: {rmse}")
    
    # Plot predicted vs actual values
    plot_predicted_vs_actual(y_test, predictions)
    
    return model

# Forecast budget using the model
def forecast_budget_with_model(model, project_details):
    features = np.array([project_details['duration'], project_details['resources'], project_details['expertise']]).reshape(1, -1)
    estimated_budget = model.predict(features)[0]
    return estimated_budget

def plot_predictions_with_actual(y_test, predictions, title="Predicted vs Actual Budget"):
    x = np.arange(len(y_test))  # Index for data points

    plt.figure(figsize=(12, 6))
    
    # Scatter plot for actual data points
    plt.scatter(x, y_test, color='blue', alpha=0.7, label='Actual Data Points')
    
    # Line for predicted values
    plt.plot(x, predictions, color='red', linewidth=2, label='Predicted Line')
    
    # Line for actual values
    plt.plot(x, y_test, color='green', linestyle='dashed', linewidth=2, label='Actual Line')
    
    # Add labels, title, and legend
    plt.xlabel("Data Point Index")
    plt.ylabel("Budget")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Main chatbot-like UI
def chatbot_with_ml():
    print("Welcome to BargainBot Chatbot")
    
    # Generate historical data
    historical_data = generate_historical_data(num_samples=1000)

    # Train the model
    model = train_budget_model(historical_data)

    # Get predictions
    features = historical_data[['duration', 'resources', 'expertise']].to_numpy()
    target = historical_data['budget'].to_numpy()
    _, X_test, _, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)

    # Plot predictions with actual values
    plot_with_polynomial_fit(y_test, predictions)


    # while True:
    #     print("\nPlease choose an option:")
    #     print("1. Ask about project types")
    #     print("2. Calculate complexity score")
    #     print("3. Estimate budget with ML")
    #     print("4. Get a price suggestion")
    #     print("5. Adjust price based on feedback")
    #     print("6. Exit")
        
    #     option = input("Enter your choice: ")
        
    #     if option == "1":
    #         query = input("Enter your project query: ")
    #         print("BargainBot: We can help with your query!")
        
    #     elif option == "2":
    #         duration = int(input("Enter project duration (weeks): "))
    #         resources = int(input("Enter resources required (count): "))
    #         expertise = int(input("Enter expertise level (1-10): "))
    #         complexity_score = (duration * 2) + (resources * 3) + (expertise * 5)
    #         print(f"BargainBot: Complexity Score is {complexity_score}")
        
    #     elif option == "3":
    #         duration = int(input("Enter project duration (weeks): "))
    #         resources = int(input("Enter resources required (count): "))
    #         expertise = int(input("Enter expertise level (1-10): "))
    #         project_details = {"duration": duration, "resources": resources, "expertise": expertise}
    #         budget = forecast_budget_with_model(model, project_details)
    #         print(f"BargainBot: Estimated Budget is {budget}")
        
    #     elif option == "4":
    #         complexity_score = float(input("Enter the complexity score: "))
    #         estimated_budget = float(input("Enter the estimated budget: "))
    #         weights = np.array([0.4, 0.6])
    #         inputs = np.array([complexity_score, estimated_budget])
    #         suggested_price = np.dot(weights, inputs)
    #         print(f"BargainBot: Suggested Price is {suggested_price}")
        
    #     elif option == "5":
    #         current_price = float(input("Enter the current price: "))
    #         feedback = input("Enter client feedback (positive/negative/neutral): ").lower()
    #         adjustment = {'positive': 0.95, 'negative': 1.1, 'neutral': 1.0}
    #         adjusted_price = current_price * adjustment.get(feedback, 1.0)
    #         print(f"BargainBot: Adjusted Price is {adjusted_price}")
        
    #     elif option == "6":
    #         print("Goodbye!")
    #         break
        
    #     else:
    #         print("Invalid choice. Please try again.")

# Run the chatbot interface
if __name__ == "__main__":
    chatbot_with_ml()
