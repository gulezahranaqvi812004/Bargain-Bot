import pandas as pd
import numpy as np

def process_client_message(message):
    responses = pd.DataFrame({
        'keyword': ['web', 'app', 'design'],
        'response': [
            'We can help with web development projects!',
            'App development is one of our specialties!',
            'We offer top-notch design services!'
        ]
    })
    for _, row in responses.iterrows():
        if row['keyword'] in message.lower():
            return row['response']
    return "Could you provide more details about your project?"

def calculate_complexity_score(project_details):
    weights = np.array([2, 3, 5])  
    values = np.array([project_details['duration'], project_details['resources'], project_details['expertise']])
    return np.dot(weights, values)

# Updated function to generate synthetic historical data dynamically
def generate_historical_data(num_samples=10):
    # Generate random data for duration, resources, and expertise
    durations = np.random.randint(1, 12, num_samples)
    resources = np.random.randint(1, 10, num_samples)
    expertise = np.random.randint(1, 11, num_samples)
    
    # Assuming the budget is some function of these factors (e.g., linearly related)
    budgets = (durations * 100) + (resources * 150) + (expertise * 200)  # Example budget formula
    
    # Create a DataFrame with the generated data
    historical_data = pd.DataFrame({
        'duration': durations,
        'resources': resources,
        'expertise': expertise,
        'budget': budgets
    })
    
    return historical_data

# Updated function to forecast the budget using dynamically generated historical data
def forecast_budget(project_details):
    # Generate historical data
    historical_data = generate_historical_data(num_samples=10)
    
    # Calculate the average of historical factors (duration, resources, expertise)
    factor_weights = historical_data[['duration', 'resources', 'expertise']].mean(axis=0)
    
    # Estimate the budget based on the current project details using the mean factor weights
    estimated_budget = np.dot(
        np.array([project_details['duration'], project_details['resources'], project_details['expertise']]),
        factor_weights.values
    )
    return estimated_budget

def suggest_price(complexity_score, estimated_budget):
    weights = np.array([0.4, 0.6])  
    inputs = np.array([complexity_score, estimated_budget])
    return np.dot(weights, inputs)

def adjust_price(current_price, feedback):
    adjustment = {'positive': 0.95, 'negative': 1.1, 'neutral': 1.0}
    return current_price * adjustment.get(feedback, 1.0)

def chatbot():
    print("🤖 Welcome to BargainBot!")
    while True:
        print("\nOptions:")
        print("1. Ask about project types (FR-1)")
        print("2. Calculate complexity score (FR-2)")
        print("3. Estimate budget (FR-3)")
        print("4. Get a price suggestion (FR-4)")
        print("5. Adjust price based on feedback (FR-5)")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            message = input("Enter your project query: ")
            response = process_client_message(message)
            print(f"BargainBot: {response}")

        elif choice == "2":
            duration = int(input("Enter project duration (weeks): "))
            resources = int(input("Enter resources required (count): "))
            expertise = int(input("Enter expertise level (1-10): "))
            project_details = {"duration": duration, "resources": resources, "expertise": expertise}
            score = calculate_complexity_score(project_details)
            print(f"BargainBot: Complexity Score is {score}")

        elif choice == "3":
            duration = int(input("Enter project duration (weeks): "))
            resources = int(input("Enter resources required (count): "))
            expertise = int(input("Enter expertise level (1-10): "))
            project_details = {"duration": duration, "resources": resources, "expertise": expertise}
            budget = forecast_budget(project_details)
            print(f"BargainBot: Estimated Budget is {budget}")

        elif choice == "4":
            complexity_score = float(input("Enter the complexity score: "))
            estimated_budget = float(input("Enter the estimated budget: "))
            price = suggest_price(complexity_score, estimated_budget)
            print(f"BargainBot: Suggested Price is {price}")

        elif choice == "5":
            current_price = float(input("Enter the current price: "))
            feedback = input("Enter client feedback (positive/negative/neutral): ").lower()
            adjusted_price = adjust_price(current_price, feedback)
            print(f"BargainBot: Adjusted Price is {adjusted_price}")

        elif choice == "6":
            print("BargainBot: Goodbye! Have a great day! 👋")
            break

        else:
            print("BargainBot: Invalid choice. Please try again.")

if __name__ == "__main__":
    chatbot()
