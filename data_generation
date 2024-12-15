import pandas as pd
import numpy as np

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

# Save historical data to a CSV file
def save_historical_data_to_csv(num_samples, filename="historical_data.csv"):
    historical_data = generate_historical_data(num_samples)
    historical_data.to_csv(filename, index=False)
    print(f"Dataset saved as {filename}")

# Generate and save the dataset
save_historical_data_to_csv(num_samples=1000)
