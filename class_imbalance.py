import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit Application
def app():
    st.title("Data Distribution Visualization")
    
    # Upload CSV File
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        # Load the dataset
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.write(df.head())  # Display first few rows

        # Get numeric columns
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_columns.empty:
            st.subheader("Numeric Columns Distribution")
            
            # Create a histogram for each numeric column
            for column in numeric_columns:
                st.write(f"Distribution of {column}")
                plt.figure(figsize=(8, 4))
                sns.histplot(df[column], kde=True, color='blue', bins=10)
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                
                # Render plot in Streamlit
                st.pyplot(plt)
                plt.clf()  # Clear figure to avoid overlap
        else:
            st.warning("No numeric columns found in the dataset.")
    else:
        st.info("Please upload a CSV file to proceed.")

# Run the Streamlit App
if __name__ == "__main__":
    app()
