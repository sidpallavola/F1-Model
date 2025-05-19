import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import pandas as pd  # For handling dataframes
import os  # For file system operations

# Function: plot_prediction_vs_actual
# Plots actual vs predicted race positions for model evaluation.
def plot_prediction_vs_actual(y_test, y_pred):
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.scatter(y_test, y_pred, alpha=0.5)  # Scatter plot for actual vs predicted

    # Add perfect prediction line for reference
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # Red dashed line

    plt.xlabel('Actual Position')  # X-axis label
    plt.ylabel('Predicted Position')  # Y-axis label
    plt.title('F1 Position Prediction: Actual vs Predicted')  # Plot title
    plt.grid(True)  # Add grid to the plot

    os.makedirs('plots', exist_ok=True)  # Create 'plots' directory if it doesn't exist
    plt.savefig('plots/prediction_vs_actual.png')  # Save plot to file
    plt.close()  # Close the plot to free memory
    print("Saved prediction vs actual plot to plots/prediction_vs_actual.png")

# Function: plot_race_predictions
# Creates a horizontal bar plot of win probabilities for drivers.
def plot_race_predictions(result_df):
    result_df = result_df.sort_values('win_probability', ascending=False)  # Sort by win probability
    result_df['driver_name'] = result_df['forename'] + ' ' + result_df['surname']  # Create full driver name

    plt.figure(figsize=(12, 8))  # Set the figure size
    bars = plt.barh(result_df['driver_name'], result_df['win_probability'], color='skyblue')  # Horizontal bar chart

    # Add text labels to each bar
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.2f}%', ha='left', va='center')  # Add label

    plt.xlabel('Win Probability (%)')  # X-axis label
    plt.ylabel('Driver')  # Y-axis label
    plt.title('F1 Race Win Probability Predictions')  # Plot title
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add dashed grid on x-axis
    plt.tight_layout()  # Adjust layout to fit labels

    os.makedirs('plots', exist_ok=True)  # Ensure 'plots' folder exists
    plt.savefig('plots/race_predictions.png')  # Save plot to file
    plt.close()  # Close the plot to free memory
    print("Saved race predictions plot to plots/race_predictions.png")

# Function: print_formatted_probabilities
# Prints the win probabilities for each driver in order.
def print_formatted_probabilities(result_df, race_name="Grand Prix"):
    print(f"\n{race_name}:")  # Print race name
    for i, row in enumerate(result_df.itertuples(), 1):
        print(f"{i}. {row.forename} {row.surname} – {row.win_probability:.2f}%")  # Print each driver's info
