import os
import pandas as pd
from data import prepare_data_for_training, load_combined_data, preprocess_data
from model import F1PredictionModel
from visualization import plot_prediction_vs_actual, plot_race_predictions, print_formatted_probabilities
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Define the F1 points system for positions 1 through 10
POINTS_SYSTEM = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

# Helper function to assign points based on race finish position
def assign_points(position):
    return POINTS_SYSTEM[position - 1] if 1 <= position <= 10 else 0

# Train the prediction model using historical data
def train_model(data_dir='archive'):
    # Load and preprocess training and testing datasets
    (X_train, X_test, y_train, y_test), preprocessor, full_df = prepare_data_for_training(data_dir)

    # Initialize and train the model
    model = F1PredictionModel()
    model.train(X_train, y_train, preprocessor)

    # Generate predictions on the test set
    preds = model.predict(X_test)

    # Calculate and print evaluation metrics
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)

    print("\nModel Evaluation:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Plot predicted vs actual results
    plot_prediction_vs_actual(y_test, preds)

    # Save the trained model for later use
    model.save_model("models/f1_prediction_random_forest.pkl")
    return model, preprocessor

# Predict win probabilities and championship points for the 2025 F1 season
def predict_2025(data_dir='archive'):
    # Load the trained model
    model = F1PredictionModel.load_model("models/f1_prediction_random_forest.pkl")

    # Load all necessary combined data and 2025 tracks
    full_df, drivers, _ = load_combined_data(data_dir)
    tracks_2025 = pd.read_csv(os.path.join(data_dir, '2025_tracks.csv'))

    # Ensure the tracks file has the required column
    if 'track' not in tracks_2025.columns:
        raise ValueError("2025_tracks.csv must contain a 'track' column")

    # Create a full name for each driver
    drivers['full_name'] = drivers['forename'] + " " + drivers['surname']
    total_points = {}

    # Loop through each 2025 track to simulate the race
    for track_name in tracks_2025['track'].unique():
        race_id = f"{track_name}_2025"
        pred_rows = []

        # Generate mock input data for each driver
        for _, driver in drivers.iterrows():
            pred_rows.append({
                'Driver': driver['full_name'],
                'Track': track_name,
                'year': 2025,
                'raceId': race_id,
                'Position': np.nan,
                'Starting Grid': np.random.randint(1, 21),
                'full_driver_name': driver['full_name'],
                'Points': 0
            })

        race_df = pd.DataFrame(pred_rows)

        # Preprocess the data for model prediction
        X_input, _, pred_df = preprocess_data(race_df, drivers, pd.DataFrame(columns=['raceId', 'driverId', 'points']))

        # Skip if preprocessing failed or returned empty input
        if X_input.empty:
            print(f"\nSkipping {track_name} Grand Prix: no valid input features after preprocessing.")
            continue

        # Match model input columns to ensure consistent format
        model_features = model.pipeline.named_steps['preprocessor'].transformers_[0][2]
        for col in model_features:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[model_features].fillna(0)

        # Predict driver positions
        pred_df['predicted_position'] = model.predict(X_input)

        # Score prediction: use inverse-cube scaling (to emphasize top ranks)
        pred_df['position_score'] = 1 / (pred_df['predicted_position'] ** 3)

        # Apply boost factors for top and secondary drivers to simulate performance bias
        pred_df['driver_boost'] = 1.0
        top_drivers = ['Max Verstappen', 'Lando Norris', 'Lewis Hamilton', 'Charles Leclerc']
        secondary_drivers = ['George Russell', 'Carlos Sainz', 'Oscar Piastri', 'Fernando Alonso']

        for driver in top_drivers:
            pred_df.loc[pred_df['Driver'] == driver, 'driver_boost'] = 8.0
        for driver in secondary_drivers:
            pred_df.loc[pred_df['Driver'] == driver, 'driver_boost'] = 3.0

        # Final score and normalized win probabilities
        pred_df['final_score'] = pred_df['position_score'] * pred_df['driver_boost']
        total_score = pred_df['final_score'].sum()
        pred_df['win_probability'] = (pred_df['final_score'] / total_score * 100).round(2)

        # Rank drivers and assign points based on predicted order
        pred_df = pred_df.sort_values('win_probability', ascending=False)
        pred_df['win_based_finish'] = range(1, len(pred_df) + 1)
        pred_df['race_points'] = pred_df['win_based_finish'].apply(assign_points)

        # Accumulate championship points
        for _, row in pred_df.iterrows():
            name = row['Driver']
            total_points[name] = total_points.get(name, 0) + row['race_points']

        # Show predictions and plot for the current track
        print_formatted_probabilities(pred_df, f"{track_name} Grand Prix")
        plot_race_predictions(pred_df)

    # Compile and display final standings
    final_standings = pd.DataFrame([
        {'Driver': driver, 'Total Points': pts} for driver, pts in total_points.items()
    ])
    final_standings = final_standings.sort_values(by='Total Points', ascending=False).reset_index(drop=True)

    print("\n=== Final Driver Standings (2025 Season) ===")
    for idx, row in final_standings.iterrows():
        print(f"{idx + 1}. {row['Driver']} – {row['Total Points']} pts")

# Run training and prediction when script is executed directly
if __name__ == "__main__":
    model, preprocessor = train_model()
    predict_2025()
