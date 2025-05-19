import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Safely parse time strings (e.g., '1:23.456' or '0:59.123') into milliseconds
def safe_parse_ms(time_str):
    try:
        if pd.isna(time_str) or isinstance(time_str, (float, int)):
            return np.nan
        parts = time_str.strip().split(':')
        if len(parts) == 2:
            # Format: MM:SS.sss
            return (int(parts[0]) * 60 + float(parts[1])) * 1000
        elif len(parts) == 3:
            # Format: HH:MM:SS.sss
            return (int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])) * 1000
    except:
        return np.nan
    return np.nan

# Load race results and combine across multiple years
def load_combined_data(data_dir='archive'):
    races = []
    race_files = [
        ('formula1_2020season_raceResults.csv', 2020),
        ('formula1_2021season_raceResults.csv', 2021),
        ('Formula1_2023season_raceResults.csv', 2023),
        ('Formula1_2024season_raceResults.csv', 2024),
    ]

    # Load each race result file and append if valid
    for filename, year in race_files:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if {'Driver', 'Position', 'Track'}.issubset(df.columns):
                df['year'] = year
                df['raceId'] = df['Track'].astype(str) + "_" + df['year'].astype(str)
                races.append(df)
            else:
                print(f"Skipping {filename}: required columns missing")
        else:
            print(f"Missing: {path}")

    # Error if no valid data loaded
    if not races:
        raise ValueError("No valid race result files found.")

    # Combine all race results into a single DataFrame
    race_df = pd.concat(races, ignore_index=True)

    # Load driver metadata and standings
    drivers = pd.read_csv(os.path.join(data_dir, 'drivers.csv'))
    standings = pd.read_csv(os.path.join(data_dir, 'driver_standings.csv'))

    print("Qualifying and circuit data skipped.")
    return race_df, drivers, standings

# Preprocess combined race and driver data to extract features and target variable
def preprocess_data(race_df, drivers, standings):
    # Create full name to match driver info
    drivers['full_name'] = drivers['forename'] + " " + drivers['surname']
    race_df = race_df.merge(drivers, left_on='Driver', right_on='full_name', how='left')

    # Parse or coerce points column
    if 'Points' in race_df.columns:
        race_df['points'] = pd.to_numeric(race_df['Points'], errors='coerce')
    else:
        race_df['points'] = np.nan

    # Clean Position column
    race_df['Position'] = pd.to_numeric(race_df['Position'], errors='coerce')
    if race_df['Position'].notna().sum() > 0:
        race_df = race_df.dropna(subset=['Position'])

    # Add features:
    # - Driver experience: number of races participated before current race
    # - Last 3 races average position
    # - Average position at each circuit
    # - Grid start position
    race_df['driver_experience'] = race_df.groupby('Driver').cumcount()
    race_df['last_3_avg_position'] = race_df.groupby('Driver')['Position'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    race_df['avg_circuit_position'] = race_df.groupby(['Driver', 'Track'])['Position'].transform('mean')
    race_df['grid'] = pd.to_numeric(race_df['Starting Grid'], errors='coerce')

    # Select relevant features
    feature_columns = ['driver_experience', 'last_3_avg_position', 'avg_circuit_position', 'year', 'grid', 'points']
    X = race_df[feature_columns].copy()

    # Define the target variable (finishing position)
    y = race_df['Position'].dropna().astype(int) if race_df['Position'].notna().sum() > 0 else pd.Series(dtype=int)

    # Handle missing or infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean(numeric_only=True))

    return X, y, race_df

# Create a preprocessing pipeline that scales numerical features and encodes categorical ones
def create_feature_pipeline(X):
    num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    transformers = []
    if num:
        transformers.append(('num', StandardScaler(), num))
    if cat:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), cat))
    return ColumnTransformer(transformers=transformers)

# Prepare training data: load data, preprocess it, and create train/test splits
def prepare_data_for_training(data_dir='archive'):
    race_df, drivers, standings = load_combined_data(data_dir)
    X, y, full_df = preprocess_data(race_df, drivers, standings)
    preprocessor = create_feature_pipeline(X)
    return train_test_split(X, y, test_size=0.2, random_state=42), preprocessor, full_df
