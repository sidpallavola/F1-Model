import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Define a class for the F1 prediction model
class F1PredictionModel:
    def __init__(self):
        # Initialize the pipeline attribute (will be defined in training)
        self.pipeline = None

    # Train the model using the given features, target values, and preprocessing pipeline
    def train(self, X, y, preprocessor):
        # Define a pipeline that includes preprocessing and the Random Forest model
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),  # Step 1: preprocess the input features
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Step 2: apply the regressor
        ])
        # Fit the full pipeline to the training data
        self.pipeline.fit(X, y)

    # Predict positions using the trained pipeline
    def predict(self, X):
        return self.pipeline.predict(X)

    # Save the entire pipeline (preprocessor + model) to disk
    def save_model(self, path):
        joblib.dump(self.pipeline, path)

    # Load a saved pipeline from disk and return a F1PredictionModel instance with it
    @staticmethod
    def load_model(path):
        model = F1PredictionModel()
        model.pipeline = joblib.load(path)
        return model
