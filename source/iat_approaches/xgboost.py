import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

class XGBoostIATGenerator():
    """
    Generates inter-arrival times by fitting an XGBoost model to the training data.
    """

    def __init__(self, train_arrival_times, data_n_seqs, seed=0) -> None:
        self.train = pd.to_datetime(train_arrival_times)  # Ensure timestamps are in datetime format
        # sort the timestamps
        self.train = self.train.sort_values()
        self.n_seqs = data_n_seqs
        self.model = None
        self.seed = seed
        self._train_model()

    def _train_model(self):
        """Train an XGBoost model to predict inter-arrival times."""
        df = pd.DataFrame({'timestamp': self.train})
        df['inter_arrival'] = df['timestamp'].diff().dt.total_seconds()
        print(f"Inter-arrival times: {df['inter_arrival']}")
        df = df.dropna()

        # Feature Engineering
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['lag_1'] = df['inter_arrival'].shift(1)
        df['lag_2'] = df['inter_arrival'].shift(2)
        df['lag_3'] = df['inter_arrival'].shift(3)
        df = df.dropna()
        
        # Prepare train data
        feature_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'lag_1', 'lag_2', 'lag_3']
        
        # Initialize and fit the scaler
        self.scaler = StandardScaler()
        X = pd.DataFrame(self.scaler.fit_transform(df[feature_columns]), columns=feature_columns)
        y = df['inter_arrival']
        
        # Define hyperparameter search space
        param_dist = {
            'n_estimators': [100, 1000, 2000, 3000],
            'max_depth': [5, 10, 15, 20],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        # Initialize base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.seed
        )

        # Setup RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,  # Number of parameter settings sampled
            cv=5,       # 5-fold cross-validation
            verbose=1,
            random_state=self.seed,
            n_jobs=-1   # Use all available cores
        )
        
        # Perform hyperparameter search
        search.fit(X, y)
        
        # Get the best model
        self.model = search.best_estimator_
        print(f"Best parameters found: {search.best_params_}")
        print(f"Best cross-validation score: {search.best_score_:.4f}")

    def generate_arrivals(self, start_time):
        """Generate new arrival timestamps based on learned inter-arrival times."""
        generated_timestamps = []
        preds = []
        current_time = start_time
        target_end_time = start_time + timedelta(days=self.n_seqs)
        
        # Compute mean inter-arrival time from training data
        mean_inter_arrival = np.mean([(t2 - t1).total_seconds() for t1, t2 in zip(self.train[:-1], self.train[1:])])
        print(f"Mean inter-arrival time: {mean_inter_arrival}")
        
        while current_time < target_end_time:
            # Create input features for prediction
            hour = current_time.hour
            day_of_week = current_time.dayofweek
            day_of_month = current_time.day
            month = current_time.month
            
            if len(generated_timestamps) >= 3:
                lag_1 = (current_time - generated_timestamps[-1]).total_seconds()
                lag_2 = (generated_timestamps[-1] - generated_timestamps[-2]).total_seconds()
                lag_3 = (generated_timestamps[-2] - generated_timestamps[-3]).total_seconds()
            else:
                lag_1, lag_2, lag_3 = mean_inter_arrival, mean_inter_arrival, mean_inter_arrival
                
            X_pred = np.array([[hour, day_of_week, day_of_month, month, float(lag_1), float(lag_2), float(lag_3)]], dtype=np.float32)
            # Scale the features before prediction
            X_pred_scaled = self.scaler.transform(X_pred)
            inter_arrival_pred = max(float(self.model.predict(X_pred_scaled)[0]), 1)  # Ensure positive time
            preds.append(inter_arrival_pred)
            current_time += timedelta(seconds=inter_arrival_pred)
            if current_time < target_end_time:
                generated_timestamps.append(current_time)
        print(f"Average predicted inter-arrival time: {np.mean(preds)}")
        
        return generated_timestamps
