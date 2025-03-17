import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from datetime import timedelta, datetime
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from tensorflow.keras.utils import to_categorical
import copy

class LSTM_IAT_Generator:
    def __init__(self, train_arrival_times, data_n_seqs, inter_arrival_durations):
        self.train_arrival_times = sorted(train_arrival_times)
        self.data_n_seqs = data_n_seqs
        self.sequence_length = 5
        self.model = None
        # Switch to MinMaxScaler for better scaling of small values
        self.scaler = MinMaxScaler(feature_range=(0.1, 0.9))
        # self.inter_arrival_durations = inter_arrival_durations
        # Train the model during initialization
        self._prepare_and_train_model()

    @staticmethod
    def _transform_features(df, scaler):
        """Transforms inter-arrival times and extracts temporal features."""
        df = df.copy()
        
        # Scale inter-arrival times
        df[['inter_time']] = scaler.transform(df[['inter_time']])
        
        # Extract additional temporal features
        timestamps = df['timestamp']
        
        # Hour of day (sine and cosine for cyclical nature)
        hours = timestamps.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        
        # Day of week (one-hot encoded)
        weekday = timestamps.dt.weekday
        weekday_ohe = to_categorical(weekday, num_classes=7)
        for i in range(7):
            df[f'day_{i}'] = weekday_ohe[:, i]
        
        # Week of month (1-5)
        df['week_of_month'] = timestamps.dt.day.apply(lambda x: (x - 1) // 7 + 1) / 5
        
        # Month (sine and cosine for cyclical nature)
        months = timestamps.dt.month
        df['month_sin'] = np.sin(2 * np.pi * months / 12)
        df['month_cos'] = np.cos(2 * np.pi * months / 12)
        
        # Quarter of year (1-4)
        df['quarter'] = timestamps.dt.quarter / 4
        
        # Is weekend
        df['is_weekend'] = timestamps.dt.weekday.isin([5, 6]).astype(float)
        
        # Select features for model
        features = ['inter_time', 'hour_sin', 'hour_cos', 
                   'week_of_month', 'month_sin', 'month_cos', 'quarter', 
                   'is_weekend'] + [f'day_{i}' for i in range(7)]
        
        return df[features]

    def _vectorize(self, log, ngram_size):
        """Converts the DataFrame into a sequence-based format for LSTM training."""
        num_samples = len(log) - ngram_size  # Adjust to avoid padding
        dt_prefixes, dt_expected = [], []

        # Remove padding approach and use sliding window
        for i in range(num_samples):
            dt_prefixes.append(log.iloc[i:i + ngram_size])
            dt_expected.append(log.iloc[i + ngram_size:i + ngram_size + 1][['inter_time']])

        dt_prefixes = pd.concat(dt_prefixes, axis=0, ignore_index=True)
        dt_expected = pd.concat(dt_expected, axis=0, ignore_index=True)

        # Reshape the data
        dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples, ngram_size, -1)
        dt_expected = dt_expected.to_numpy().reshape((num_samples, 1))

        return dt_prefixes, dt_expected

    def _prepare_training_data(self):
        """Prepares training data with inter-arrival times and temporal features."""
        # Convert list to DataFrame first
        df_arrivals = pd.DataFrame({'timestamp': self.train_arrival_times})
        daily_times = df_arrivals.to_dict('records')
        
        # Create DataFrame with timestamps and inter-arrival durations
        inter_arrival_times = []
        for i, event in enumerate(daily_times):
            delta = (daily_times[i]['timestamp'] - 
                    daily_times[i - 1]['timestamp']).total_seconds() if i > 0 else 0
            inter_arrival_times.append({
                'inter_time': delta,
                'timestamp': daily_times[i]['timestamp']
            })
        
        train_data = pd.DataFrame(inter_arrival_times)
        print(f"train_data\n: {train_data}")
        
        # Fit and apply transformation
        self.scaler.fit(train_data[['inter_time']])
        train_data = self._transform_features(train_data, self.scaler)
        
        # Generate training sequences
        return self._vectorize(train_data, self.sequence_length)

    def _build_model(self, n_features):
        """Builds a LSTM model for inter-arrival time prediction."""
        model = Sequential([
            LSTM(64, input_shape=(self.sequence_length, n_features), 
                 return_sequences=True, dropout=0.1),
            LSTM(32, return_sequences=False, dropout=0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        # Use Huber loss which is less sensitive to outliers than MSE
        # and more sensitive to differences than MAE
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss=tf.keras.losses.Huber())
        return model

    def _prepare_and_train_model(self):
        """Prepares data and trains the LSTM model."""
        X_train, y_train = self._prepare_training_data()
        
        # Print some statistics about the data
        print("Training data statistics:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_train mean: {np.mean(y_train)}")
        print(f"y_train std: {np.std(y_train)}")
        print(f"y_train min: {np.min(y_train)}")
        print(f"y_train max: {np.max(y_train)}")

        n_samples = len(X_train)
        batch_size = min(32, max(4, n_samples // 10))

        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001
            )
        ]

        self.model = self._build_model(n_features=X_train.shape[2])
        self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def generate_arrivals(self, start_time):
        """Generates arrival times based on learned patterns."""
        generated_times = [start_time]
        current_sequence = []

        # Initialize sequence with first few actual times
        train_end_idx = min(self.sequence_length, len(self.train_arrival_times) - 1)
        for i in range(train_end_idx):
            timestamp = self.train_arrival_times[i]
            iat = (self.train_arrival_times[i + 1] - self.train_arrival_times[i]).total_seconds()
            scaled_iat = self.scaler.transform([[iat]])[0][0]
            
            # Calculate temporal features
            hour_sin = np.sin(2 * np.pi * timestamp.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamp.hour / 24)
            week_of_month = ((timestamp.day - 1) // 7 + 1) / 5
            month_sin = np.sin(2 * np.pi * timestamp.month / 12)
            month_cos = np.cos(2 * np.pi * timestamp.month / 12)
            quarter = timestamp.quarter / 4
            is_weekend = float(timestamp.weekday() in [5, 6])
            
            # One-hot encode weekday
            weekday_ohe = [0] * 7
            weekday_ohe[timestamp.weekday()] = 1
            
            features = [scaled_iat, hour_sin, hour_cos, week_of_month, 
                       month_sin, month_cos, quarter, is_weekend] + weekday_ohe
            current_sequence.append(np.array(features))

        target_end_time = start_time + timedelta(days=self.data_n_seqs)
        
        # Keep track of the last few actual IATs for stability
        recent_iats = []
        
        while generated_times[-1] < target_end_time:
            last_time = generated_times[-1]
            
            # Calculate temporal features for last time
            hour_sin = np.sin(2 * np.pi * last_time.hour / 24)
            hour_cos = np.cos(2 * np.pi * last_time.hour / 24)
            week_of_month = ((last_time.day - 1) // 7 + 1) / 5
            month_sin = np.sin(2 * np.pi * last_time.month / 12)
            month_cos = np.cos(2 * np.pi * last_time.month / 12)
            quarter = last_time.quarter / 4
            is_weekend = float(last_time.weekday() in [5, 6])
            
            # One-hot encode weekday
            weekday_ohe = [0] * 7
            weekday_ohe[last_time.weekday()] = 1

            # Prepare input sequence
            sequence = np.array(current_sequence[-self.sequence_length:])
            sequence[-1, 1:] = [hour_sin, hour_cos, week_of_month, 
                              month_sin, month_cos, quarter, is_weekend] + weekday_ohe
            sequence = sequence.reshape(1, self.sequence_length, sequence.shape[1])
            print(f"sequence: {sequence}")

            # Predict next IAT
            predicted_scaled_iat = self.model.predict(sequence, verbose=0)[0][0]
            
            # Clip the scaled prediction to ensure it stays within the scaler's range
            predicted_scaled_iat = np.clip(predicted_scaled_iat, 0.1, 0.9)
            
            # Transform back to original scale
            predicted_iat = self.scaler.inverse_transform([[predicted_scaled_iat]])[0][0]
            
            # Add some randomness to prevent getting stuck
            predicted_iat *= np.random.normal(1, 0.1)  # 10% random variation
            
            # Ensure prediction is positive and reasonable
            predicted_iat = max(1.0, predicted_iat)  # Minimum 1 second between events
            
            # Store for monitoring
            recent_iats.append(predicted_iat)
            if len(recent_iats) > 10:
                recent_iats.pop(0)
            
            # If predictions are collapsing, reset using mean of recent valid predictions
            if len(recent_iats) > 5 and predicted_iat < np.mean(recent_iats) * 0.1:
                predicted_iat = np.mean(recent_iats)
            
            print(f"predicted_scaled_iat: {predicted_scaled_iat}, predicted_iat: {predicted_iat}")

            # Generate next timestamp
            next_time = generated_times[-1] + timedelta(seconds=float(predicted_iat))
            
            if next_time < target_end_time:
                generated_times.append(next_time)
                
                # Calculate features for next time
                hour_sin = np.sin(2 * np.pi * next_time.hour / 24)
                hour_cos = np.cos(2 * np.pi * next_time.hour / 24)
                week_of_month = ((next_time.day - 1) // 7 + 1) / 5
                month_sin = np.sin(2 * np.pi * next_time.month / 12)
                month_cos = np.cos(2 * np.pi * next_time.month / 12)
                quarter = next_time.quarter / 4
                is_weekend = float(next_time.weekday() in [5, 6])
                
                # One-hot encode weekday
                weekday_ohe = [0] * 7
                weekday_ohe[next_time.weekday()] = 1
                
                features = [predicted_scaled_iat, hour_sin, hour_cos, week_of_month, 
                          month_sin, month_cos, quarter, is_weekend] + weekday_ohe
                current_sequence.append(np.array(features))
            else:
                break

        return generated_times
