import pandas as pd
import numpy as np
from numpy.random import triangular as triang
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import timedelta, datetime, timezone
import itertools
import traceback

from source.arrival_distribution import get_boundaries_of_day


class ProphetIATGenerator():
    """
    Generates inter arrival times by training the Prophet Time Series Model
    """

    def __init__(self, train_arrival_times, data_n_seqs) -> None:
        self.train = train_arrival_times
        self.lower_bound, self.upper_bound = get_boundaries_of_day(self.train) # bounds are given in elapsed time of the day in seconds
        self.n_seqs = data_n_seqs

        # TODO check if trained model is already available and if so load that
        # TODO save models


    def generate_arrivals(self, start_time):
        time_series_df, max_cap = self.get_time_series_df()

        model = self.train_prophet_model(df=time_series_df)

        # use trained model to generate arrivals
        gen = list()
        n_gen_inst = 0
        while n_gen_inst < self.n_seqs:
            future = pd.date_range(start=start_time, end=(start_time + timedelta(days=30)), freq='D').to_frame(
                name='ds', index=False)

            future['cap'] = max_cap
            future['floor'] = 0
            future['ds'] = future['ds'].dt.tz_localize(None)
            forecast = model.predict(future)

            def rand_value(x):
                raw_val = triang(x.yhat_lower, x.yhat, x.yhat_upper, size=1)
                raw_val = raw_val[0] if raw_val[0] > 0 else 0
                return raw_val

            forecast['gen'] = forecast.apply(rand_value, axis=1)
            forecast['gen_round'] = np.ceil(forecast['gen'])
            # n_gen_inst += np.sum(forecast['gen_round'])
            n_gen_inst += len(forecast['gen_round'])
            gen.append(forecast[forecast.gen_round > 0][['ds', 'gen_round']])
            start_time = forecast.ds.max()
        gen = pd.concat(gen, axis=0, ignore_index=True)
        gen = gen[:self.n_seqs]

        def pp(start, n):
            # start_unix = int(start.value // 10 ** 9)
            # end_unix = int((start + timedelta(hours=1)).value // 10 ** 9)
            day = datetime.combine(start.date(), datetime.min.time())
            timestamp_first = day + timedelta(seconds=self.lower_bound)
            timestamp_last = day + timedelta(seconds=self.upper_bound)

            # transform lower and upper timestamp into unix timestamps
            start_unix = (timestamp_first - datetime(1970, 1, 1)) / timedelta(seconds=1)
            end_unix = (timestamp_last - datetime(1970, 1, 1)) / timedelta(seconds=1)

            timestamps_unix = np.random.randint(start_unix, end_unix, int(n))
            timestamps_unix = sorted(timestamps_unix)

            return pd.to_datetime(timestamps_unix, unit='s').to_frame(name='timestamp')

        gen_cases = list()
        for row in gen.itertuples(index=False):
            gen_cases.append(pp(row.ds, row.gen_round))
        times = pd.concat(gen_cases, axis=0, ignore_index=True)
        # times = times.iloc[:num_instances]
        case_arrival_times = list(times['timestamp'])

        return case_arrival_times

    def get_time_series_df(self,):
        """
        Convert train arrivals into a pandas df, aggregated by the number of arrivals for each unique hour.
        The column 'ds' represents the hour and 'y' represents the number of arrivals in that hour.
        The attributes 'cap' and 'floor' are need because we use a logistic growth model.
        """
        # Convert timestamp data to a pandas DataFrame
        df = pd.DataFrame(self.train, columns=['timestamp'])

        # Convert the 'timestamp' column to pandas datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f').dt.tz_localize(None)

        # Aggregating data: Number of arrivals per hour
        df = df.set_index('timestamp').resample('D').size().reset_index(name='arrivals')

        # Rename columns to fit Prophet's expected format
        df.rename(columns={'timestamp': 'ds', 'arrivals': 'y'}, inplace=True)

        # fill missing values with 0
        df = df.fillna(0)

        max_cap = df.y.max() * 1.2
        df['cap'] = max_cap
        df['floor'] = 0
        df['y'] = df['y'].astype(np.float64)
        df['cap'] = df['cap'].astype(np.float64)
        df['floor'] = df['floor'].astype(np.float64)

        return df, max_cap
    
    def train_prophet_model(self, df):
        days = df.ds.max() - df.iloc[int(len(df) * 0.8)].ds
        periods = days * 0.5

        param_grid = {'changepoint_prior_scale': [0.1],# [0.001, 0.01, 0.1, 0.5],
                        'seasonality_prior_scale': [0.1],#[0.01, 0.1, 1.0, 10.0]
                        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            params['growth'] = 'logistic'
            try:
                m = Prophet(**params).fit(df)  # Fit model with given params
                df_cv = cross_validation(m, horizon=days, period=periods, parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])
            except:
                # df_train.to_csv('df_train_fail.csv')
                traceback.print_exc()
                pass

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        best_params = all_params[np.argmin(rmses)]
        m = Prophet(**best_params).fit(df)

        return m