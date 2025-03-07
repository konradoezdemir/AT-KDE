import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from datetime import timedelta

class ChronosIATGenerator():
    """
    Generates inter arrival times by fitting a Chronos model to the training data
    """

    def __init__(self, train_arrival_times, data_n_seqs) -> None:
        self.train = train_arrival_times
        self.n_seqs = data_n_seqs

    def generate_arrivals(self, start_time):
        time_series_df, test_df = self.get_time_series_df(start_time)
        print(time_series_df)
        # transform the arrival counts into a torch tensor
        time_series_tensor = torch.tensor(time_series_df.values)

        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )
        forecast = pipeline.predict(
            context=time_series_tensor,
            prediction_length=len(test_df),
            num_samples=1,
            limit_prediction_length=False,
        )
        # print(forecast[0])
        forecast_index = range(len(time_series_df), len(time_series_df) + 12)
        low, median, high = np.quantile(forecast[0].numpy(), [0.1, 0.5, 0.9], axis=0)
        median_forecast = np.round(median).astype(int)
        # print(median_forecast)
        # print(ChronosPipeline.predict.__doc__)

        test_df['arrivals'] = median_forecast
        print(test_df)


        def pp(start, n):
            start_u = int(start.value // 10 ** 9)
            end_u = int((start + timedelta(hours=1)).value // 10 ** 9)
            return pd.to_datetime(np.random.randint(start_u, end_u, int(n)), unit='s').to_frame(name='timestamp')
        
        gen_cases = list()
        for idx, row in test_df.iterrows():
            start_time = idx  # This is the timestamp
            count = row['arrivals']
            gen_cases.append(pp(start=start_time, n=count))
        times = pd.concat(gen_cases, axis=0, ignore_index=True)
        # times = times.iloc[:num_instances]
        case_arrival_times = list(times['timestamp'])

        return case_arrival_times


    def get_time_series_df(self, start_time):
        """
        Convert train arrivals into a pandas df, aggregated by the number of arrivals for each unique hour.
        Create a test df with the timestamp as index and 0 values as placeholder; the start_time should be the first timestamp in the df
        """ 
        df = pd.DataFrame(self.train, columns=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f').dt.tz_localize(None)
        df = df.set_index('timestamp').resample('h').size()

        start_time = start_time.replace(minute=0, second=0, microsecond=0)
        test_df = pd.DataFrame(columns=['timestamp', 'arrivals'])
        test_df['timestamp'] = pd.date_range(start=start_time, end=(start_time + timedelta(days=self.n_seqs+1)).replace(hour=0, minute=0, second=0), freq='h')
        test_df['arrivals'] = 0
        test_df = test_df.set_index('timestamp')

        return df, test_df