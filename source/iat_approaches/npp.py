import pandas as pd
from datetime import datetime, timedelta
import pytz
from datetime import timezone
from npp_core import NonHomogeneousHawkes
from source.arrival_distribution import get_min_max_time_per_day, random_sample_timestamps, increment_day_of_week, get_average_occurence_of_cases_per_day, get_boundaries_of_day
from utils.helper import get_arrival_likelihood_per_day, sample_arrival, transform_to_float, transform_to_timestamp

class NPPIATGenerator():
    """
    Generates inter arrival times by fitting a Hawkes Process
    """

    def __init__(self, train_arrival_times, data_n_seqs, probabilistic_day) -> None:
        self.train = train_arrival_times
        self.n_seqs = data_n_seqs
        
        self.lower_bound, self.upper_bound = get_boundaries_of_day(self.train)
        self.probabilistic_day = probabilistic_day
        
        if probabilistic_day == "True":
            self.arrival_likelihood = get_arrival_likelihood_per_day(self.train)
        else: 
            self.arrival_likelihood = None


    def generate_arrivals(self, start_time):
        
        hawkes_model = NonHomogeneousHawkes(n_bins=10)
    
        # Fit the model to the data
        train_grouped_floats = transform_to_float(self.train)
        
        hawkes_model.fit(train_grouped_floats, progress = True)
        # Generate new synthetic data (list of realizations)
        generated_grouped_floats = hawkes_model.generate(n = self.n_seqs,T_start = self.lower_bound/3600, T_end=self.upper_bound/3600)
        
        case_arrival_times = transform_to_timestamp(
                                                        generated_times=generated_grouped_floats, 
                                                        start_timestamp=start_time,
                                                        probabilistic=self.probabilistic_day,
                                                        arrival_likelihood=self.arrival_likelihood
                                                    )
        # sample case start timestamps
        # case_arrival_times = self.get_case_arrival_times_synthetic(start_timestamp=start_time, num_sequences_to_simulate=self.n_seqs)

        return case_arrival_times
    
    def get_min_max_timestamp(self, current_timestamp):
        day = datetime.combine(current_timestamp.date(), datetime.min.time())
        timestamp_first = (day + timedelta(seconds=self.lower_bound)).replace(tzinfo=pytz.UTC)
        timestamp_last = (day + timedelta(seconds=self.upper_bound)).replace(tzinfo=pytz.UTC)

        return timestamp_first, timestamp_last
    
    # def get_case_arrival_times_synthetic(self, start_timestamp, num_sequences_to_simulate):
    #     # Make start_timestamp timezone-aware if it's not already
    #     if start_timestamp.tzinfo is None:
    #         start_timestamp = start_timestamp.replace(tzinfo=timezone.utc)
    #     current_timestamp = start_timestamp
    #     sampled_cases = []
    #     num_sequences = 0
    #     new_day = True

    #     while num_sequences < num_sequences_to_simulate:
    #         if sample_arrival(pd.Timestamp(current_timestamp.date(), tz='UTC'), self.arrival_likelihood) == False and new_day == True:
    #             current_timestamp = current_timestamp + pd.Timedelta(days=1)
    #             new_day = True
    #         else:
    #             if new_day:
    #                 timestamp_first, timestamp_last = self.get_min_max_timestamp(current_timestamp)
                
    #             [duration] = self.arrival_distribution.generate_sample(1)

    #             if (current_timestamp + timedelta(seconds=duration)) <= timestamp_last:
    #                 current_timestamp = current_timestamp + timedelta(seconds=duration)
    #                 sampled_cases.append(current_timestamp)
    #                 new_day = False
    #             else:
    #                 current_timestamp = timestamp_first + timedelta(days=1)
    #                 new_day = True
    #                 num_sequences += 1

    #     return sampled_cases