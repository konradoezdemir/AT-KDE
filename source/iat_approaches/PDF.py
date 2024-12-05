import pandas as pd
from datetime import datetime, timedelta
import pytz
from datetime import timezone


from source.arrival_distribution import get_best_fitting_distribution, get_min_max_time_per_day, random_sample_timestamps, increment_day_of_week, get_average_occurence_of_cases_per_day, get_boundaries_of_day
from utils.helper import get_arrival_likelihood_per_day, sample_arrival

class PDFIATGenerator():
    """
    Generates inter arrival times by applying a PDF to the training data.
    This includes the following distinct methods: 
        mean: take the mean inter arrival time as fixed value 
        exponential: fit an exponential distribution
        best_distribution: try different distributions and take the best fitting one
    """

    def __init__(self, train_arrival_times, inter_arrival_durations, arrival_distribution, data_n_seqs, probabilistic_day, kwargs) -> None:
        self.train = train_arrival_times
        self.inter_arrival_durations = inter_arrival_durations
        self.arrival_distribution = arrival_distribution
        self.n_seqs = data_n_seqs

        self.lower_bound, self.upper_bound = get_boundaries_of_day(self.train)
        
        self.kwargs = kwargs

        if probabilistic_day == "True":
            self.arrival_likelihood = get_arrival_likelihood_per_day(self.train)
        else: 
            self.arrival_likelihood = None

        if 'n_seqs' in kwargs:
            self.n_seqs = int(kwargs['n_seqs'])

    def generate_arrivals(self, start_time):
        if self.arrival_distribution == None:
            self.arrival_distribution = get_best_fitting_distribution(
                data=self.inter_arrival_durations,
                filter_outliers=False,
                outlier_threshold=20.0,
            )
        print(f"Distribution: {self.arrival_distribution.type}")
        print(f"Distribution parameters (mean, var, stdev): {self.arrival_distribution.mean, self.arrival_distribution.var, self.arrival_distribution.std}")

        # sample case start timestamps
        case_arrival_times = self.get_case_arrival_times_synthetic(start_timestamp=start_time, num_sequences_to_simulate=self.n_seqs)

        return case_arrival_times
    
    def get_min_max_timestamp(self, current_timestamp):
        day = datetime.combine(current_timestamp.date(), datetime.min.time())
        timestamp_first = (day + timedelta(seconds=self.lower_bound)).replace(tzinfo=pytz.UTC)
        timestamp_last = (day + timedelta(seconds=self.upper_bound)).replace(tzinfo=pytz.UTC)

        return timestamp_first, timestamp_last
    
    def get_case_arrival_times_synthetic(self, start_timestamp, num_sequences_to_simulate):
        # Make start_timestamp timezone-aware if it's not already
        if start_timestamp.tzinfo is None:
            start_timestamp = start_timestamp.replace(tzinfo=timezone.utc)
        current_timestamp = start_timestamp
        sampled_cases = []
        num_sequences = 0
        new_day = True

        while num_sequences < num_sequences_to_simulate:
            if sample_arrival(pd.Timestamp(current_timestamp.date(), tz='UTC'), self.arrival_likelihood) == False and new_day == True:
                current_timestamp = current_timestamp + pd.Timedelta(days=1)
                new_day = True
            else:
                if new_day:
                    timestamp_first, timestamp_last = self.get_min_max_timestamp(current_timestamp)
                
                [duration] = self.arrival_distribution.generate_sample(1)

                if (current_timestamp + timedelta(seconds=duration)) <= timestamp_last:
                    current_timestamp = current_timestamp + timedelta(seconds=duration)
                    sampled_cases.append(current_timestamp)
                    new_day = False
                else:
                    current_timestamp = timestamp_first + timedelta(days=1)
                    new_day = True
                    num_sequences += 1

        return sampled_cases
