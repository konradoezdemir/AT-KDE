import json 
import numpy as np 
from tqdm import tqdm 
import pandas as pd
import random
from collections import defaultdict
import argparse 
from datetime import timedelta

def is_list_of_lists_of_floats(data):
    if not isinstance(data, list):
        return False
    for sublist in data:
        if not isinstance(sublist, list):
            return False
        for item in sublist:
            if not isinstance(item, float):
                return False
    return True

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    print(f'File correctly read: {path}.')
    return data

def write_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file) 
    print(f'File write to {path} complete.')

def parse_float_list(string): #designed for --params argument to work with lists like [1,2,3]
    """Parse a comma-separated string into a list of floats."""
    return [float(item) for item in string.split(',')]

class KwargsAction(argparse.Action): #designed to enable python script.py --kwargs key1=value1 key2=value2 in an argparser
    def __call__(self, parser, namespace, values, option_string=None):
        kwargs = {}
        for kv in values:
            k, v = kv.split('=')
            kwargs[k] = v
        setattr(namespace, self.dest, kwargs)
        
def MinMaxScaler(data):
    """Min-Max Normalizer.
    
    Args:
        - data: raw data
        
    Returns:
        - norm_data: normalized data of shape (batch_size, seq_len, feature_dim)
        - min_val: minimum values (for renormalization)
        - max_val: maximum values (for renormalization)
    """    
    min_val = np.min(np.min(data, axis = 0), axis = 0) #bias correction 
    data = data - min_val
    
    max_val = np.max(np.max(data, axis = 0), axis = 0) #scale correction (on debiased data)
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val

def InverseMinMaxScaler(normalised_data, min, max):
    unscaled = normalised_data*max
    unbiased = unscaled + min 
    return unbiased

def transform_to_float(arrival_times):
    """
    Transforms arrival timestamps into floats representing hours since midnight,
    grouped by date. Prints a message if any time float exceeds 24 hours.

    Parameters:
    -----------
    arrival_times : list of pd.Timestamp
        List of arrival timestamps.

    Returns:
    --------
    grouped_timestamps_floats : list of lists
        List of lists where each sublist contains time floats for a specific date.
    """
    # Ensure all timestamps are timezone-aware and in UTC
    arrival_times = [
        ts.tz_convert('UTC') if ts.tzinfo else ts.tz_localize('UTC')
        for ts in arrival_times
    ]
    
    # Dictionary to hold lists of timestamps grouped by day
    grouped_by_day = defaultdict(list)
    
    for timestamp in arrival_times:
        # Extract date in UTC
        date_str = timestamp.strftime('%Y-%m-%d')
        grouped_by_day[date_str].append(timestamp)
    
    # Convert timestamps to floats representing hours since midnight
    grouped_timestamps_floats = []
    for date, times in grouped_by_day.items():
        time_floats = []
        for time_obj in times:
            # Calculate time difference from midnight
            midnight = pd.Timestamp(date + ' 00:00:00', tz='UTC')
            time_delta = (time_obj - midnight).total_seconds() / 3600  # Convert seconds to hours

            # Handle negative time differences (if any)
            if time_delta < 0:
                time_delta += 24  # Adjust for times after midnight

            # Print statement if time_delta >= 24
            if time_delta >= 24:
                print(f"Time float exceeds 24 hours: {time_delta} for timestamp {time_obj}")
                # Cap time_float at maximum valid time before midnight
                time_delta = 23 + 59/60 + 59/3600 + 999999/1e6  # Approximately 23.9999997222 hours

            time_floats.append(time_delta)
        grouped_timestamps_floats.append(time_floats)
    
    return grouped_timestamps_floats


def sample_arrival(date, arrival_likelihood):
        if arrival_likelihood == None:
            return True
        base_date_day = date.day_name()
        p = arrival_likelihood[base_date_day]
        arrive_today = random.random() < p
        return arrive_today

def transform_to_timestamp(generated_times, start_timestamp="01.01.2024", probabilistic=False, arrival_likelihood=None):
    """
    Transforms the generated times into actual timestamps.
    
    Parameters:
    - generated_times (list[list[float]]): The generated times.
    - start_timestamp (str): The time when the process should start.
    - probabilistic (str): Whether to consider weekends and other non-working times by probabilistically including zero sequences
    - arrival_likelihood (dict): Arrival likelihoods for each day from Monday to Sunday 

    Returns:
    - list_of_timestamps (list[pd.Timestamp]): A list of timestamps that starts at the date of start_timestamp and represents the generated times.
    Note that this return object does not need to be of form list[list[pd.Timestamp]] since true timestamps
    already contain a day indicator due to their nature. 
    """
    # base timestamp
    start_time = pd.to_datetime(start_timestamp, format='mixed', utc=True)
    base_date = pd.Timestamp(start_time.date(), tz='UTC')

    list_of_timestamps = []

    seq_counter = 0 # for checking if it's the first day to consider exact start timestamp

    if probabilistic == False:
        for seq in generated_times:
            for time in seq:
                total_seconds = time * 3600
                delta = pd.Timedelta(seconds=total_seconds)
                if seq_counter == 0:
                    if start_time <= base_date + delta:
                        list_of_timestamps.append(base_date + delta)
                else:
                    list_of_timestamps.append(base_date + delta)
            base_date = base_date + pd.Timedelta(days=1)
            seq_counter += 1

    else:
        for seq in generated_times:
            while sample_arrival(base_date, arrival_likelihood) == False:
                base_date = base_date + pd.Timedelta(days=1)

            for time in seq:
                total_seconds = time * 3600
                delta = pd.Timedelta(seconds=total_seconds)
                if seq_counter == 0:
                    if start_time <= base_date + delta:
                        list_of_timestamps.append(base_date + delta)
                else:
                    list_of_timestamps.append(base_date + delta)
            base_date = base_date + pd.Timedelta(days=1)
            seq_counter += 1

    return list_of_timestamps 


def get_arrival_likelihood_per_day(arrivals):
        arrivals = pd.Series(arrivals)
        arrivals = arrivals.dt.date.unique()

        probs = {
            'Monday': [0,0],
            'Tuesday': [0,0],
            'Wednesday': [0,0],
            'Thursday': [0,0],
            'Friday': [0,0],
            'Saturday': [0,0],
            'Sunday': [0,0],
        }

        curr_date = arrivals[0] 

        for i in range(len(arrivals)):    
            if arrivals[i] == curr_date: 
                day = arrivals[i].strftime('%A')
                probs[day][0] += 1
                probs[day][1] += 1
            else:
                day = curr_date.strftime('%A')
                while arrivals[i] > curr_date:
                    probs[day][1] += 1
                    curr_date = curr_date + timedelta(days=1)
                    day = curr_date.strftime('%A')
                probs[day][0] += 1
                probs[day][1] += 1
            curr_date = curr_date + timedelta(days=1)

        frequencies = probs
        for key, value in probs.items():
            frequencies[key] = value[0] / value[1]

        return frequencies

def get_inter_arrival_times_from_list_of_timestamps(arrival_times):
    current_day = arrival_times[0].strftime('%Y-%m-%d')
    # Compute durations between one arrival and the next one (inter-arrival durations)
    new_day = []
    inter_arrival_durations = []
    last_arrival = None
    for arrival in arrival_times:
        if last_arrival:
            if arrival.strftime('%Y-%m-%d') == current_day:
                inter_arrival_durations += [(arrival - last_arrival).total_seconds()]
            else:
                new_day.append(arrival)
        last_arrival = arrival
        current_day = arrival.strftime('%Y-%m-%d')

    return inter_arrival_durations