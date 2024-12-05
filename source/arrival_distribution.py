import pandas as pd
import numpy as np
from typing import List

import math
import sys
from collections import Counter
# from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional

import scipy.stats as st
from scipy.stats import wasserstein_distance


def get_inter_arrival_times(event_log: pd.DataFrame) -> List[float]:
    # Get the arrival times from the event log
    arrival_times = []
    for case_id, events in event_log.groupby('case_id'):
        arrival_times += [events['start_timestamp'].min()]
    # Sort them
    arrival_times.sort()
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

def remove_outliers(data: list, m: float = 20.0) -> list:
    """
    Remove outliers from a list of values following the approach presented in https://stackoverflow.com/a/16562028.
    :param data: list of values.
    :param m: maximum ratio between the difference (value - median) and the median of these differences, to NOT be
    considered an outlier. Decreasing the [m] ratio increases the number of detected outliers (observations closer
    to the median are considered as outliers).
    :return: the received list of values without the outliers.
    """
    # Compute distance of each value from the median
    data = np.asarray(data)
    d = np.abs(data - np.median(data))
    # Compute the median of these distances
    mdev = np.median(d)
    # Compute the ratio between each distance and the median of distances
    s = d / (mdev if mdev else 1.0)
    # Keep values with a ratio lower than the specified threshold
    return data[s < m].tolist()



class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "norm"
    TRIANGULAR = "triang"
    EXPONENTIAL = "expon"
    LOG_NORMAL = "lognorm"
    GAMMA = "gamma"
    FIXED = "fix"

    @staticmethod
    def from_string(value: str) -> "DistributionType":
        name = value.lower()
        if name == "uniform":
            return DistributionType.UNIFORM
        elif name in ("norm", "normal"):
            return DistributionType.NORMAL
        elif name in ("triang", "triangular"):
            return DistributionType.TRIANGULAR
        elif name in ("expon", "exponential"):
            return DistributionType.EXPONENTIAL
        elif name in ("lognorm", "log_normal", "lognormal"):
            return DistributionType.LOG_NORMAL
        elif name == "gamma":
            return DistributionType.GAMMA
        elif name in ["fix", "fixed"]:
            return DistributionType.FIXED
        else:
            raise ValueError(f"Unknown distribution: {value}")



class DurationDistribution:
    def __init__(
        self,
        name: Union[str, DistributionType] = "fix",
        mean: Optional[float] = None,
        var: Optional[float] = None,
        std: Optional[float] = None,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None
    ):
        self.type = DistributionType.from_string(name) if isinstance(name, str) else name
        self.mean = mean
        self.var = var
        self.std = std
        self.min = minimum
        self.max = maximum

    def generate_sample(self, size: int) -> list:
        """
        Generates a sample of [size] elements following this (self) distribution parameters. The elements are
        positive, and within the limits [self.min, self.max].

        :param size: number of elements to add to the sample.
        :return: list with the elements of the sample.
        """
        # Instantiate empty sample list
        sample = []
        i = 0  # flag for iteration limit
        # Generate until full of values within limits
        while len(sample) < size and i < 100:
            # Generate missing elements
            local_sample = self._generate_raw_sample(size - len(sample))
            # Filter out negative and out of limits elements
            local_sample = [element for element in local_sample if element >= 0.0]
            if self.min is not None:
                local_sample = [element for element in local_sample if element >= self.min]
            if self.max is not None:
                local_sample = [element for element in local_sample if element <= self.max]
            # Add generated elements to sample
            sample += local_sample
            i += 1
        # Check if all elements got generated
        if len(sample) < size:
            default_value = self._replace_out_of_bounds_value()
            print(f"Warning when generating sample of distribution {self}. "
                  "Too many iterations generating durations out of the distribution limits! "
                  f"Filling missing values with default ({default_value})!")
            sample += [default_value] * (size - len(sample))
        # Return complete sample
        return sample

    def _generate_raw_sample(self, size: int) -> list:
        """
        Generates a sample of [size] elements following this (self) distribution parameters not ensuring that the
        returned elements are within the interval [self.min, self.max] and positive.

        :param size: number of elements to add to the sample.
        :return: list with the elements of the sample.
        """
        sample = []
        if self.type == DistributionType.FIXED:
            sample = [self.mean] * size
        elif self.type == DistributionType.EXPONENTIAL:
            # 'loc' displaces the samples, a loc=100 will be the same as adding 100 to each sample taken from a loc=1
            scale = self.mean - self.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = self.mean
            sample = st.expon.rvs(loc=self.min, scale=scale, size=size)
        elif self.type == DistributionType.NORMAL:
            sample = st.norm.rvs(loc=self.mean, scale=self.std, size=size)
        elif self.type == DistributionType.UNIFORM:
            sample = st.uniform.rvs(loc=self.min, scale=self.max - self.min, size=size)
        elif self.type == DistributionType.LOG_NORMAL:
            # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            pow_mean = pow(self.mean, 2)
            phi = math.sqrt(self.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=size)
        elif self.type == DistributionType.GAMMA:
            # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
            # dunno how to take that into account
            sample = st.gamma.rvs(
                pow(self.mean, 2) / self.var,
                loc=0,
                scale=self.var / self.mean,
                size=size,
            )
        return sample

    def _replace_out_of_bounds_value(self):
        new_value = None
        if self.mean is not None and self.mean >= 0.0:
            # Set to mean
            new_value = self.mean
        if self.min is not None and self.max is not None:
            # If we have boundaries, check that mean is inside
            if new_value is None or new_value < self.min or new_value > self.max:
                # Invalid, set to middle between min and max
                new_value = self.min + ((self.max - self.min) / 2)
        if new_value is None or new_value < 0.0:
            new_value = 0.0
        # Return fixed value
        return new_value

    def scale_distribution(self, alpha: float) -> "DurationDistribution":
        return DurationDistribution(
            name=self.type,
            mean=self.mean * alpha,  # Mean: scaled by multiplying by [alpha]
            var=self.var * alpha * alpha,  # Variance: scaled by multiplying by [alpha]^2
            std=self.std * alpha,  # STD: scaled by multiplying by [alpha]
            minimum=self.min * alpha,  # Min: scaled by multiplying by [alpha]
            maximum=self.max * alpha,  # Max: scaled by multiplying by [alpha]
        )



    @staticmethod
    def from_dict(distribution_dict: dict) -> "DurationDistribution":
        """
        Deserialize function distribution provided as a dictionary
        """
        distr_name = DistributionType(distribution_dict["distribution_name"])
        distr_params = distribution_dict["distribution_params"]

        distribution = None

        if distr_name == DistributionType.FIXED:
            # no min and max values
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
            )
        elif distr_name == DistributionType.EXPONENTIAL:
            # TODO: discuss whether we need to differentiate min and scale.
            # right now, min is used for calculating the scale
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                minimum=float(distr_params[1]["value"]),
                maximum=float(distr_params[2]["value"]),
            )
        elif distr_name == DistributionType.UNIFORM:
            distribution = DurationDistribution(
                name=distr_name,
                minimum=float(distr_params[0]["value"]),
                maximum=float(distr_params[1]["value"]),
            )
        elif distr_name == DistributionType.NORMAL:
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                std=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        elif distr_name == DistributionType.LOG_NORMAL:
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                var=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )
        elif distr_name == DistributionType.GAMMA:
            distribution = DurationDistribution(
                name=distr_name,
                mean=float(distr_params[0]["value"]),
                var=float(distr_params[1]["value"]),
                minimum=float(distr_params[2]["value"]),
                maximum=float(distr_params[3]["value"]),
            )

        return distribution

    def __str__(self):
        return "DurationDistribution(name: {}, mean: {}, var: {}, std: {}, min: {}, max: {})".format(
            self.type.value, self.mean, self.var, self.std, self.min, self.max
        )



def get_best_fitting_distribution(
    data: list,
    filter_outliers: bool = True,
    outlier_threshold: float = 20.0,
) -> DurationDistribution:
    """
    Discover the distribution (exponential, normal, uniform, log-normal, and gamma) that best fits the values in [data].

    :param data: Values to fit a distribution for.
    :param filter_outliers: If true, remove outliers from the sample.
    :param outlier_threshold: Threshold to consider an observation an outlier. Increasing this outlier increases the
                              flexibility of the detection method, i.e., an observation needs to be further from the
                              mean to be considered as outlier.

    :return: the best fitting distribution.
    """
    # Filter outliers
    filtered_data = remove_outliers(data, outlier_threshold) if filter_outliers else data
    # Check for fixed value
    fix_value = _check_fix(filtered_data)
    if fix_value is not None:
        # If it is a fixed value, infer distribution
        distribution = DurationDistribution("fix", fix_value, 0.0, 0.0, fix_value, fix_value)
    else:
        # Otherwise, compute basic statistics and try with other distributions
        mean = np.mean(filtered_data)
        var = np.var(filtered_data)
        std = np.std(filtered_data)
        d_min = min(filtered_data)
        d_max = max(filtered_data)
        # Create distribution candidates
        dist_candidates = [
            DurationDistribution("expon", mean, var, std, d_min, d_max),
            DurationDistribution("norm", mean, var, std, d_min, d_max),
            DurationDistribution("uniform", mean, var, std, d_min, d_max),
        ]
        if mean != 0:
            dist_candidates += [DurationDistribution("lognorm", mean, var, std, d_min, d_max)]
            if var != 0:
                dist_candidates += [DurationDistribution("gamma", mean, var, std, d_min, d_max)]
        # Search for the best one within the candidates
        best_distribution = None
        best_emd = sys.float_info.max
        for distribution_candidate in dist_candidates:
            # Generate a list of observations from the distribution
            generated_data = distribution_candidate.generate_sample(len(filtered_data))
            # Compute its distance with the observed data
            emd = wasserstein_distance(filtered_data, generated_data)
            # Update the best distribution if better
            if emd < best_emd:
                best_emd = emd
                best_distribution = distribution_candidate
        # Set the best distribution as the one to return
        distribution = best_distribution
    # Return best distribution
    return distribution


def _check_fix(data: list, delta=5):
    value = None
    counter = Counter(data)
    counter[None] = 0
    for d1 in counter:
        if (counter[d1] > counter[value]) and (sum([abs(d1 - d2) < delta for d2 in data]) / len(data) > 0.95):
            # If the value [d1] is more frequent than the current fixed one [value]
            # and
            # the ratio of values similar (or with a difference lower than [delta]) to [d1] is more than 90%
            # update value
            value = d1
    # Return fixed value with more apparitions
    return value


### Some further functions
def random_sample_timestamps(date, min_time, max_time, x, arrival_distribution, first_day, start_timestamp):
    # Combine the provided date with the min and max times
    min_timestamp = pd.to_datetime(f'{date} {min_time}', utc=True)
    max_timestamp = pd.to_datetime(f'{date} {max_time}', utc=True)
    # print(f'min_timestamp: {min_timestamp}')
    # print(f'max_timestamp: {max_timestamp}\n')
    sampled_case_starting_times = []
    if first_day:
        time = start_timestamp
    else:
        time = min_timestamp
    sampled_case_starting_times.append(time)
    smaller_than_max_time = True
    while smaller_than_max_time:# and len(sampled_case_starting_times) < x:
        if arrival_distribution.type.value == "expon":
            scale = arrival_distribution.mean - arrival_distribution.min
            if scale < 0.0:
                print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
                scale = arrival_distribution.mean
            sampled_times_between_cases = st.expon.rvs(loc=arrival_distribution.min, scale=scale, size=1)
        elif arrival_distribution.type.value == "gamma":
            sampled_times_between_cases = st.gamma.rvs(
                pow(arrival_distribution.mean, 2) / arrival_distribution.var,
                loc=0,
                scale=arrival_distribution.var / arrival_distribution.mean,
                size=1,
            )
        elif arrival_distribution.type.value == "norm":
            sampled_times_between_cases = st.norm.rvs(loc=arrival_distribution.mean, scale=arrival_distribution.std, size=1)
        elif arrival_distribution.type.value == "uniform":
            sampled_times_between_cases = st.uniform.rvs(loc=arrival_distribution.min, scale=arrival_distribution.max - arrival_distribution.min, size=1)
        elif arrival_distribution.type.value == "lognorm":
            pow_mean = pow(arrival_distribution.mean, 2)
            phi = math.sqrt(arrival_distribution.var + pow_mean)
            mu = math.log(pow_mean / phi)
            sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
            sampled_times_between_cases = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
        elif arrival_distribution.type.value == "fix":
            sampled_times_between_cases = [arrival_distribution.mean] * 1
    
        for j in range(len(sampled_times_between_cases)): #len 1
            time = sampled_case_starting_times[-1] + pd.Timedelta(seconds=sampled_times_between_cases[j])
            # print(f"time: {time}")
            if time >= min_timestamp and time <= max_timestamp:
                sampled_case_starting_times.append(time)
            elif time > max_timestamp:
                smaller_than_max_time = False

    sampled_case_starting_times = sorted(sampled_case_starting_times)
    
    return sampled_case_starting_times

def increment_day_of_week(day_of_week):
    # Define a mapping of days of the week to their numerical representation
    days_mapping = {
        'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2,
        'THURSDAY': 3, 'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6
    }

    # Get the numerical representation of the current day
    current_day_numeric = days_mapping[day_of_week]

    # Increment the day by one (circular increment, so Sunday wraps around to Monday)
    next_day_numeric = (current_day_numeric + 1) % 7

    # Get the corresponding day name for the incremented day
    next_day_of_week = [day for day, value in days_mapping.items() if value == next_day_numeric][0]

    return next_day_of_week

def get_min_max_time_per_day(case_start_timestamps):
    # get the min and max time of case arrivals per day of the week
    days_of_week = [day.strftime('%A').upper() for day in case_start_timestamps]
    days_of_week_set = set(days_of_week)
    min_max_time_per_day = {day: [pd.Timestamp('2023-01-01 23:59:59'),pd.Timestamp('2023-01-01 00:00:01')] for day in days_of_week_set}

    for i in range(len(case_start_timestamps)):
        for key, value in min_max_time_per_day.items():
            if days_of_week[i] == key:
                if case_start_timestamps[i].time() < value[0].time():
                    value[0] = case_start_timestamps[i]
                if case_start_timestamps[i].time() > value[1].time():
                    value[1] = case_start_timestamps[i]
    return min_max_time_per_day

def get_average_occurence_of_cases_per_day(case_start_timestamps):
    # get the mean and std of case arrivals per day of the week
    days_of_week = [day.strftime('%A').upper() for day in case_start_timestamps]
    days_of_week = set(days_of_week)
    count_occurrences_by_day = {day: [] for day in days_of_week}
    day = case_start_timestamps[0].strftime('%A').upper()
    counter = 0
    for time in case_start_timestamps:
        if day == time.strftime('%A').upper():
            counter += 1
        else:
            count_occurrences_by_day[day].append(counter)
            counter = 1
            day = time.strftime('%A').upper()

    average_occurrences_by_day = {day: () for day in days_of_week}
    for key, value in count_occurrences_by_day.items():
        average_occurrences_by_day[key] = (np.mean(value), np.std(value))

    return average_occurrences_by_day



def get_boundaries_of_day(train,):
        earliest_per_day = {}
        last_per_day = {}
        earliest_timestamp = None
        last_timestamp = None

        for ts in train:
            # Extract the date (year, month, day)
            date_key = ts.date()
            
            # Update the dictionary with the earliest timestamp for the day
            if date_key not in earliest_per_day or ts < earliest_per_day[date_key]:
                earliest_per_day[date_key] = ts

            if date_key not in last_per_day or ts > last_per_day[date_key]:
                last_per_day[date_key] = ts
        # print(earliest_per_day)
        # print(last_per_day)

        for date, ts in earliest_per_day.items():
            # Convert date to a Timestamp at midnight to ensure comparable types
            if ts.tzinfo is not None:
                # `ts` is timezone-aware, make `date_as_timestamp` timezone-aware
                date_as_timestamp = pd.Timestamp(date).tz_localize(ts.tzinfo)
            else:
                # `ts` is timezone-naive, make `date_as_timestamp` timezone-naive
                date_as_timestamp = pd.Timestamp(date)
            elapsed_time = (ts - date_as_timestamp).total_seconds()
            if earliest_timestamp == None or earliest_timestamp > elapsed_time:
                earliest_timestamp = elapsed_time

        for date, ts in last_per_day.items():
            # Convert date to a Timestamp at midnight to ensure comparable types
            if ts.tzinfo is not None:
                # `ts` is timezone-aware, make `date_as_timestamp` timezone-aware
                date_as_timestamp = pd.Timestamp(date).tz_localize(ts.tzinfo)
            else:
                # `ts` is timezone-naive, make `date_as_timestamp` timezone-naive
                date_as_timestamp = pd.Timestamp(date)
            elapsed_time = (ts - date_as_timestamp).total_seconds()
            if last_timestamp == None or last_timestamp < elapsed_time:
                last_timestamp = elapsed_time

        # print(earliest_timestamp)
        # print(last_timestamp)

        return earliest_timestamp, last_timestamp