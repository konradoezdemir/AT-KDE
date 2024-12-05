import datetime
import math

import pandas as pd
from scipy.stats import wasserstein_distance

from log_distance_measures.config import EventLogIDs, DistanceMetric, discretize_to_hour, discretize_to_day, discretize_to_minute#, discretize_to_second
from log_distance_measures.earth_movers_distance import earth_movers_distance

def case_arrival_distribution_distance_custom(
        original_arrival_times, 
        simulated_arrival_times,
        bin='second',  # function to discretize a total amount of seconds into bins
        metric: DistanceMetric = DistanceMetric.WASSERSTEIN,
        normalize: bool = False
):
    # determine bin size
    if bin == 'second':
        discretize_event = discretize_to_second
    elif bin == 'hour':
        discretize_event = discretize_to_hour
    elif bin == 'day':
        discretize_event = discretize_to_day
    elif bin == 'minute':
        discretize_event = discretize_to_minute
    # Get the first arrival to normalize
    first_arrival = min(
        min(original_arrival_times),
        min(simulated_arrival_times)
    ).floor(freq='S')

    # Discretize each event to its corresponding "bin"
    original_discrete_arrivals = []
    for arrival in original_arrival_times:
        difference = arrival - first_arrival
        original_discrete_arrivals.append(discretize_event(difference.total_seconds()))

    simulated_discrete_arrivals = []
    for arrival in simulated_arrival_times:
        difference = arrival - first_arrival
        simulated_discrete_arrivals.append(discretize_event(difference.total_seconds()))

    # Compute distance metric
    if metric == DistanceMetric.EMD:
        distance = earth_movers_distance(original_discrete_arrivals, simulated_discrete_arrivals) / len(original_discrete_arrivals)
    else:
        distance = wasserstein_distance(original_discrete_arrivals, simulated_discrete_arrivals)
    # Normalize if needed
    if normalize:
        print("WARNING! The normalization of a Wasserstein Distance is sensitive to the range of the two samples, "
              "long samples may cause a higher reduction of the error.")
        max_value = max(max(original_discrete_arrivals), max(simulated_discrete_arrivals))
        distance = distance / max_value if max_value > 0 else distance
    # Return metric
    print('Computing Root Wasserstein Distance')
    return math.sqrt(distance)

