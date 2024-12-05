import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from collections import Counter
from itertools import groupby
import logging

from utils.helper import transform_to_float


def tune_sensitivity(list_of_timestamps, window_size=7, max_clusters=6, sensitivity_range=[0.1,1.0,0.9,0.8,0.7,0.6]):
    df = aggregate_arrivals_per_day(list_of_timestamps)
    diff_list = sliding_window_diff(df, window_size=window_size)
    finished = False
    for sensitivity in sensitivity_range:
        # Step 1: Detect change points
        outliers = detect_outliers_iqr(diff_list, sensitivity=sensitivity)
        
        # Step 2: Get segments from change points
        arrival_df = create_ts_df(list_of_timestamps)
        break_dates = get_break_dates(outliers,df,window_size)
        segments = get_segments(arrival_df, break_dates)
        for segment in segments:
            trend_change_date, trends = detect_gradual_change(segment,window_size)
            break_dates = break_dates.append(pd.DatetimeIndex([trend_change_date])).sort_values().dropna()
        segments_new = get_segments(arrival_df,break_dates)

        # Step 3b: Check if all segments are at least 7 days long
        if not check_segment_lengths(segments_new, window_size):
            continue  # Skip this sensitivity and try the next one

        # Step 3d: Check if there are no segments
        if len(segments_new)<2:
            continue  # Skip this sensitivity and try the next one

        
        last_seg_new = analyze_last_segment(segments_new[-1],sensitivity)
        break_dates = break_dates.append(last_seg_new)
        segments_new = get_segments(arrival_df,break_dates)
        # Step 4: Cluster the segments and check the number of clusters
        labels = apply_clustering(segments_new,1,1)
        num_clusters = get_number_clusters(labels)
        if num_clusters > max_clusters:
            continue  # Skip this sensitivity and try the next one
        
        # Step 5: If all conditions are satisfied, save the results and stop
        finished = True
        segments_new = merge_segments(segments_new, labels)
        labels = apply_clustering(segments_new,1,1)
        relevant_ratio = save_results(segments_new, labels, sensitivity)
        #plot_arrivals_per_day(list_of_timestamps, outliers=outliers, window_size=window_size)
        if finished:
            return segments_new, finished, labels, relevant_ratio
    if not finished:
        relevant_segment = segments[-1] #this will not always hold true!
        timestamps_sorted = sorted(relevant_segment, key=lambda x: x.date())
        grouped_by_day = [list(group) for _, group in groupby(timestamps_sorted, key=lambda x: x.date())]
        print(f'number of sequences (days): {len(grouped_by_day)}')
        print(f'total number of timestamps: {len([s for seq in grouped_by_day for s in seq])}')
        try:
            print(f'labels:{labels}')
        except:
            print('Nope')
        arrival_df = create_ts_df(list_of_timestamps)
        segments_new = [arrival_df['Arrival_Timestamp'].to_list()]
        labels = apply_clustering(segments_new,1,1)
        relevant_ratio = save_results(segments_new, labels, sensitivity)
        
        return segments_new, finished, labels, relevant_ratio



def aggregate_arrivals_per_day(list_of_timestamps):
    """
    Aggregate arrivals per day.
    Args:
        list_of_timestamps (list): List of timestamps.
    Returns:
        pd.DataFrame: DataFrame with date and count of arrivals.
    """
    df = pd.DataFrame(list_of_timestamps, columns=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    df['date'] = df['timestamp'].dt.date
    df['count'] = 1
    df = df.groupby('date').count().reset_index()
    return df

def create_ts_df(list_of_timestamps):
    """
    Create a time series DataFrame from a list of timestamps.
    """
    df = pd.DataFrame(list_of_timestamps, columns=['Arrival_Timestamp'])
    df['Arrival_Timestamp'] = pd.to_datetime(df['Arrival_Timestamp'])
    df['Count'] =1
    return df

def sliding_window_diff(df, window_size):
    diff_list = []
    for i in range(len(df) - 2*window_size + 1):
        win1 = df['count'].iloc[i:i+window_size].mean()
        win2 = df['count'].iloc[i+window_size:i+2*window_size].mean()
        diff_list.append(win2 - win1)
    return diff_list

def detect_outliers_iqr(data, lower_percentile=15, upper_percentile=85, iqr_multiplier=1.5, sensitivity=1.0):
    # Convert the list to a numpy array
    data = np.array(data)
    
    # Calculate Q1 and Q3 using the specified percentiles
    Q1 = np.percentile(data, lower_percentile)
    Q3 = np.percentile(data, upper_percentile)
    
    # Calculate the IQR (Interquartile Range)
    IQR = Q3 - Q1
    
    # Define outlier bounds, adjusted by sensitivity
    lower_bound = Q1 - iqr_multiplier * IQR * sensitivity
    upper_bound = Q3 + iqr_multiplier * IQR * sensitivity
    
    # Detect outliers and their indices
    outlier_mask = (data < lower_bound) | (data > upper_bound)
    outlier_indices = np.where(outlier_mask)[0]
    
    # Find sequences of consecutive outliers and select the biggest from each
    outlier_sequences = []
    current_sequence = []
    for i, is_outlier in enumerate(outlier_mask):
        if is_outlier:
            current_sequence.append((data[i], i))
        elif current_sequence:
            max_outlier = max(current_sequence, key=lambda x: abs(x[0]))
            outlier_sequences.append(max_outlier)
            current_sequence = []
    
    # Don't forget the last sequence if it exists
    if current_sequence:
        max_outlier = max(current_sequence, key=lambda x: abs(x[0]))
        outlier_sequences.append(max_outlier)
    
    return outlier_sequences

def get_break_dates(outliers, df, window_size):
    break_dates = []
    for outlier in outliers:
        break_dates.append(df['date'][outlier[1]]+timedelta(days=window_size))
    break_dates = pd.to_datetime(break_dates)
    return break_dates

def get_segments(arrival_df, break_dates):
    # Initialize variables for storing the segments
    segments = []
    start_idx = 0

    for break_date in break_dates:
        # Select rows from the previous start index to the current break date
        # segment = arrival_df[(arrival_df['Arrival_Timestamp'] >= arrival_df['Arrival_Timestamp'].iloc[start_idx]) &
        #             (arrival_df['Arrival_Timestamp'] < break_date)]
        # segments.append(segment['Arrival_Timestamp'].to_list())  # Store the segment
        # Convert timestamps to naive datetime (remove timezone info)
        start_time = arrival_df['Arrival_Timestamp'].iloc[start_idx].tz_localize(None)
        break_date = break_date.tz_localize(None)
        
        # Select rows from the previous start index to the current break date
        segment = arrival_df[(arrival_df['Arrival_Timestamp'].dt.tz_localize(None) >= start_time) &
                    (arrival_df['Arrival_Timestamp'].dt.tz_localize(None) < break_date)]
        segments.append(segment['Arrival_Timestamp'].to_list())
        start_idx = arrival_df[arrival_df['Arrival_Timestamp'].dt.tz_localize(None) >= break_date].index[0]  # Update the start index

    # Add the final segment after the last breakpoint
    last_segment = arrival_df[arrival_df['Arrival_Timestamp'] >= arrival_df['Arrival_Timestamp'].iloc[start_idx]]

    segments.append(last_segment['Arrival_Timestamp'].to_list())
    return segments

def compute_day_arrival_features(day):
    if len(day) < 2:
        inter_arrival_times = [0]  # If only one arrival or less, inter-arrival time is zero
    else:
        inter_arrival_times = np.diff(sorted(day))  # Compute inter-arrival times
    num_arrivals = len(day)  # Number of arrivals
    return num_arrivals, inter_arrival_times


# Function to compute segment-level features from the arrivals
def extract_features_from_segments(segments):
    features = []
    for segment in segments:
        num_arrivals_per_day = []
        inter_arrival_times_all_days = []

        # Process each day in the segment
        for day in segment:
            num_arrivals, inter_arrival_times = compute_day_arrival_features(day)
            num_arrivals_per_day.append(num_arrivals)
            inter_arrival_times_all_days.extend(inter_arrival_times)  # Collect all inter-arrival times for the segment

        # Compute segment-level statistics
        segment_features = [
            np.mean(num_arrivals_per_day),          # Mean number of arrivals per day
            #np.std(num_arrivals_per_day),           # Std of number of arrivals per day
            #np.min(num_arrivals_per_day),           # Minimum number of arrivals per day
            #np.max(num_arrivals_per_day),           # Maximum number of arrivals per day
            np.percentile(num_arrivals_per_day, 25),  # 25th percentile of arrivals per day
            np.percentile(num_arrivals_per_day, 75),  # 75th percentile of arrivals per day
            #np.mean(inter_arrival_times_all_days),  # Mean inter-arrival time
            np.std(inter_arrival_times_all_days),   # Std of inter-arrival times
            #np.min(inter_arrival_times_all_days),   # Minimum inter-arrival time
            #np.max(inter_arrival_times_all_days),   # Maximum inter-arrival time
            np.percentile(inter_arrival_times_all_days, 25),  # 25th percentile of inter-arrival times
            np.percentile(inter_arrival_times_all_days, 75),  # 75th percentile of inter-arrival times
        ]
        features.append(segment_features)
    return np.array(features)

def apply_clustering(segments, eps, min_samples):
    processed_segments = []
    for segment in segments:
        numerical_segment = transform_to_float(segment)
        processed_segments.append(numerical_segment)
    features = extract_features_from_segments(processed_segments)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(features_scaled)
    return labels


def detect_gradual_change(list_of_timestamps, window_size):
    df = aggregate_arrivals_per_day(list_of_timestamps)
    # split the df into buckets of size window_size
    buckets = [df['count'].iloc[i:i+window_size] for i in range(0, len(df), window_size)]
    # compute the average number of arrivals within each bucket
    avg_arrivals = [bucket.mean() for bucket in buckets]
    # compute the trend as the ratio of the current average to the first average
    trends = []
    for i in range(len(avg_arrivals)-1):
        trends.append(avg_arrivals[i+1] / avg_arrivals[0])
    
    def detect_change_points_in_trends(trends):
        # check if 4 consecutive values in trends are outside the threshold
        threshold_max = 1.0 + 0.33
        threshold_min = 1.0 - 0.33
        for i in range(len(trends)-4):
            if trends[i] > threshold_max and trends[i+1] > threshold_max and trends[i+2] > threshold_max and trends[i+3] > threshold_max:
                change_point = i
                return change_point
            elif trends[i] < threshold_min and trends[i+1] < threshold_min and trends[i+2] < threshold_min and trends[i+3] < threshold_min:
                change_point = i
                return change_point
        return None
 
    change_point = detect_change_points_in_trends(trends)
    if change_point:
        # get the date of the change point. Note that the change point is the week of the change, not the day
        date = df['date'].iloc[change_point*window_size]
    else:
        date = None
    return date, trends


def check_segment_lengths(segments, min_length_days):
    # Check if all segments have a duration of at least `min_length_days`
    for segment in segments:
        # Ensure there are timestamps in the segment to calculate duration
        if segment:
            # Calculate duration in days between the first and last timestamps in each segment
            duration_days = (max(segment) - min(segment)).days
            duration_days += 1
        else:
            duration_days = 0  # Handle empty segments
            
        # Check if the segment's duration meets the minimum length
        if duration_days < min_length_days:
            return False  # If any segment is too short, return False immediately
    
    return True  # All segments meet the minimum length


def get_number_clusters(labels):
    unique_values = np.unique(labels)
    num_unique_values = len(unique_values)
    return num_unique_values


def save_results(segments, labels, sensitivity):
    # Save segment times and sizes, or other relevant info
    segment_sizes = [len(seg) for seg in segments]
    
    relevant_segment = segments[-1] #this will not always hold true!
    timestamps_sorted = sorted(relevant_segment, key=lambda x: x.date())
    grouped_by_day = [list(group) for _, group in groupby(timestamps_sorted, key=lambda x: x.date())]
    
    logging.basicConfig(level=logging.INFO, format='%(filename)s - %(message)s')
    logger = logging.getLogger(__name__)
    # for li in grouped_by_day:
    #     logger.info(f'[l.date() for l in li]: {[l.date() for l in li]}')
    #     print('\n')
    print('\n-----------')
    n_seqs_relevant_train = len(grouped_by_day)
    n_total_timestamps_relevant_train = len([s for seq in grouped_by_day for s in seq])
    logger.info(f'number of sequences (days): {n_seqs_relevant_train}')
    logger.info(f'total number of timestamps: {n_total_timestamps_relevant_train}')
    relevant_ratio = n_total_timestamps_relevant_train/ n_seqs_relevant_train
    logger.info(f"Saving results for sensitivity {sensitivity}:")
    logger.info(f"Segment sizes: {segment_sizes}")
    logger.info(f"Number of time segment-clusters:{get_number_clusters(labels)}")

    result_df = pd.DataFrame({
        'Segment': [f'Segment {i+1}' for i in range(len(segments))],
        'Cluster': labels
    })

    logger.info(f"\nClustered Segments:{result_df}")
    
    return relevant_ratio


def merge_segments(timestamps, cluster_labels):
    # Initialize the merged_segments list
    merged_segments = []
    
    # Start with the first segment and label
    current_segment = timestamps[0]
    current_label = cluster_labels[0]
    
    for i in range(1, len(timestamps)):
        next_segment = timestamps[i]
        next_label = cluster_labels[i]
        
        # Check if the current segment's last timestamp matches the next segment's first timestamp
        # and if the cluster labels are the same
        if current_label == next_label:
            # Merge the segments by extending the current segment
            current_segment += next_segment  # Append all timestamps except the first (to avoid duplication)
        else:
            # If segments can't be merged, save the current one and start a new one
            merged_segments.append(current_segment)
            current_segment = next_segment
            current_label = next_label
    
    # Append the last segment
    merged_segments.append(current_segment)
    
    return merged_segments


def analyze_last_segment(segment, sensitivity):
    seg_df = aggregate_arrivals_per_day(segment)
    diff_seg_list = sliding_window_diff(seg_df,3)
    outliers_seg = detect_outliers_iqr(diff_seg_list, sensitivity=sensitivity)
    break_dates_seg = get_break_dates(outliers_seg,seg_df,3)
    return break_dates_seg


def get_timeframe_years(list_of_timestamps):
    timestamps = [pd.Timestamp(ts) for ts in list_of_timestamps]

    # Find the minimum and maximum timestamps
    min_timestamp = min(timestamps)
    max_timestamp = max(timestamps)

    # Calculate the years parameter
    years = (max_timestamp.year - min_timestamp.year) + (max_timestamp.month - min_timestamp.month)*(1/12) + (1/12)
    return years

def detect_pattern(segments, clustered_segments, years):
    """
    Detects if there's a repeated pattern in clusters based on sequence, segment lengths, and start dates.
    Returns a dictionary indicating pattern type, cluster sequence, and segment details.
    """
    unique_clusters = np.unique(clustered_segments)
    cluster_counts = {cluster: np.sum(np.array(clustered_segments) == cluster) for cluster in unique_clusters}
    
    # Check if the sequence of clusters repeats itself
    cluster_sequence = ''.join(map(str, clustered_segments))
    repeated_pattern = ""
    
    for length in range(1, len(clustered_segments) // 2 + 1):
        candidate = cluster_sequence[:length]
        if cluster_sequence == candidate * (len(cluster_sequence) // length):
            repeated_pattern = candidate
            break
    
    # Extract the start day of the month and month for each segment
    start_days_months = [(segment[0].day, segment[0].month) for segment in segments]
    
    # Group start day and month by cluster
    cluster_start_dates = {cluster: [] for cluster in set(clustered_segments)}
    for i, cluster in enumerate(clustered_segments):
        cluster_start_dates[cluster].append(start_days_months[i])
    
    # Find the most common start day and month for each cluster
    common_start_dates = {}
    for cluster, dates in cluster_start_dates.items():
        most_common_day, most_common_month = Counter(dates).most_common(1)[0][0]
        common_start_dates[cluster] = {
            'day_of_month': most_common_day,
            'month': most_common_month
        }
    
    # Check monthly pattern validity
    monthly_pattern = all(val > np.floor(5*years) for val in cluster_counts.values())
    if monthly_pattern:
        return {"pattern": "monthly", "sequence": repeated_pattern, "start_dates": common_start_dates}
    elif repeated_pattern:
        return {"pattern": "repeated", "sequence": repeated_pattern, "start_dates": common_start_dates}
    
    # No clear pattern
    return {"pattern": "none", "last_cluster": clustered_segments[-1], "start_dates": common_start_dates}


def extend_pattern(start_date, days_to_generate, segments, clustered_segments, years):
    """
    Generate predicted dates and clusters based on detected patterns or continue last cluster if no pattern.
    
    Parameters:
        start_date (str): Starting date for predictions (in YYYY-MM-DD format).
        days_to_generate (int): Number of days to extend the pattern.
        segments (list of lists): List of segments, each containing timestamps.
        clustered_segments (list): Cluster labels for each segment.
        years (int): Number of years in the dataset to assess the monthly pattern.
        
    Returns:
        DataFrame with predicted dates and clusters.
    """
    # Initialize segment flag
    segment_flag = False

    # Take only cluster if there are no segments
    if len(segments)==1:
        start_date = pd.to_datetime(start_date)
        end_date = start_date + timedelta(days=days_to_generate - 1)
        all_dates = pd.date_range(start=start_date, end=end_date)
        output = []
        last_known_cluster = clustered_segments[0]
        for date in all_dates:
            output.append((date, last_known_cluster))
        output_df = pd.DataFrame(output, columns=["date", "predicted_cluster"])
        output_df = output_df.sort_values(by="date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        output_df = output_df.set_index('date').reindex(all_dates, method='ffill').reset_index()
        output_df.columns = ["date", "predicted_cluster"]
        return output_df, segment_flag

    # Detect pattern using the updated detect_pattern function
    pattern_info = detect_pattern(segments, clustered_segments, years)
    start_date = pd.to_datetime(start_date)
    end_date = start_date + timedelta(days=days_to_generate - 1)
    
    # Generate the full date range for the output
    all_dates = pd.date_range(start=start_date, end=end_date)
    output = []
    
    # Initialize the last known cluster
    last_known_cluster = pattern_info.get("last_cluster", None)
    
    # Extend based on the detected pattern type
    if pattern_info["pattern"] == "monthly":
        common_start_dates = pattern_info["start_dates"]#
        for future_date in all_dates:
            # Check all clusters for the corresponding month
            for cluster, start_info in common_start_dates.items():
                # Check if the day of the month is valid for the future date's month
                if start_info['day_of_month'] <= future_date.days_in_month:
                    # Predict the date for the current cluster
                    prediction_date = future_date.replace(day=start_info['day_of_month'])
                    output.append((prediction_date, cluster))
                    last_known_cluster = cluster
    
    elif pattern_info["pattern"] == "repeated":
        repeated_sequence = list(map(int, pattern_info["sequence"]))
        common_start_dates = pattern_info["start_dates"]
        
        for i, future_date in enumerate(all_dates):
            cluster = repeated_sequence[i % len(repeated_sequence)]
            start_info = common_start_dates.get(cluster, None)
            if start_info:
                # Set prediction date to align with the common start day of each cluster
                if start_info['day_of_month'] <= future_date.days_in_month:
                    prediction_date = future_date.replace(day=start_info['day_of_month'], month=start_info['month'])
                    output.append((prediction_date, cluster))
                    last_known_cluster = cluster  # Update last known cluster
    
    else:
        # No clear pattern; continue the last cluster if sufficiently long
        if len(aggregate_arrivals_per_day(segments[-1]))<7:
            last_known_cluster = clustered_segments[-2]
            segment_flag = True
        start_info = pattern_info["start_dates"].get(last_known_cluster, None)
        if start_info:
            for future_date in all_dates:
                # Use the last cluster for all days in the range
                #if start_info['day_of_month'] <= future_date.days_in_month:
                adjusted_date = future_date #.replace(day=start_info['day_of_month'], month=start_info['month'])
                output.append((adjusted_date, last_known_cluster))
    
    # Convert to DataFrame and ensure dates are unique and sorted
    output_df = pd.DataFrame(output, columns=["date", "predicted_cluster"])
    output_df = output_df.sort_values(by="date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    
    # Fill remaining dates with the last known cluster
    output_df = output_df.set_index('date').reindex(all_dates, method='ffill').reset_index()
    output_df.columns = ["date", "predicted_cluster"]
    
    return output_df, segment_flag