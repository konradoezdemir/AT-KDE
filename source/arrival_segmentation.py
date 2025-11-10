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
from typing import Iterable, Tuple, List
from datetime import date
from tqdm import tqdm 
from utils.helper import transform_to_float

def tune_sensitivity(list_of_timestamps, window_size=7, max_clusters=6, sensitivity_range=[0.1,0.6,0.7,0.8,0.9,1.0]):
    """
    Iteratively tune an outlier-detection sensitivity to segment arrival timestamps,
    enforcing basic quality checks and a clustering cap.

    Workflow
    --------
    For each `sensitivity` in `sensitivity_range` (in order):
      1) Aggregate arrivals per day and compute sliding-window diffs.
      2) Detect change points (IQR-based outliers) → derive break dates.
      3) Split into segments; optionally add gradual-trend change points.
      4) Validate segments: at least `window_size` days each and ≥2 segments.
      REMOVED: 5) Optionally refine the last segment; resegment.
      6) Cluster segments and require `num_clusters ≤ max_clusters`.
    On first sensitivity that passes all checks: merge segments, recluster, save results, and return.

    If no sensitivity passes, falls back to a single segment covering all timestamps,
    prints brief diagnostics, and returns with `finished=False`.

    Parameters
    ----------
    list_of_timestamps : Iterable[pd.Timestamp]
        Arrival times; tz-aware or naive. Order is not required.
    window_size : int, default 7
        Sliding window (days) used for diffs, minimum segment length, and trend detection.
    max_clusters : int, default 6
        Maximum allowed number of clusters produced by `apply_clustering`.
    sensitivity_range : Iterable[float], default [0.1, 1.0, 0.9, 0.8, 0.7, 0.6]
        Candidate IQR sensitivities to try, in descending priority (first valid wins).

    Returns
    -------
    segments_new : list[list[pd.Timestamp]]
        Final list of segments (each a list of timestamps), possibly merged.
    finished : bool
        True if a sensitivity satisfied all checks; False if fallback was used.
    labels : list[int] or np.ndarray
        Cluster labels for `segments_new` as returned by `apply_clustering`.
    relevant_ratio : float
        Summary metric returned by `save_results` for the final segmentation.
    """
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
        
        # last_seg_new = analyze_last_segment(segments_new[-1],sensitivity)
        # break_dates = break_dates.append(last_seg_new)
        # segments_new = get_segments(arrival_df,break_dates)
        
        # Step 4: Cluster the segments and check the number of clusters
        labels = apply_clustering(segments_new,1,1)
        num_clusters = len(np.unique(labels))
        if num_clusters > max_clusters:
            continue  # Skip this sensitivity and try the next one
        
        # Step 5: If all conditions are satisfied, save the results and stop
        finished = True
        print(f'labels:{labels}')
        print(f'[len(s) for s in segments_new]:{[len(s) for s in segments_new]}')
        # segments_new = merge_segments(segments_new, labels) -> not needed anymore as we dont refine last segment in addition
        # labels = apply_clustering(segments_new,1,1)
        # print(f'post merge -- [len(s) for s in segments_new]:{[len(s) for s in segments_new]}')
        # print(f'post merge -- labels:{labels}')
        
        # relevant_ratio = save_results(segments_new, labels, sensitivity)
        #plot_arrivals_per_day(list_of_timestamps, outliers=outliers, window_size=window_size)
        if finished:
            return segments_new, finished, labels#, relevant_ratio
    if not finished:
        # print(f'##########Tuning sensitivity yielded no results, falling back to one segment/cluster.##########')
        relevant_segment = segments[-1] #this will not always hold true!
        # relevant_segment = [s for seg in segments for s in seg] #fallback:all are relevant
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
        # relevant_ratio = save_results(segments_new, labels, sensitivity)
        
        return segments_new, finished, labels#, relevant_ratio

def aggregate_arrivals_per_day(list_of_timestamps):
    """
    Aggregate arrivals per day.
    Args:
        list_of_timestamps (list): List of timestamps.
    Returns:
        pd.DataFrame: DataFrame with date and count of arrivals.
    """
    df = pd.DataFrame(list_of_timestamps, columns=['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df.drop(columns=['timestamp'], inplace=True)
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
            print(f'len(segment):{len(segment)}')
            duration_days = (max(segment) - min(segment)).days
            duration_days += 1
            print(f'duration_days:{duration_days}')
        else:
            duration_days = 0  # Handle empty segments
            
        # Check if the segment's duration meets the minimum length
        if duration_days < min_length_days:
            return False  # If any segment is too short, return False immediately
    
    return True  # All segments meet the minimum length

def save_results(segments, labels, sensitivity):
    """function needs to be overhauled, doesnt work as intended (relevant_segment)"""
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
    # print('\n-----------')
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

def split_dayindex_by_stamp_span(
    stamps: Iterable[pd.Timestamp],
    day_index: pd.DatetimeIndex,
    tz: str = "UTC",
) -> Tuple[List[date], List[date]]:
    """
    From `day_index` (sequence of days), return:
      - in_days:  days whose date lies within [min(stamps_date), max(stamps_date)]
      - out_days: days before min(stamps_date) or after max(stamps_date)

    Dates are computed after aligning both `stamps` and `day_index` to timezone `tz`.
    """
    # Handle empty stamps defensively
    ts = pd.DatetimeIndex(stamps)
    if len(ts) == 0:
        di = day_index.tz_localize(tz) if day_index.tz is None else day_index.tz_convert(tz)
        day_dates = [d.date() for d in di.normalize()]
        return [], day_dates

    # Align tz for stamps
    ts = ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)
    min_date = ts.min().normalize().date()
    max_date = ts.max().normalize().date()

    # Align tz for day_index and convert to date objects (preserve order)
    di = day_index.tz_localize(tz) if day_index.tz is None else day_index.tz_convert(tz)
    day_dates = [d.date() for d in di.normalize()]

    in_days  = [d for d in day_dates if (min_date <= d <= max_date)]
    out_days = [d for d in day_dates if not (min_date <= d <= max_date)]
    return in_days, out_days

def _construct_output_df_segment_flags(date_range, pattern_info, segments, clustered_segments):
    segment_flag = False
    # Take only cluster if there are no segments
    all_dates = date_range #like pd.date_range(start, end)
    if len(segments)==1:
        output = []
        last_known_cluster = clustered_segments[0]
        for date in all_dates:
            output.append((date, last_known_cluster))
        output_df = pd.DataFrame(output, columns=["date", "predicted_cluster"])
        output_df = output_df.sort_values(by="date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        output_df = output_df.set_index('date').reindex(all_dates, method='ffill').reset_index()
        output_df.columns = ["date", "predicted_cluster"]
        return output_df, segment_flag

    # Initialize the last known cluster
    last_known_cluster = pattern_info.get("last_cluster", None)

    output = []
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
                adjusted_date = future_date 
                output.append((adjusted_date, last_known_cluster))
    
    # Convert to DataFrame and ensure dates are unique and sorted
    output_df = pd.DataFrame(output, columns=["date", "predicted_cluster"])
    output_df = output_df.sort_values(by="date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    
    # Fill remaining dates with the last known cluster
    output_df = output_df.set_index('date').reindex(all_dates, method='ffill').reset_index()
    output_df.columns = ["date", "predicted_cluster"]
    
    return output_df, segment_flag

def cluster_calendar_from_segments(
    segments: List[Iterable[pd.Timestamp]],
    clustered_segments: List[int],
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Return a DataFrame mapping each calendar day covered by the segments
    to its cluster label, using the implied continuous day range of each segment.

    Only days that actually appear in the provided segments are retained.
    Output columns: ['date', 'cluster'] with 'date' as midnight-normalized pd.Timestamp (tz-naive).
    """
    if len(segments) != len(clustered_segments):
        raise ValueError("segments and clustered_segments must have the same length.")

    rows = []
    valid_days = set()  # union of actual segment days

    for seg, lab in zip(segments, clustered_segments):
        seg = pd.DatetimeIndex(seg)
        if len(seg) == 0:
            continue
        # Align tz for consistent day comparisons
        if tz is not None:
            seg = seg.tz_localize(tz) if seg.tz is None else seg.tz_convert(tz)

        # Collect actual days present in this segment
        seg_days = seg.normalize().tz_localize(None)  # tz-naive midnights
        valid_days.update(seg_days.unique().tolist())

        # Build continuous range for labeling (min..max days)
        day_min = seg_days.min()
        day_max = seg_days.max()
        days = pd.date_range(day_min, day_max, freq="D")
        rows.extend((pd.Timestamp(d).tz_localize(None), int(lab)) for d in days)

    if not rows:
        return pd.DataFrame(columns=["date", "cluster"])

    cal = (
        pd.DataFrame(rows, columns=["date", "cluster"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")  # last wins on overlaps
        .reset_index(drop=True)
    )

    if valid_days: #make sure invalid days (in date range but not in test) are not propagated
        cal = cal[cal["date"].isin(valid_days)].reset_index(drop=True)

    # print(f'---->cal (tail):\n{cal.tail(20)}')
    return cal

def label_train_timestamps_with_segments(
    train_timestamps: Iterable[pd.Timestamp],
    calendar_df: pd.DataFrame,
    tz: str = "UTC",
) -> pd.DataFrame:
    """
    Map each train timestamp to a cluster via the calendar.
    Returns columns: ['timestamp', 'date', 'cluster'].
    """
    ts = pd.DatetimeIndex(train_timestamps)
    if tz is not None:
        ts = ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)
    df = pd.DataFrame({"timestamp": ts})
    df["date"] = df["timestamp"].normalize().dt.tz_localize(None)
    out = df.merge(calendar_df, on="date", how="left")
    return out

def label_days_with_segments(
    days: Iterable[pd.Timestamp],  # or iterable of datetime.date
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Label a set of days with clusters using the calendar.
    Input 'days' can be date-like; result has ['date','predicted_cluster'].
    """
    days = pd.to_datetime(list(days))
    df = pd.DataFrame({"date": pd.DatetimeIndex(days).normalize().tz_localize(None)})
    df = df.drop_duplicates().sort_values("date")
    df = df.merge(calendar_df, on="date", how="left")
    return df.rename(columns={"cluster": "predicted_cluster"})


def extend_pattern(train, start_date, end_date, segments, clustered_segments, years):
    """
    Generate predicted dates and clusters based on detected patterns or continue last cluster if no pattern.
    
    Parameters:
        train: new 
        end_date: new 
        
        start_date (str): Starting date for predictions (in YYYY-MM-DD format). #not added newly
        segments (list of lists): List of segments, each containing timestamps.
        clustered_segments (list): Cluster labels for each segment.
        years (int): Number of years in the dataset to assess the monthly pattern.
        
        REMOVE: days_to_generate (int): Number of days to extend the pattern.
    Returns:
        output_df contains dates mapped to cluster labels from start to end date
    """
    #incision: to allow for train data simulation, we need to dissect the 
    #range of simulation days into the subset that falls into train itself -> no prediction needed
    #and the part outside -> prediction needed 
    start_date = pd.to_datetime(start_date) #from run_all
    end_date = pd.to_datetime(end_date) #from run_all
    all_dates = pd.date_range(start=start_date, end=end_date)
    
    within_train, out_of_train = split_dayindex_by_stamp_span(train, all_dates, tz="UTC")
    # print(f'----->within_train:{within_train}')
    # Detect pattern using the updated detect_pattern function
    pattern_info = detect_pattern(segments, clustered_segments, years) #info on complete train horizon 
    # {'pattern': 'none', 'last_cluster': 1, 'start_dates': {0: {'day_of_month': 25, 'month': 3}, 1: {'day_of_month': 2, 'month': 5}}}
    
    # Build the segment calendar once
    calendar_df = cluster_calendar_from_segments(segments, clustered_segments, tz="UTC")

    if len(within_train) == 0:
        # Only out-of-train, predict clusters
        # print('Only out-of-train, predict clusters')
        output_df, segment_flag = _construct_output_df_segment_flags(
            pd.to_datetime(out_of_train), pattern_info, segments, clustered_segments
        )

    elif len(out_of_train) == 0:
        # Only within-train, take segment labels 
        # print('Only within-train, take segment labels')
        within_df = label_days_with_segments(within_train, calendar_df)
        output_df, segment_flag = within_df, False

    else:
        # Mixed case: concatenate labeled within-train + predicted out-of-train
        # print('Mixed case: concatenate labeled within-train + predicted out-of-train')
        within_df = label_days_with_segments(within_train, calendar_df)
        predicted_df, segment_flag = _construct_output_df_segment_flags(
            pd.to_datetime(out_of_train), pattern_info, segments, clustered_segments
        )
        output_df = (
            pd.concat([within_df[["date","predicted_cluster"]], predicted_df], axis=0)
            .sort_values("date")
            .drop_duplicates(subset=["date"])
            .reset_index(drop=True)
        )
    #reasons: within_df based on daterange (including not existing dates), cal df has true dates -> merge yields NAs for predicted clusters
    output_df = output_df.dropna() 
        
    return output_df, segment_flag