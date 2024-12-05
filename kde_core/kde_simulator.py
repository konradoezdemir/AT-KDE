import numpy as np
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt
import argparse
import random 
import sys
from pathlib import Path
import os 
import math
import copy
import pandas as pd
from tqdm import tqdm 
from datetime import timedelta, time, datetime
from KDEpy.bw_selection import improved_sheather_jones, silvermans_rule
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
#make the 'utils' module discoverable
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

from utils import read_json, write_json, transform_to_float, is_list_of_lists_of_floats
    
class DataSimulator:
    """
        Simulates timestamp data based on KDE models of interarrival times.

        Parameters
        ----------
        reference_dataset : pandas.DataFrame
            Reference dataset containing timestamps for training.
        domain : str, optional
            Domain name for logging or metadata (default is None).
        reference_data_lengths : dict, optional
            Dictionary of segment lengths in the reference dataset (default is None).
        train_segmented : dict, optional
            Segmented training data by clusters or other criteria (default is None).
        test_segment_estim : pandas.DataFrame, optional
            DataFrame containing test data and predicted clusters (default is None).
        path : bool, optional
            Indicates whether to save intermediate results (default is False).
        bw_smooth_factor : float, optional
            Bandwidth smoothing factor for KDE (default is None).
        bin_size_hours : int, optional
            Size of time bins in hours (default is 3).

        Attributes
        ----------
        ref_data : pandas.DataFrame
            Copy of the reference dataset.
        lower_bound : float
            Lower time bound (hours) based on training data.
        upper_bound : float
            Upper time bound (hours) based on training data.
        diffed_kernel_std_dict : dict
            KDE bandwidths for each cluster and bin.
        diffed_data_dict : dict
            Interarrival time differences for each cluster and bin.
        kde_generation_factor : int
            Factor for generating additional interarrival times (default is 10).
        bin_size_hours : int
            Time bin size in hours.

        Methods
        -------
        prepare_kde_models()
            Prepares KDE models based on the training data.
        sample_kde(n)
            Simulates timestamp data for `n` days using KDE models.
        sample_interarrivals(n_interarrivals, cluster_segment, cluster_weekday, bin_name)
            Samples interarrival times for a specific cluster and bin.
    """
    
    def __init__(self, reference_dataset, domain = None, reference_data_lengths=None, train_segmented = None, test_segment_estim = None, path=False, bw_smooth_factor=None, bin_size_hours=3):
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(filename)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.ref_data = reference_dataset.copy()
        # self.printer_index = 0
        self.train_segmented = train_segmented
        self.test_segment_estim = copy.deepcopy(test_segment_estim)
        self.test_segment_estim['date'] = self.test_segment_estim['date'].dt.date #remove redundant hrs:mins:sec in each date indication
        self.test_segment_estim.set_index('date', inplace = True)
        self.diffed_kernel_std_dict, self.diffed_data_dict  = {}, {}
        self.lower_bound, self.upper_bound = None, None
        self.kde_generation_factor = 10  # factor for generating extra interarrival times
        
        self.bw_smooth_factor = bw_smooth_factor # per (1+factor)*calced_bandwidth
        
        self.bin_size_hours = bin_size_hours

        self.prepare_kde_models()

        #set global upper and lower bound working hours
        self.train_floats = transform_to_float(self.ref_data)
        all_times = [item for sublist in self.train_floats for item in sublist]
        self.lower_bound = math.floor(min(all_times))
        self.upper_bound = min(math.ceil(max(all_times)), 23 + 59/60 + 59/3600 + 999999/1e6)

    
    def create_bins(self, earliest_time, latest_time):
        """
        Creates three equidistant time bins between earliest_time and latest_time.

        Parameters:
        - earliest_time (float): Earliest time in seconds.
        - latest_time (float): Latest time in seconds.

        Returns:
        - bin_edges (list): Edges of the bins.
        - bin_labels (list): Labels for the bins.
        """
        # Calculate the size of each bin

        #dynamically bin into hours
        factor = 3 #int(np.round((latest_time-earliest_time)/3600))
        
        delta = (latest_time - earliest_time) / factor #3 = morning afternoon evening 
        # Generate bin edges
        bin_edges = [earliest_time + i * delta for i in range(factor + 1)]
        # Create bin labels
        bin_labels = [f'bin_{i}' for i in range(factor)]
        return bin_edges, bin_labels

    def prepare_kde_models(self):
        """
        Prepares KDE models for interarrival times by processing training data.

        Notes
        -----
        - Groups timestamps by weekday and calculates interarrival time statistics.
        - Clusters weekdays based on interarrival and arrival statistics using hierarchical clustering.
        - Handles missing weekdays by assigning them to a new cluster with default values.
        - Maps timestamps to weekday clusters and organizes interarrival times into bins.
        - Computes first differences of interarrival times and estimates KDE standard deviations for each bin.
        - Stores processed data and KDE parameters in `self.diffed_data_dict` and `self.diffed_kernel_std_dict`.

        Outputs
        -------
        - `self.date_to_cluster` : Mapping of weekdays to cluster labels.
        - `self.diffed_data_dict` : Dictionary of processed interarrival data by cluster and bin.
        - `self.diffed_kernel_std_dict` : Dictionary of KDE standard deviations for each cluster and bin.
        """
        self.diffed_data_dict = {}
        self.diffed_kernel_std_dict = {}
        # self.global_means = {}
        
        for segment_cluster, timestamp_list in self.train_segmented.items():
            timestamps_df = pd.DataFrame({'timestamp': timestamp_list})
            timestamps_df['weekday_num'] = timestamps_df['timestamp'].apply(lambda x: x.isoweekday()) #flag weekday
            timestamps_grouped = timestamps_df.groupby('weekday_num')
            
            #construct feature dataframe to apply clustering methods 
            weekdays = []
            mean_num_arrivals_per_weekday = []
            mean_interarrival_time_per_weekday = []
            std_interarrival_time_per_weekday = []
            for weekday, group in timestamps_grouped:
                weekdays.append(weekday) #grouper variable in feature matrix
                #calculate average number of arrivals for a weekday
                times = group['timestamp']
                dates_df = pd.DataFrame({'date':times.dt.date, 'count': 1})
                num_arrivals_per_weekday = dates_df.groupby('date').agg('sum')
                mean_num_arrivals_per_weekday.append(np.mean(num_arrivals_per_weekday))

                #calculate mean and stdev of interarrival times for a weekday 
                interarrivals = []
                for date, day_times in times.groupby(times.dt.date):
                    day_diffs = day_times.diff().dt.total_seconds().dropna().values
                    interarrivals.extend(day_diffs)
                interarrivals = np.array(interarrivals)
                if len(interarrivals) > 0:
                    mean_interarrival_time_per_weekday.append(np.mean(interarrivals))
                    std_interarrival_time_per_weekday.append(np.std(interarrivals))
                else:
                    mean_interarrival_time_per_weekday.append(0)
                    std_interarrival_time_per_weekday.append(0)

            #assemble basis for feature matrix 
            statistics_only_existing_weekdays = pd.DataFrame({
                'weekday': weekdays,
                'mean_num_arrivals': mean_num_arrivals_per_weekday,
                'mean_interarrival': mean_interarrival_time_per_weekday,
                'std_interarrival': std_interarrival_time_per_weekday
            })
            statistics_only_existing_weekdays_sorted = statistics_only_existing_weekdays.sort_values('weekday').reset_index(drop=True)

            
            #construct feature matrix 
            feature_matrix = statistics_only_existing_weekdays_sorted[['mean_num_arrivals','mean_interarrival', 'std_interarrival']].values #'mean_interarrival', 'std_interarrival' // 'mean_num_arrivals

            #standardize feature_matrix
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            hierarchical_clusters = linkage(feature_matrix_scaled, method='ward')  # 'ward' minimises variance within clusters
            max_clusters = 7
            labels_hierarchical_clusters = fcluster(hierarchical_clusters, max_clusters, criterion='maxclust') - 1  # Subtract 1 for zero-based labels
            
            self.logger.info(f'weekday cluster labels:{labels_hierarchical_clusters}\n')
            statistics_only_existing_weekdays_sorted['cluster'] = labels_hierarchical_clusters
            
            #make sure that missing days are represented as such 
            all_weekdays = set(range(1, 8))
            existing_weekdays = set(statistics_only_existing_weekdays['weekday'])
            missing_weekdays = all_weekdays - existing_weekdays

            # Find the highest cluster index
            max_cluster_index = labels_hierarchical_clusters.max()
            
            # Assign missing weekdays to a new cluster
            statistics_only_missing_weekdays_with_cluster_label = pd.DataFrame({
                'weekday': list(missing_weekdays),
                'mean_num_arrivals': 0,
                'mean_interarrival': 0,
                'std_interarrival': 0,
                'cluster': max_cluster_index + 1
            })
            
            # combine existing weekdays with missing weekdays
            complete_weekdays_statistics_with_labels = pd.concat([statistics_only_existing_weekdays_sorted, statistics_only_missing_weekdays_with_cluster_label], ignore_index=True)
            complete_weekdays_statistics_with_labels_sorted = complete_weekdays_statistics_with_labels.sort_values('weekday').reset_index(drop=True)

            #update the mapping from weekdays to clusters
            self.date_to_cluster = dict(zip(complete_weekdays_statistics_with_labels_sorted['weekday'], complete_weekdays_statistics_with_labels_sorted['cluster']))
        
            # map dates to clusters in the original DataFrame
            timestamps_df['cluster'] = timestamps_df['weekday_num'].map(self.date_to_cluster)

            self.weekday_cluster = timestamps_df.groupby('weekday_num')['cluster'].agg(lambda x: x.value_counts().index[0]) 
            
            #get float equivalents of timestamps from each existing (containing data) cluster in a dict 
            weekday_cluster_dict = {}
            for weekday_cluster in timestamps_df['cluster'].unique():
                weekday_cluster_timestamps = timestamps_df[timestamps_df['cluster'] == weekday_cluster]['timestamp'].tolist()
                weekday_cluster_floats = transform_to_float(weekday_cluster_timestamps)  # return list of lists of floats
                weekday_cluster_dict[weekday_cluster] = weekday_cluster_floats
            
            #account for clusters of days with no data (dead weekend may still be present in test)
            missing_weekday_cluster_dict = {}
            for weekday_cluster in statistics_only_missing_weekdays_with_cluster_label['cluster'].unique():
                missing_weekday_cluster_dict[weekday_cluster] = []
                
            #compute first differences and KDE standard deviations for each float cluster
            diffed_weekday_data_dict = {}
            # for weekday_cluster in weekday_cluster_dict.keys():
            #     diffed_weekday_data_dict[weekday_cluster] = np.array(self.first_diff_data(weekday_cluster_dict[weekday_cluster]))
            # for weekday_cluster in missing_weekday_cluster_dict.keys():
            #     diffed_weekday_data_dict[weekday_cluster] = np.array([])

            diffed_weekday_kernel_std_dict = {}
            
            #get a dictionary that holds the actual timestamps in a list for each cluster and contains all possible clusters 
            all_weekday_cluster_dict_timestamps = {}
            for weekday_cluster in timestamps_df['cluster'].unique():
                all_weekday_cluster_dict_timestamps[weekday_cluster] = timestamps_df[timestamps_df['cluster'] == weekday_cluster]['timestamp'].tolist()
            for weekday_cluster in statistics_only_missing_weekdays_with_cluster_label['cluster'].unique():
                all_weekday_cluster_dict_timestamps[weekday_cluster] = []
                
            for weekday_cluster, cluster_timestamps in all_weekday_cluster_dict_timestamps.items():
                # print(cluster_timestamps)
                # print('\n')
                if len(cluster_timestamps) == 0:
                    continue  # Skip empty clusters

                cluster_df = pd.DataFrame({'Timestamp': cluster_timestamps})
                cluster_df['Timestamp'] = pd.to_datetime(cluster_df['Timestamp'])
                cluster_df['Date'] = cluster_df['Timestamp'].dt.date
                cluster_df['Time_in_seconds'] = (
                    cluster_df['Timestamp'].dt.hour * 3600 +
                    cluster_df['Timestamp'].dt.minute * 60 +
                    cluster_df['Timestamp'].dt.second +
                    cluster_df['Timestamp'].dt.microsecond / 1e6
                )
                # Find earliest and latest times
                earliest_time_in_seconds = cluster_df['Time_in_seconds'].min()
                latest_time_in_seconds = cluster_df['Time_in_seconds'].max()
                # print(cluster_df.head())
                # print('\n')
                # print(earliest_time_in_seconds)
                # print(latest_time_in_seconds)
                # print('\n')
                
                # Create bins
                bin_edges, bin_labels = self.create_bins(earliest_time_in_seconds, latest_time_in_seconds)
                # Assign bins
                cluster_df['Bin'] = pd.cut(
                    cluster_df['Time_in_seconds'],
                    bins=bin_edges,
                    labels=bin_labels,
                    include_lowest=True,
                    right=False
                )
                # print(cluster_df.head(20)) #should look like this for 1hr bins: 
                #                          Timestamp        Date  Time_in_seconds    Bin
                # 0         2020-04-06 09:00:00+00:00  2020-04-06        32400.000  bin_0
                # 1  2020-04-06 10:51:34.931000+00:00  2020-04-06        39094.931  bin_1
                # 2  2020-04-06 10:53:31.833000+00:00  2020-04-06        39211.833  bin_1
                # 3  2020-04-06 11:12:54.486000+00:00  2020-04-06        40374.486  bin_2
                # 4  2020-04-06 12:19:28.474000+00:00  2020-04-06        44368.474  bin_3
                
                # Group by 'Date' and 'Bin' to get counts
                counts_per_bin_per_day = cluster_df.groupby(['Date', 'Bin']).size().unstack(fill_value=0)
                # Compute global means per bin
                # global_means = counts_per_bin_per_day.mean()
                # Store global means for later use
                # for bin_label in bin_labels:
                    # key = f'weekday_cluster_{weekday_cluster}_{bin_label}'
                    # self.global_means[key] = global_means.get(bin_label, 0) #global_means is local variant only 
                # Add '_high' flags for each bin
                # for bin_label in bin_labels:
                    # counts_per_bin_per_day[bin_label + '_high'] = counts_per_bin_per_day[bin_label] > global_means[bin_label]
                # print('\n')
                # print(counts_per_bin_per_day.head(10))
                # Bin         bin_0  bin_1  bin_2  bin_0_high  bin_1_high  bin_2_high
                # Date
                # 2020-04-06      4      7      2        True        True        True
                # 2020-04-13      3      2      4        True       False        True
                
                # Process each date
                for date in counts_per_bin_per_day.index:
                    day_df = cluster_df[cluster_df['Date'] == date]
                    for idx, bin_label in enumerate(bin_labels):
                        bin_data = day_df[day_df['Bin'] == bin_label]
                        timestamps = bin_data['Timestamp'].sort_values()
                        # print(f'\ntimestamps:{timestamps}')
                        #0          2020-04-06 09:00:00+00:00
                        # 1   2020-04-06 10:51:34.931000+00:00
                        # 2   2020-04-06 10:53:31.833000+00:00
                        # 3   2020-04-06 11:12:54.486000+00:00
                        # Convert timestamps to float hours
                        time_floats = timestamps.apply(lambda x: x.hour + x.minute / 60 + x.second / 3600 + x.microsecond / (1e6 * 3600)).values #correct computation as so far
                        # print(f'time_floats:{time_floats}')
                        #[ 9.         10.85970306 10.89217583 11.215135  ]

                        if len(time_floats) > 1:
                            interarrivals = np.diff(time_floats)
                            key = f'weekday_cluster_{weekday_cluster}_{bin_label}'
                            if key in diffed_weekday_data_dict:
                                diffed_weekday_data_dict[key] = np.concatenate([diffed_weekday_data_dict[key], interarrivals])
                            else:
                                diffed_weekday_data_dict[key] = interarrivals

                                    
                #compute kde standard deviations for each key
                for key in diffed_weekday_data_dict.keys():
                    if key.startswith(f'weekday_cluster_{weekday_cluster}_'):
                        data = diffed_weekday_data_dict[key]
                        if data.size > 0:
                            base_bw = silvermans_rule(data.reshape(-1, 1))
                            
                            kernel_std = (1+self.bw_smooth_factor) * base_bw
                            
                            diffed_weekday_kernel_std_dict[key] = kernel_std
                        else:
                            diffed_weekday_kernel_std_dict[key] = None  # No data to compute KDE                    
                                    
            self.diffed_data_dict[segment_cluster] = diffed_weekday_data_dict
            self.diffed_kernel_std_dict[segment_cluster] = diffed_weekday_kernel_std_dict

    def first_diff_data(self, data):
        """
        Computes the first differences of the data sequences.

        Parameters:
        -----------
        data : list of lists
            A list where each element is a list of time floats for a cluster.

        Returns:
        --------
        diffed_data : numpy.ndarray
            Flattened array of first differences.
        """
        diffed_data = []
        for sequence in data:
            if len(sequence) > 1:
                diffs = np.diff(sequence)
                diffed_data.extend(diffs)
        return diffed_data

    def float_to_time_pandas(self, time_float):
        """
        Converts a float representing hours into a `datetime.time` object.

        Parameters
        ----------
        time_float : float
            The time in float hours (e.g., 13.5 represents 1:30 PM).

        Returns
        -------
        time_obj : datetime.time
            The corresponding `datetime.time` object.

        Raises
        ------
        ValueError
            If `time_float` is not in the range [0, 24).
        """
        try:
            if not (0 <= time_float <= 24):
                raise ValueError(f"time_float {time_float} is out of valid range [0, 24).")

            epsilon = 1e-8  # Small value to prevent floating-point errors
            hours = int(time_float - epsilon)
            remainder = time_float - hours
            minutes = int((remainder * 60) - epsilon)
            remainder = remainder * 60 - minutes
            seconds = int((remainder * 60) - epsilon)
            remainder = remainder * 60 - seconds
            microseconds = int(round(remainder * 1_000_000))

            # Ensure values are within valid ranges
            if hours > 23:
                hours = 23
                minutes = 59
                seconds = 59
                microseconds = 999_999
            if minutes > 59:
                minutes = 59
            if seconds > 59:
                seconds = 59
            if microseconds > 999_999:
                microseconds = 999_999

            return time(hour=hours, minute=minutes, second=seconds, microsecond=microseconds)
        except Exception as e:
            print(f"Error converting time_float {time_float}: {e}")
            print(f"Computed values - hours: {hours}, minutes: {minutes}, seconds: {seconds}, microseconds: {microseconds}")
            raise  # Re-raise the exception after printing
    
    def create_bins(self, earliest_time, latest_time):
        """
        Creates time bins with lengths specified as multiples of 1 hour,
        starting from earliest_time until latest_time is overstepped.
        The final bin that oversteps latest_time is truncated to latest_time.

        Parameters:
        - earliest_time (float): Earliest time in seconds.
        - latest_time (float): Latest time in seconds.

        Returns:
        - bin_edges (list): Edges of the bins.
        - bin_labels (list): Labels for the bins.
        """
        bin_size_seconds = self.bin_size_hours * 3600
        bin_edges = [earliest_time]
        while bin_edges[-1] + bin_size_seconds < latest_time:
            bin_edges.append(bin_edges[-1] + bin_size_seconds)
        # Add the last bin edge, which is the latest_time
        bin_edges.append(latest_time)
        bin_labels = [f'bin_{i}' for i in range(len(bin_edges) - 1)]
        return bin_edges, bin_labels

    def sample_interarrivals(self, n_interarrivals, cluster_segment, cluster_weekday, bin_name):
        """
        Samples interarrival times for a specified bin using KDE models.

        Parameters
        ----------
        n_interarrivals : int
            Number of interarrival times to sample.
        cluster_segment : str
            Cluster segment identifier.
        cluster_weekday : int
            Weekday cluster identifier (e.g., 1 for Monday).
        bin_name : str
            Bin label combining cluster and time bin information.

        Returns
        -------
        sampled_interarrivals : numpy.ndarray
            Array of sampled interarrival times. Returns an empty array if no data is available.

        Notes
        -----
        - Samples times from the KDE model, adding Gaussian noise based on the bin's standard deviation.
        - Retains only positive interarrival times.
        - Returns an empty array if data or kernel standard deviation for the bin is unavailable.
        """
        #e.g. bin_name = weekday_cluster_2_bin_0 = 'cluster_weekday' + 'bin_label'
        data = self.diffed_data_dict[cluster_segment].get(bin_name, np.array([]))
        
        # data = self.diffed_data_dict[cluster_segment][bin_name]
        if len(data) == 0:
            return np.array([]) #return empty array if no data available on this day
        
        base_samples = np.random.choice(data, size=n_interarrivals, replace=True)
        kernel_std = self.diffed_kernel_std_dict[cluster_segment].get(bin_name)
        #kernel_std = self.diffed_kernel_std_dict[cluster_segment][bin_name]
        
        if kernel_std is None:
            return np.array([])
        sampled_interarrivals = base_samples + np.random.randn(n_interarrivals) * kernel_std
        sampled_interarrivals = sampled_interarrivals[sampled_interarrivals > 0]
        return sampled_interarrivals

    def sample_kde(self, n):
        """
        Simulates data sequences for `n` days using KDE models of interarrival times.

        Parameters
        ----------
        n : int
            Number of days to simulate.

        Returns
        -------
        all_sequences : list of pandas.Timestamp
            Combined simulated timestamps for all days.
            
        sequence_lengths : list of int
            Lengths of sequences for each day.

        Notes
        -----
        - Starts simulation from the last training timestamp.
        - Predicts clusters for each day, updating missing dates with closest available data.
        - Divides each day into bins and samples interarrival times per bin.
        - Ensures timestamps fit within daily bounds and converts them to UTC format.
        - Combines results across all days for the final output.
        """
        #simulation shall begin at the end of training data 
        self.final_timestamp_train = max(self.ref_data)
        
        start_date = self.final_timestamp_train.date()
        sequences_per_day = {}
        sequence_lengths_per_day = {}

        for day_offset in range(n):
            current_date = start_date + timedelta(days=day_offset)
            corresponding_weekday_cluster = self.date_to_cluster[current_date.isoweekday()] #.isoweekday() gets integer value (range 1-7) for a date object

            # Check if current_date is not in the index of self.test_segment_estim
            #if this is the case, add this date and log it for a quick fix 
            if current_date not in self.test_segment_estim.index:
                self.logger.info(f"Current date {current_date} not found in test_segment_estim.index.")
                self.logger.info(f"Start date: {start_date}")
                self.logger.info(f"self.test_segment_estim:\n{self.test_segment_estim}")

                #find the closest date in the DataFrame
                closest_date = min(
                    self.test_segment_estim.index,
                    key=lambda date: abs(date - current_date)
                )
                #get the 'predicted_cluster' value from the closest date
                predicted_cluster = self.test_segment_estim.loc[closest_date, 'predicted_cluster']
                # Insert current_date into the DataFrame with the predicted_cluster value and sort df afterwards
                self.test_segment_estim.loc[current_date] = {'predicted_cluster': predicted_cluster}

                self.test_segment_estim = self.test_segment_estim.sort_index()
                self.logger.info(f"updated self.test_segment_estim:\n{self.test_segment_estim}")
                
            current_date_predicted_cluster = int(self.test_segment_estim.loc[current_date, 'predicted_cluster'])
            
            lower_time, upper_time = self.lower_bound, self.upper_bound
            if day_offset == 0:
                # Starting day
                lower_time = self.final_timestamp_train.hour + self.final_timestamp_train.minute / 60 + self.final_timestamp_train.second / 3600
            
            # print(f'lower_time: {lower_time}')
            # print(f'upper_time: {upper_time}')
            
            # Create bins for this cluster
            bin_edges, bin_labels = self.create_bins(lower_time * 3600, upper_time * 3600)

            float_bin_edges = [edge / 3600 for edge in bin_edges]
            
            # print('\n')
            # print(f'bin_edges: {bin_edges}')
            # print(f'bin_labels: {bin_labels}')
            # print('\n')
            # print(f'float_bin_edges: {float_bin_edges}')
            
            current_time = lower_time
            final_sequence = []
            # Simulate arrivals for each bin
            for i, bin_label in enumerate(bin_labels):
                current_bin_edge = float_bin_edges[i + 1]
                current_bin_name = f'weekday_cluster_{corresponding_weekday_cluster}_{bin_label}'
                interarrival_samples = self.sample_interarrivals(
                                                                    n_interarrivals = 1000, 
                                                                    cluster_segment = current_date_predicted_cluster, 
                                                                    cluster_weekday = corresponding_weekday_cluster,
                                                                    bin_name = current_bin_name
                                                                )
                #gnerate raw sequence of arrival times
                raw_bin_seq = current_time + np.cumsum(interarrival_samples)
                #find index where the sequence surpasses the current bin edge
                surpass_indices = np.where(raw_bin_seq > current_bin_edge)[0]
                if len(surpass_indices) == 0:
                    last_valid_index = len(raw_bin_seq) - 1
                else:
                    last_valid_index = surpass_indices[0] - 1
                # Slice the sequence up to the bin edge
                current_bin_seq = raw_bin_seq[:last_valid_index + 1]
                # Append to the final sequence
                final_sequence.extend(current_bin_seq.tolist())
                # Update current_time for the next iteration
                if len(current_bin_seq) > 0:
                    current_time = current_bin_seq[-1]
                else:
                    current_time = current_bin_edge  # Move to the next bin edge if no arrivals

            #after the loop, ensure the final times do not exceed upper_time
            final_sequence = [t for t in final_sequence if t <= upper_time]
            
            day_sequences = []
            for time_float in final_sequence:
                hours = int(time_float)
                minutes = int((time_float - hours) * 60)
                seconds = int(((time_float - hours) * 60 - minutes) * 60)
                microseconds = int(((((time_float - hours) * 60 - minutes) * 60 - seconds) * 1_000_000))
                timestamp = pd.Timestamp(datetime.combine(current_date, time(hour=hours, minute=minutes, second=seconds, microsecond=microseconds)), tz='UTC')
                day_sequences.append(timestamp)
            sequences_per_day[current_date] = day_sequences
            sequence_lengths_per_day[current_date] = len(day_sequences)

        #combine sequences from all days
        all_sequences = []
        for day_seq in sequences_per_day.values():
            all_sequences.extend(day_seq)
        sequence_lengths = list(sequence_lengths_per_day.values())
        return all_sequences, sequence_lengths