import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import copy
from scipy.stats import ks_2samp, wasserstein_distance

from utils.helper import transform_to_timestamp, transform_to_float, get_arrival_likelihood_per_day, get_inter_arrival_times_from_list_of_timestamps
from kde_core.kde_simulator import DataSimulator
from source.arrival_segmentation import tune_sensitivity, extend_pattern, get_timeframe_years

class KDEIATGenerator():
    """
    Generates arrivals using KDE
    """
    def __init__(self, train_arrival_times, data_n_seqs, kwargs, probabilistic_day) -> None:
        self.train = train_arrival_times
        # if probabilistic_day == "True":
        #     self.arrival_likelihood = True
        #     self.probabilistic = True
        # else:
        self.arrival_likelihood = None
        self.probabilistic = False
        self.n_seqs = data_n_seqs #data-based parameter, will be overwritten if **kwargs specifies the amount
        self.float_format = 'False' #always for event log, will be overwritten if **kwargs specifies the amount 
        
        if 'lower' in kwargs:
            self.domain = [float(kwargs['lower']), float(kwargs['upper'])]
        else:
            self.domain = None
        if 'n_seqs' in kwargs:
            self.n_seqs = int(kwargs['n_seqs'])
        if 'float_format' in kwargs:
            self.float_format = kwargs['float_format'] #transform to timestamps or retain float type (relevant for synthetic case)
        if 'is_event_log' in kwargs:
            self.is_event_log = kwargs['is_event_log'] #declare whether dataset is TPP or event_log, for run-time improvement
        
    def generate_arrivals(self, start_time):
        if self.arrival_likelihood == True:
            self.arrival_likelihood = get_arrival_likelihood_per_day(self.train)

        # segment timeseries and get output_df, segment_tuned, and labels
        # output_df is a df representing the predicted cluster for each date in the test set
        # segments_tuned is a list of lists, where each sublist is a segment of consecutive days
        # labels is a list of labels, where each label is the cluster number for the corresponding segment
        years = get_timeframe_years(self.train)
        segments_tuned, status_finished, labels, bandwidth_ratio = tune_sensitivity(self.train)
        output_df, segment_flag = extend_pattern(start_time, self.n_seqs, segments_tuned, labels, years)
        if segment_flag:
            new_last_segment = segments_tuned[-2]+segments_tuned[-1]
            segments_tuned = segments_tuned[:-2]
            segments_tuned.append(new_last_segment)
            labels = labels[:-1]
        
        # print('output_df:')
        # print(output_df)
        # print('\n-----------------------')
        # print('labels:')
        # print(labels)
        # print('\n-----------------------')
        # print('segments_tuned:')
        # print(segments_tuned)
        # print('\n-----------------------')
        
        segmented_train_dict = {}
        for label, corresponding_timestamps in zip(labels, segments_tuned):
            # Fix: we do not want to overwrite the timestamps of one cluster if multiple segments of that cluster exist
            # segmented_train_dict[label] = corresponding_timestamps

            if label in segmented_train_dict:
                # Extend existing list with new timestamps
                segmented_train_dict[label].extend(corresponding_timestamps)
            else:
                # Create new entry
                segmented_train_dict[label] = corresponding_timestamps

        #Fix: we ned to look at train data for this 
        # ratio = len(segmented_train_dict[output_df['predicted_cluster'][0]])/ self.n_seqs 

        # We perform bandwidth optimization based on the validation set, which we take from the train data
        # get cluster label of test data
        test_cluster_label = output_df['predicted_cluster'][0]
        # get corresponding training data
        training_data = segmented_train_dict[test_cluster_label]
        # split the training data into 80% train and 20% validation set 
        train_data = training_data[:int(0.8*len(training_data))]
        val_data = training_data[int(0.8*len(training_data)):]
        # get a list of all possible dates in the validation set
        all_dates = pd.date_range(start=val_data[0], end=val_data[-1])
        # specifiy output_df for validation set by creating a new df with columns 'date' and 'predicted_cluster' 
        output = []
        last_known_cluster = test_cluster_label
        for date in all_dates:
            output.append((date, last_known_cluster))
        output_df_val = pd.DataFrame(output, columns=["date", "predicted_cluster"])
        output_df_val = output_df_val.sort_values(by="date").drop_duplicates(subset=["date"]).reset_index(drop=True)
        output_df_val = output_df_val.set_index('date').reindex(all_dates, method='ffill').reset_index()
        output_df_val.columns = ["date", "predicted_cluster"]

        # adapt segmented_train_dict by removing val_data
        segmented_train_dict_val = copy.deepcopy(segmented_train_dict)
        segmented_train_dict_val[test_cluster_label] = train_data

        # define n_seqs for validation set
        val_datetimes = pd.to_datetime(val_data)
        earliest_date = val_datetimes.min().date()
        latest_date = val_datetimes.max().date()
        data_n_seqs_val = (latest_date - earliest_date).days + 1
        

        bw_smooth_factors = [199, 149, 124, 99, 74, 49, 24, 9, 4, 2, 0.5, 0.0, -0.1, -0.25, -0.5, -0.75, -0.85, -0.99]
        
        best_emd_iat = float('inf')
        best_bw_factor = None
        bw_emd_dict = {bw: 0 for bw in bw_smooth_factors}
        for factor in bw_smooth_factors:
            ds_class = DataSimulator(
                        domain = self.domain, 
                        reference_dataset=train_data, 
                        reference_data_lengths=None,
                        train_segmented = segmented_train_dict_val,
                        test_segment_estim = output_df_val,
                        path = False, 
                        bw_smooth_factor = factor
                    )
            simulated_data, _ = ds_class.sample_kde(n = data_n_seqs_val)
            print(f'simulated_data:\n{simulated_data}')

            # evaluate validation performance
            emd_iat = evaluate_validation_performance(simulated_data, val_data)
            print(f'EMD IAT for bw_smooth_factor {factor}: {emd_iat}')
            bw_emd_dict[factor] = emd_iat
            if emd_iat < best_emd_iat:
                best_bw_factor = factor
                best_emd_iat = emd_iat

        print(f'best_emd_iat: {best_emd_iat}')
        print(f'bw_emd_dict: {bw_emd_dict}')
        print(f'best_bw_factor: {best_bw_factor}')
            
        #the transform to float operation has been moved inside of the DataSimulator
        #self.domain is being parsed to evaluate within DataSimulator as well
        # simulate arrivals with KDE    
        ds_class = DataSimulator(
                        domain = self.domain, 
                        reference_dataset=self.train, 
                        reference_data_lengths=None,
                        train_segmented = segmented_train_dict,
                        test_segment_estim = output_df,
                        path = False, 
                        bw_smooth_factor = best_bw_factor
                    )
        simulated_data, simulated_data_lens = ds_class.sample_kde(n = self.n_seqs)
        
        #we will move everything into the kde class
        simulated_times = simulated_data
            
        return simulated_times


    # def get_arrival_likelihood_per_day(self, arrivals):
    #     arrivals = pd.Series(arrivals)
    #     arrivals = arrivals.dt.date.unique()

    #     probs = {
    #         'Monday': [0,0],
    #         'Tuesday': [0,0],
    #         'Wednesday': [0,0],
    #         'Thursday': [0,0],
    #         'Friday': [0,0],
    #         'Saturday': [0,0],
    #         'Sunday': [0,0],
    #     }

    #     curr_date = arrivals[0] 

    #     for i in range(len(arrivals)):    
    #         if arrivals[i] == curr_date: 
    #             day = arrivals[i].strftime('%A')
    #             probs[day][0] += 1
    #             probs[day][1] += 1
    #         else:
    #             day = curr_date.strftime('%A')
    #             while arrivals[i] > curr_date:
    #                 probs[day][1] += 1
    #                 curr_date = curr_date + timedelta(days=1)
    #                 day = curr_date.strftime('%A')
    #             probs[day][0] += 1
    #             probs[day][1] += 1
    #         curr_date = curr_date + timedelta(days=1)

    #     frequencies = probs
    #     for key, value in probs.items():
    #         frequencies[key] = value[0] / value[1]

    #     return frequencies


def evaluate_validation_performance(simulated_data, val_data):
    """
    Compute the wasserstein distance between the inter-arrival times of the validation set and the simulated data
    """
    # test_data_series = pd.Series(val_data.copy())
    # test_data_dt = pd.to_datetime(test_data_series, format="%d.%m.%Y %H:%M:%S")
    # test_data_dt_list = test_data_dt.tolist()

    # sim_data_series = pd.Series(simulated_data.copy())
    # sim_data_dt = pd.to_datetime(sim_data_series, format="%d.%m.%Y %H:%M:%S")
    # sim_data_dt_list = sim_data_dt.tolist()

    # test_data_dt_list_sorted = sorted(test_data_dt_list)
    # sim_data_dt_list_sorted = sorted(sim_data_dt_list)

    # test_interarrival_times = np.diff(test_data_dt_list_sorted)
    # test_interarrival_times = [delta.total_seconds() for delta in test_interarrival_times]
    # test_data_for_distance = test_interarrival_times
    
    # sim_interarrival_times = np.diff(sim_data_dt_list_sorted)
    # sim_interarrival_times = [delta.total_seconds() for delta in sim_interarrival_times]
    # sim_data_for_distance = sim_interarrival_times

    if len(simulated_data) == 0:
        sim_data_for_distance = []
    else:
        sim_data_for_distance = get_inter_arrival_times_from_list_of_timestamps(simulated_data)
    test_data_for_distance = get_inter_arrival_times_from_list_of_timestamps(val_data)
    
    if len(sim_data_for_distance) == 0:
        emd_iat = np.infty
    else:
        emd_iat = wasserstein_distance(test_data_for_distance, sim_data_for_distance)

    return np.sqrt(emd_iat)