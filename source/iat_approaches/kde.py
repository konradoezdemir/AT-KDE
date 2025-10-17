import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import copy
from scipy.stats import ks_2samp, wasserstein_distance
import logging
from tqdm import tqdm 
from utils.helper import (
                            transform_to_timestamp, 
                            transform_to_float, 
                            get_arrival_likelihood_per_day, 
                            get_inter_arrival_times_from_list_of_timestamps
                        )
from kde_core.kde_simulator import DataSimulator
from source.arrival_segmentation import tune_sensitivity, extend_pattern, get_timeframe_years

class KDEIATGenerator():
    """Generate KDE arrivals"""
    def __init__(self, train_arrival_times, kwargs) -> None:
        logging.basicConfig(level=logging.INFO, format="%(filename)s:%(lineno)d - %(message)s")
        self.logger = logging.getLogger(__name__)
        
        self.train = train_arrival_times.copy() #list of pd.timestamps 
        self.arrival_likelihood = None
        # self.n_seqs = data_n_seqs #data-based parameter, will be overwritten if **kwargs specifies the amount
        self.float_format = 'False' #always for event log, will be overwritten if **kwargs specifies the amount 
        
        if ('lower' in kwargs) and ('upper' in kwargs):
            self.domain = [float(kwargs['lower']), float(kwargs['upper'])]
        else:
            self.domain = None
        if 'n_seqs' in kwargs: #deprecated
            self.n_seqs = int(kwargs['n_seqs'])
        if 'float_format' in kwargs:
            self.float_format = kwargs['float_format'] #transform to timestamps or retain float type (relevant for synthetic case)
        if 'is_event_log' in kwargs:
            self.is_event_log = kwargs['is_event_log'] #declare whether dataset is TPP or event_log, for run-time improvement
            
    def generate_arrivals(self, start_time, end_time):
        def _setup_clustered_train_dict(train):
            """
            some intermediate info:
            segment timeseries and get output_df, segment_tuned, and labels
            output_df is a df representing the predicted cluster for each date in the test set
            segments_tuned is a list of lists, where each sublist is a segment of consecutive days
            labels is a list of labels, where each label is the cluster number for the corresponding segment
            """
            if self.arrival_likelihood == True:
                self.arrival_likelihood = get_arrival_likelihood_per_day(train)

            years = get_timeframe_years(train)
            segments_tuned, status_finished, labels, bandwidth_ratio = tune_sensitivity(train)

            self.logger.info(f'status_finished: {status_finished}')
            self.logger.info(f'labels: {labels}')
            self.logger.info(f'bandwidth_ratio: {bandwidth_ratio}')
            output_df, segment_flag = extend_pattern(
                                                        train, 
                                                        start_time, 
                                                        end_time, 
                                                        segments_tuned, 
                                                        labels, 
                                                        years
                                                    )
            self.logger.info(f'output_df prior segment_flag (unchanged if False, here: {segment_flag}): {output_df}')
            self.logger.info(f'segment_flag: {segment_flag}')

            if segment_flag:
                # segment_flag == True means: no trust in very last (too-short) segment; continue the previous cluster instead
                #last segment now becomes original last and the one preceeding merged together
                new_last_segment = segments_tuned[-2]+segments_tuned[-1] 
                segments_tuned = segments_tuned[:-2]
                segments_tuned.append(new_last_segment)
                
                faulty_segment_cl = labels[-1]
                replacement_segment_cl = labels[-2]
                labels = labels[:-1] #remove the faulty_segment_cluster_label for construction of clustered_train_dict
                
                #should typically only affect the inital test value since it is both shared by train and test 
                output_df['predicted_cluster'] = (
                                                    output_df['predicted_cluster']
                                                    .apply(
                                                            lambda c: replacement_segment_cl if c == faulty_segment_cl else c
                                                            )
                                                )
                self.logger.info(f'output_df after segment_flag: {output_df}')

            clustered_train_dict = {}
            for label, corresponding_timestamps in zip(labels, segments_tuned):
                # we do not want to overwrite the timestamps of one cluster if multiple segments of that cluster exist
                if label in clustered_train_dict:
                    # Extend existing list with new timestamps
                    clustered_train_dict[label].extend(corresponding_timestamps)
                else:
                    # Create new entry
                    clustered_train_dict[label] = corresponding_timestamps
            return output_df, clustered_train_dict

        def _run_bandwidth_optimisation(train):
            # We perform bandwidth optimization based on the validation set, which we take from the train data
            # clustered_train_dict: {cluster_label: [date1, date2, ...], ...}
            
            # split the training data into 80% train and 20% validation set 
            train_bw = train[:int(0.8*len(train))]
            val_bw = train[int(0.8*len(train)):]
            
            output_df, clustered_train_dict = _setup_clustered_train_dict(train_bw)
            
            train_df_clustered = (
                pd.Series(clustered_train_dict)      # index = cluster, values = list of dates
                .explode()                         # one row per (cluster, date)
                .rename_axis("cluster")
                .reset_index(name="date")
                .sort_values("date", ignore_index=True)
            )

            # adapt clustered_train_dict by removing val_data
            clustered_train_no_val_dict = (
                                            train_df_clustered.groupby("cluster")["date"]
                                                    .apply(lambda s: list(s.tolist()))   #retains train_data order
                                                    .to_dict()
                                        )
            self.logger.info(f'clustered_train_dict:{clustered_train_dict.keys()}')
            self.logger.info(f'clustered_train_no_val_dict:{clustered_train_no_val_dict.keys()}')

            output_df = output_df.rename(columns = {'cluster':'predicted_cluster'})
            output_df["date"] = pd.to_datetime(output_df["date"])
            output_df["date"] = output_df["date"].dt.date
            output_df = output_df.drop_duplicates().reset_index(drop = True)

            # self.logger.info(f'output_df:{output_df}')

            #run optimization of bandwidth parameter
            bw_smooth_factors = [199, 149, 124, 99, 74, 49, 24, 9, 4, 2, 0.5, 0.0, -0.1, -0.25, -0.5, -0.75, -0.85, -0.99]
            
            best_emd_iat = float('inf')
            best_bw_factor = None
            bw_emd_dict = {bw: 0 for bw in bw_smooth_factors}
            for factor in tqdm(bw_smooth_factors):
                ds_class = DataSimulator(
                            domain = self.domain, 
                            reference_dataset=train_df_clustered, 
                            reference_data_lengths=None,
                            train_clustered = clustered_train_no_val_dict,
                            test_cluster_estim = output_df,
                            path = False, 
                            bw_smooth_factor = factor
                        )
                self.logger.info(f'ds_class inception completed for bw_factor {factor}')
                simulated_data, _ = ds_class.sample_kde(start_time = start_time, end_time = end_time)

                # evaluate validation performance
                validation_data =  pd.to_datetime(val_bw)
                emd_iat = evaluate_validation_performance(simulated_data, validation_data)
                self.logger.info(f'EMD IAT for bw_smooth_factor {factor}: {emd_iat}')
                bw_emd_dict[factor] = emd_iat
                if emd_iat < best_emd_iat:
                    best_bw_factor = factor
                    best_emd_iat = emd_iat

            self.logger.info(f'best_emd_iat: {best_emd_iat}')
            self.logger.info(f'bw_emd_dict: {bw_emd_dict}')
            self.logger.info(f'best_bw_factor: {best_bw_factor}')
            return best_emd_iat, bw_emd_dict, best_bw_factor
        
        _, _, best_bw_factor = _run_bandwidth_optimisation(self.train.copy())
        
        output_df, clustered_train_dict = _setup_clustered_train_dict(self.train.copy())
        
        train_df_clustered = (
                pd.Series(clustered_train_dict)      # index = cluster, values = list of dates
                .explode()                         # one row per (cluster, date)
                .rename_axis("cluster")
                .reset_index(name="date")
                .sort_values("date", ignore_index=True)
            )
        # simulate arrivals with KDE    
        ds_class = DataSimulator(
                        domain = self.domain, 
                        reference_dataset=train_df_clustered, 
                        reference_data_lengths=None,
                        train_clustered = clustered_train_dict,
                        test_cluster_estim = output_df,
                        path = False, 
                        bw_smooth_factor = best_bw_factor
                    )

        simulated_data, _ = ds_class.sample_kde(start_time = start_time, end_time = end_time)
        
        return simulated_data


def evaluate_validation_performance(simulated_data, val_data):
    """
    Compute the wasserstein distance between the inter-arrival times of the validation set and the simulated data
    """

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