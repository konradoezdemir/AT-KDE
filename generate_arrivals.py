import pandas as pd 
from datetime import datetime, timedelta
import argparse
import warnings
import os
import logging  

from utils.helper import read_json, write_json, transform_to_timestamp, KwargsAction, transform_to_float
from source.iat_approaches.IAT_Generator import IAT_Generator
# from source.arrival_distribution import get_inter_arrival_times

def parse_arguments():
    # parse arguments 
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--input_type', type=str, default='list',
                        help='If the dataset is a list or a whole event log')
    parser.add_argument('--dataset', type=str, default='SC_bs1000_min10',
                        help='Synthetic data filename')
    parser.add_argument('--method', type=str, default='all', help='Method to simulate inter-arrival times (mean, exponential, best_distribution, prophet, kde)')
    parser.add_argument('--prob_day', type=str, default='True', help='If the method should probabilistically consider non-working days')
    parser.add_argument('--run', type=int, default=1, help='current run index - necessary for robust result construction')
    parser.add_argument('--seed', type=int, default=0, help='seed for the xgboost model')
    parser.add_argument('--start_date', type=str, default="start_test", help='set the start date for simulation period (%Y-%m-%d), if empty (default), resort to last train date')
    parser.add_argument('--end_date', type=str, default="end_test", help='set the end date for simulation period (%Y-%m-%d), if empty (default), resort to last test date')
    parser.add_argument('--kwargs', nargs='*', action=KwargsAction, default={}, help='Method-specific parameters as key=value pairs.')
    args = parser.parse_args()

    return args

def read_data(path):
    data = read_json(path)
    if len(data) == 1:
        return data
    else:
        return data[0] # only return the times, remove list of sequence lenghts

def get_arrival_times(df, TS):
    arrival_times = []
    for case_id, events in df.groupby('case_id'):
        arrival_times += [events[TS].min()]
    arrival_times.sort()

    return arrival_times

def split_arrival_times(list_of_timestamps):
    """
    Perform temporal hold out split with 80% training and 20% testing.

    Parameters:
    - list_of_timestamps (list): The generated timestamps.

    Returns:
    - train (list): The timestamps of the train set.
    - test (list): The timestamps of the test set.
    """
    arrival_times = list_of_timestamps
    arrival_times.sort()

    number_times = (len(arrival_times))
    train_size = int(0.8 * number_times)

    train = arrival_times[:train_size]
    test = arrival_times[train_size:]

    return train, test


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

def store_data(case_arrival_times, start_time, end_time, train, test, args, logger):
    full_dataset = train + test 
    idx = pd.to_datetime(full_dataset, format = '%d.%m.%Y %H:%M:%S')
    start = pd.Timestamp(start_time)
    end   = pd.Timestamp(end_time)
    sim_period = idx[(idx >= start) & (idx < end)]
    
    logger.info(f"Number of simulated arrivals: {len(case_arrival_times)}")
    logger.info(f"Number of reference arrivals: {len(sim_period)}")

    if 'float_format' in args.kwargs and args.kwargs['float_format'] == 'True' and args.kwargs['tpp_data'] == 'True': #akin to only relevant for compensator calc
        n = int(args.kwargs['n_seqs'])
        fn_sim = f'{args.method}_simulated_n_{n}_from_{args.dataset}.json'
    else:
        fn_sim = f'simulated_run_{args.run}.json'
    fn_test = f'test.json'
    fn_train = f'train.json'

    if not args.kwargs: # if we run evaluation for CADD metric
        dir = 'event_log_simulations'
    else:
        dir = os.path.join('synthetic_data', 'simulations')

    if args.prob_day == "True" and args.method != 'prophet':
        method = f"{args.method}_prob"
    else:
        method = args.method

    data_dir = os.path.join(os.getcwd(), dir, method, args.dataset)
    path_to_file_sim = os.path.join(os.getcwd(), dir, method, args.dataset, fn_sim)
    path_to_file_test = os.path.join(os.getcwd(), dir, method, args.dataset, fn_test)
    path_to_file_train = os.path.join(os.getcwd(), dir, method, args.dataset, fn_train)

    if not os.path.exists(data_dir):
    # If it doesn't exist, create the directory
        os.makedirs(data_dir)
        
    if 'tpp_data' not in args.kwargs or not args.kwargs['tpp_data'] == 'True':
        write_json(path_to_file_sim, [timestamp.strftime("%d.%m.%Y %H:%M:%S") for timestamp in case_arrival_times])
        write_json(path_to_file_test, [timestamp.strftime("%d.%m.%Y %H:%M:%S") for timestamp in test]) if args.run == 1 else None
        write_json(path_to_file_train, [timestamp.strftime("%d.%m.%Y %H:%M:%S") for timestamp in train]) if args.run == 1 else None
    else:
        print('writing case_arrival_times..')
        write_json(path_to_file_sim, case_arrival_times)
        print('complete.')
        
        print('writing test..')
        write_json(path_to_file_test, test) if args.run == 1 else None
        print('complete.')
        
        print('writing train..')
        write_json(path_to_file_train, train) if args.run == 1 else None
        print('complete.')
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(filename)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    warnings.filterwarnings("ignore")
    args = parse_arguments()
    logger.info(f'Current dataset: {args.dataset}')
    method = args.method
    prob_day = args.prob_day #bool

    if args.input_type == 'event_log': # extract case arrival timestamps from event log and transform to list
        event_log_data = True
        current_dir = os.getcwd()
        data_dir = f"{os.path.join(current_dir, 'data', 'event_logs', args.dataset)}.csv"
        event_log = pd.read_csv(data_dir)
        if 'start_timestamp' in event_log.columns:
            TS = 'start_timestamp'
        else:
            TS = 'end_timestamp'
        event_log[TS] = pd.to_datetime(event_log[TS], utc=True, format='mixed')
        list_of_timestamps = get_arrival_times(event_log, TS)
        
    #temporal split needs to take place regardless of start time due to downstream tasks etc.
    train, test = split_arrival_times(list_of_timestamps) #returns sorted lists of timestamps
    # cvs_pharmacy
    # earliest_date train: 2019-03-25
    # latest_date train: 2019-05-14
    # earliest_date test: 2019-05-14
    # latest_date test: 2019-05-27

    inter_arrival_durations = get_inter_arrival_times_from_list_of_timestamps(arrival_times=train)

    if args.start_date is not None and args.end_date is not None:
        if args.start_date == 'train_start' and args.end_date == 'train_end':
            #default mode to simulate the set-out train period
            mode = 'full train'
            start_time = train[0]
            end_time = train[-1]
        elif args.start_date == 'test_start' and args.end_date == 'test_end':
            #default mode to simulate the set-out test period
            mode = 'full test'
            start_time = test[0]
            end_time = test[-1]
        else:
            mode = 'custom specified'
            try: 
                start_time = datetime.strptime(args.start_date, '%Y-%m-%d')
                end_time = datetime.strptime(args.end_date, '%Y-%m-%d')
            except Exception as e:
                logger.info(f'Simulation period could not be inferred correctly: {e}')
                
    earliest_date = start_time.date()
    latest_date = end_time.date()
    logger.info(f'Simulation of {mode} data enabled. earliest_date of sim: {earliest_date}, latest_date of sim:{latest_date}')
    # define generator
    generator = IAT_Generator(
                                method=method, 
                                prob_day=prob_day,
                                train_arrival_times=train, 
                                inter_arrival_durations=inter_arrival_durations,
                                seed = args.seed,
                                kwargs = args.kwargs
                            )

    # generate arrivals
    case_arrival_times = generator.generate(start_time=start_time, end_time=end_time)

    if 'float_format' in args.kwargs:
        if args.kwargs['float_format'] == 'True':
            train = transform_to_float(train)
            test = transform_to_float(test)
            if 'kde' not in args.method:
                case_arrival_times = transform_to_float(case_arrival_times)
            
    store_data(case_arrival_times, start_time, end_time, train, test, args, logger)