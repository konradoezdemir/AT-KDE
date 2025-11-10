import pandas as pd
import numpy as np

from source.iat_approaches.PDF import PDFIATGenerator
from source.iat_approaches.prophet_ import ProphetIATGenerator
from source.iat_approaches.kde import KDEIATGenerator
from source.arrival_distribution import DurationDistribution
from source.iat_approaches.lstm import LSTM_IAT_Generator
from source.iat_approaches.chronos import ChronosIATGenerator
from source.iat_approaches.xgboost import XGBoostIATGenerator
from source.iat_approaches.npp import NPPIATGenerator

class IAT_Generator():
    """
    This class selects the according IAT generator method 
    """

    def __init__(self, method, prob_day, train_arrival_times, inter_arrival_durations, kwargs, seed) -> None:
        self.train_arrival_times = train_arrival_times
        self.inter_arrival_durations = inter_arrival_durations
        self.prob_day = prob_day
        self.seed = seed
        self._get_generator(method = method, kwargs = kwargs)

    def generate(self, start_time, end_time):
        return self.generator.generate_arrivals(start_time, end_time)

    def _get_generator(self, method, kwargs):
        if method == 'mean':
            mean = np.mean(self.inter_arrival_durations)
            arrival_distribution = DurationDistribution("fix", mean)
            self.generator = PDFIATGenerator(
                                            train_arrival_times=self.train_arrival_times, 
                                            inter_arrival_durations=self.inter_arrival_durations, 
                                            arrival_distribution=arrival_distribution,
                                            probabilistic_day=self.prob_day,
                                            kwargs=kwargs
                                            )
        elif method == 'exponential':
            # set distribution as exponential
            mean = np.mean(self.inter_arrival_durations)
            var = np.var(self.inter_arrival_durations)
            std = np.std(self.inter_arrival_durations)
            d_min = min(self.inter_arrival_durations)
            d_max = max(self.inter_arrival_durations)
            arrival_distribution = DurationDistribution("expon", mean, var, std, d_min, d_max)
            self.generator = PDFIATGenerator(
                                            train_arrival_times=self.train_arrival_times, 
                                            inter_arrival_durations=self.inter_arrival_durations, 
                                            arrival_distribution=arrival_distribution,
                                            probabilistic_day=self.prob_day,
                                            kwargs = kwargs)
        elif method == 'best_distribution':
            self.generator = PDFIATGenerator(
                                            train_arrival_times = self.train_arrival_times, 
                                            inter_arrival_durations = self.inter_arrival_durations, 
                                            arrival_distribution=None,
                                            probabilistic_day=self.prob_day,
                                            kwargs = kwargs
                                            )
        elif method == 'prophet':
            self.generator = ProphetIATGenerator(
                                                self.train_arrival_times, 
                                                )
        elif method == 'npp':
            self.generator = NPPIATGenerator(
                                                self.train_arrival_times, 
                                                probabilistic_day=self.prob_day
                                            )
        elif method == 'kde':
            self.generator = KDEIATGenerator(
                                            train_arrival_times = self.train_arrival_times, 
                                            kwargs = kwargs
                                            )
        elif method == 'lstm':
            self.generator = LSTM_IAT_Generator(self.train_arrival_times, inter_arrival_durations=self.inter_arrival_durations)
        elif method == 'chronos':
            self.generator = ChronosIATGenerator(self.train_arrival_times)
        elif method == 'xgboost':
            self.generator = XGBoostIATGenerator(self.train_arrival_times, seed=self.seed)
        else:
            raise ValueError('Nonexistent generator')