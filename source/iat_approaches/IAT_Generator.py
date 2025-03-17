import pandas as pd
import numpy as np

from source.iat_approaches.PDF import PDFIATGenerator
from source.iat_approaches.prophet_ import ProphetIATGenerator
from source.iat_approaches.kde import KDEIATGenerator
from source.iat_approaches.npp import NPPIATGenerator
from source.arrival_distribution import DurationDistribution

class IAT_Generator():
    """
    This class selects the according IAT generator method 
    """

    def __init__(self, method, prob_day, train_arrival_times, inter_arrival_durations, data_n_seqs, kwargs) -> None:
        self.train_arrival_times = train_arrival_times
        self.inter_arrival_durations = inter_arrival_durations
        self.data_n_seqs = data_n_seqs
        self.prob_day = prob_day

        self._get_generator(method = method, kwargs = kwargs)

    def generate(self, start_time):
        return self.generator.generate_arrivals(start_time)

    def _get_generator(self, method, kwargs):
        if method == 'mean':
            mean = np.mean(self.inter_arrival_durations)
            arrival_distribution = DurationDistribution("fix", mean)
            self.generator = PDFIATGenerator(
                                            train_arrival_times=self.train_arrival_times, 
                                            inter_arrival_durations=self.inter_arrival_durations, 
                                            arrival_distribution=arrival_distribution,
                                            data_n_seqs = self.data_n_seqs,
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
                                            data_n_seqs = self.data_n_seqs,
                                            probabilistic_day=self.prob_day,
                                            kwargs = kwargs)
        elif method == 'best_distribution':
            self.generator = PDFIATGenerator(
                                            train_arrival_times = self.train_arrival_times, 
                                            inter_arrival_durations = self.inter_arrival_durations, 
                                            arrival_distribution=None,
                                            data_n_seqs = self.data_n_seqs,
                                            probabilistic_day=self.prob_day,
                                            kwargs = kwargs
                                            )
        elif method == 'prophet':
            self.generator = ProphetIATGenerator(
                                                self.train_arrival_times, 
                                                data_n_seqs = self.data_n_seqs
                                                )
        elif method == 'npp':
            self.generator = NPPIATGenerator(
                                                self.train_arrival_times, 
                                                data_n_seqs = self.data_n_seqs,
                                                probabilistic_day=self.prob_day
                                            )
        elif method == 'kde':
            arrival_likelihood = None
            # probabilistic = False
            self.generator = KDEIATGenerator(
                                            train_arrival_times = self.train_arrival_times, 
                                            data_n_seqs = self.data_n_seqs, 
                                            probabilistic_day=self.prob_day,
                                            kwargs = kwargs
                                            )
        # elif method == 'kde_prob':
        #     arrival_likelihood = True
        #     # probabilistic = True
        #     self.generator = KDEIATGenerator(
        #                                     train_arrival_times = self.train_arrival_times, 
        #                                     arrival_likelihood = arrival_likelihood, 
        #                                     # probabilistic = probabilistic, 
        #                                     data_n_seqs = self.data_n_seqs, 
        #                                     probabilistic_day=self.prob_day,
        #                                     kwargs = kwargs
        #                                     )
        else:
            raise ValueError('Unexistent generator')