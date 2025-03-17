import pandas as pd
from datetime import datetime, timedelta
import pytz
from datetime import timezone
from scipy.optimize import minimize
from tqdm import tqdm
from source.arrival_distribution import get_boundaries_of_day
from utils.helper import get_arrival_likelihood_per_day, transform_to_float, transform_to_timestamp
import numpy as np 

class NPPIATGenerator():
    """
    Generates inter arrival times by fitting a Hawkes Process
    """

    def __init__(self, train_arrival_times, data_n_seqs, probabilistic_day) -> None:
        self.train = train_arrival_times
        self.n_seqs = data_n_seqs
        
        self.lower_bound, self.upper_bound = get_boundaries_of_day(self.train)
        self.probabilistic_day = probabilistic_day
        
        if probabilistic_day == "True":
            self.arrival_likelihood = get_arrival_likelihood_per_day(self.train)
        else: 
            self.arrival_likelihood = None


    def generate_arrivals(self, start_time):
        
        hawkes_model = NonHomogeneousHawkes(n_bins=10)
    
        # Fit the model to the data
        train_grouped_floats = transform_to_float(self.train)
        
        hawkes_model.fit(train_grouped_floats, progress = True)
        # Generate new synthetic data (list of realizations)
        generated_grouped_floats = hawkes_model.generate(n = self.n_seqs,T_start = self.lower_bound/3600, T_end=self.upper_bound/3600)
        
        case_arrival_times = transform_to_timestamp(
                                                        generated_times=generated_grouped_floats, 
                                                        start_timestamp=start_time,
                                                        probabilistic=self.probabilistic_day,
                                                        arrival_likelihood=self.arrival_likelihood
                                                    )
        # sample case start timestamps
        # case_arrival_times = self.get_case_arrival_times_synthetic(start_timestamp=start_time, num_sequences_to_simulate=self.n_seqs)

        return case_arrival_times
    
    def get_min_max_timestamp(self, current_timestamp):
        day = datetime.combine(current_timestamp.date(), datetime.min.time())
        timestamp_first = (day + timedelta(seconds=self.lower_bound)).replace(tzinfo=pytz.UTC)
        timestamp_last = (day + timedelta(seconds=self.upper_bound)).replace(tzinfo=pytz.UTC)

        return timestamp_first, timestamp_last
    
class NonHomogeneousHawkes:
    """
    A class to fit and simulate a non-homogeneous Hawkes process with an exponential kernel 
    and a piecewise constant baseline.

    The intensity function is defined as:
        λ(t) = μ(t) +  α*∑_{t_i < t} exp(-β (t - t_i)),
    where μ(t) is modeled as a piecewise constant function over a set of bins.

    Attributes:
        n_bins (int): Number of bins used for the piecewise constant baseline.
        mu_bins (np.array): Estimated baseline intensity for each bin.
        alpha (float): Estimated excitation magnitude.
        beta (float): Estimated decay rate for the exponential kernel.
        bin_edges (np.array): Bin edges used to approximate the baseline.
        T_starts (list of float): Original start times (first event) for each realization. Should be reported in hrs.
        T_ends (list of float): Aligned observation window lengths (last event minus T_start) for each realization. Should be reported in hrs.
    """

    def __init__(self, n_bins=10):
        """
        Initialize the NonHomogeneousHawkes model.

        Parameters:
            n_bins (int): Number of bins to approximate the piecewise constant baseline.
        """
        self.n_bins = n_bins
        self.mu_bins = None
        self.alpha = None
        self.beta = None
        self.bin_edges = None
        self.T_starts = None
        self.T_ends = None

    def preprocess_data(data):
        """
        Preprocess the input data by aligning each realization so that its first event occurs at time 0.

        Parameters:
            data (list of list of float): Each inner list represents event times for one realization.

        Returns:
            data_aligned (list of list of float): Data with each realization shifted to start at 0.
            T_starts (list of float): The original start time (first event) of each realization.
            T_ends (list of float): The aligned end time (i.e., last event time minus first event time) for each realization.
        """
        data_aligned = []
        T_starts = []
        T_ends = []
        for seq in data:
            t_start = seq[0]
            T_starts.append(t_start)
            seq_aligned = [t - t_start for t in seq]
            data_aligned.append(seq_aligned)
            T_ends.append(seq_aligned[-1])
        return data_aligned, T_starts, T_ends

    def create_bins(T_global, n_bins):
        """
        Create equally spaced bin edges for the piecewise constant baseline.

        Parameters:
            T_global (float): Global maximum observation length among the realizations.
            n_bins (int): Number of bins.

        Returns:
            bin_edges (np.array): Array of bin edges of length n_bins+1.
        """
        return np.linspace(0, T_global, n_bins + 1)

    def baseline_at_time(t, mu_bins, bin_edges):
        """
        Retrieve the baseline intensity at time t using the piecewise constant approximation.

        Parameters:
            t (float): Time at which to evaluate the baseline.
            mu_bins (np.array): Baseline intensities for each bin.
            bin_edges (np.array): The edges of the bins.

        Returns:
            baseline (float): Baseline intensity corresponding to time t.
        """
        idx = np.searchsorted(bin_edges, t, side='right') - 1
        if idx >= len(mu_bins):
            idx = len(mu_bins) - 1
        return mu_bins[idx]

    def baseline_integral_for_realization(T_end, mu_bins, bin_edges):
        """
        Compute the integral of the piecewise constant baseline over the observation window [0, T_end].

        Parameters:
            T_end (float): End time of the observation window in the aligned time domain.
            mu_bins (np.array): Baseline intensities per bin.
            bin_edges (np.array): The bin edges.

        Returns:
            integral (float): The integrated baseline value over [0, T_end].
        """
        integral = 0.0
        for i in range(len(mu_bins)):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            if T_end <= lower:
                break
            effective_upper = min(upper, T_end)
            integral += mu_bins[i] * (effective_upper - lower)
        return integral

    def neg_log_likelihood(params, data_aligned, T_ends, bin_edges):
        """
        Compute the negative log likelihood for a Hawkes process with an exponential kernel and a
        piecewise constant baseline.

        Parameters:
            params (np.array): Model parameters concatenated as 
                               [mu_bins (length n_bins), alpha, beta].
            data_aligned (list of list of float): Aligned event times for each realization.
            T_ends (list of float): Aligned observation window lengths for each realization.
            bin_edges (np.array): Bin edges for the baseline.

        Returns:
            nll (float): Negative log likelihood value.
        """
        n_bins = len(bin_edges) - 1
        mu_bins = params[:n_bins]
        alpha = params[n_bins]
        beta = params[n_bins + 1]
        nll = 0.0
        eps = 1e-8 
        for seq, T_end in zip(data_aligned, T_ends):
            integral_baseline = NonHomogeneousHawkes.baseline_integral_for_realization(T_end, mu_bins, bin_edges)
            excitation_integral = sum((alpha / beta) * (1 - np.exp(-beta * (T_end - t))) for t in seq)
            log_sum = 0.0
            for j in range(len(seq)):
                t_j = seq[j]
                mu_t = NonHomogeneousHawkes.baseline_at_time(t_j, mu_bins, bin_edges)
                excitation = sum(alpha * np.exp(-beta * (t_j - seq[k])) for k in range(j))
                intensity = mu_t + excitation
                log_sum += np.log(intensity + eps)
            ll = - (integral_baseline + excitation_integral) + log_sum
            nll -= ll
        return nll

    def fit(self, data, progress=True):
        """
        Fit the non-homogeneous Hawkes process to the provided data using maximum likelihood estimation.

        Parameters:
            data (list of list of float): Each inner list contains event times (floats) for one realization.
                                          The realizations may have varying lengths.
            progress (bool): If True, displays a progress bar (via tqdm) for each iteration of the optimizer.

        After fitting, the model stores the estimated parameters as instance attributes:
            - self.mu_bins: Baseline intensities per bin (np.array).
            - self.alpha: Excitation parameter (float).
            - self.beta: Decay rate (float).
            - self.bin_edges: Bin edges used for the baseline (np.array).
            - self.T_starts: Original start times for each realization.
            - self.T_ends: Aligned observation window lengths for each realization.
        """
        
        # print('Params pre-set.')            
        # self.mu_bins = [3.64453615, 1.86494458 ,2.85479714 ,3.25570662 ,3.15787771 ,2.32843035, 2.67322909 ,2.84184376 ,3.0072845  ,4.43404998]
        # self.alpha = 3.4736557116104643
        # self.beta = 6.344644091566653
        # _, T_starts, T_ends = NonHomogeneousHawkes.preprocess_data(data)
        # self.T_starts = T_starts
        # self.T_ends = T_ends
        # T_global = max(T_ends)
        # self.bin_edges = NonHomogeneousHawkes.create_bins(T_global, self.n_bins)
        data_aligned, T_starts, T_ends = NonHomogeneousHawkes.preprocess_data(data)
        self.T_starts = T_starts
        self.T_ends = T_ends
        T_global = max(T_ends)
        self.bin_edges = NonHomogeneousHawkes.create_bins(T_global, self.n_bins)

        total_events = sum(len(seq) for seq in data_aligned)
        total_time = sum(T_ends)
        avg_rate = total_events / total_time
        init_mu = np.full(self.n_bins, avg_rate * 0.5)
        init_alpha = 0.5
        init_beta = 1.0
        init_params = np.concatenate([init_mu, [init_alpha, init_beta]])
        bounds = [(0, None)] * (self.n_bins + 2)

        callback = None
        if progress:
            pbar = tqdm(desc="Fitting", unit="iter")
            def _callback(xk):
                pbar.update(1)
            callback = _callback

        result = minimize(NonHomogeneousHawkes.neg_log_likelihood, init_params,
                          args=(data_aligned, T_ends, self.bin_edges),
                          bounds=bounds, method='L-BFGS-B', callback=callback)
        if progress:
            pbar.close()

        if result.success:
            est_params = result.x
            self.mu_bins = est_params[:self.n_bins]
            self.alpha = est_params[self.n_bins]
            self.beta = est_params[self.n_bins + 1]
            print("Fitting succeeded:")
            print("Estimated mu_bins:", self.mu_bins)
            print("Estimated alpha:", self.alpha)
            print("Estimated beta:", self.beta)
        else:
            raise RuntimeError("Optimization failed: " + result.message)

    def simulate_single(self, T_start=0, T_end=10):
        """
        Simulate a single realization of the Hawkes process over an observation window [T_start, T_end]
        using Ogata's thinning algorithm. The simulation is performed in the aligned (starting-at-zero)
        time domain and then shifted by T_start.

        Parameters:
            T_start (float): Start time of the observation window in hours
            T_end (float): End time of the observation window in hours.

        Returns:
            events (list of float): Simulated event times within [T_start, T_end].
        """
        length = T_end - T_start
        events = []
        t = 0.0
        # Use a conservative upper bound: maximum baseline plus the excitation jump.
        lambda_bar = np.max(self.mu_bins) + self.alpha
        while t < length:
            u = np.random.uniform()
            w = -np.log(u) / lambda_bar
            t += w
            if t >= length:
                break
            mu_t = NonHomogeneousHawkes.baseline_at_time(t, self.mu_bins, self.bin_edges)
            excitation = sum(self.alpha * np.exp(-self.beta * (t - ti)) for ti in events)
            intensity = mu_t + excitation
            d = np.random.uniform()
            if d <= intensity / lambda_bar:
                events.append(t) 
        # Shift events by T_start to return events in the original time domain.
        return [T_start + ev for ev in events]

    def generate(self, n, T_start=0, T_end=None):
        """
        Generate new synthetic realizations of the Hawkes process.

        Parameters:
            n (int): Number of new realizations to generate.
            T_start (float): Start time of the observation window (in hours) for each realization.
            T_end (float, optional): End time of the observation window (in hours) for each realization.
                                     If not provided, defaults to T_start plus the maximum training window length.

        Returns:
            simulated_data (list of list of float): A list of n simulated realizations,
                                                    each a list of event times within [T_start, T_end].
        """
        if self.T_ends is None:
            raise RuntimeError("The model must be fitted before generating data.")
        if T_end is None:
            T_end = T_start + max(self.T_ends)
        simulated_data = []
        for _ in tqdm(range(n), desc="Generating NPP-Samples..."):
            sim_events = self.simulate_single(T_start=T_start, T_end=T_end)
            simulated_data.append(sim_events)
        return simulated_data