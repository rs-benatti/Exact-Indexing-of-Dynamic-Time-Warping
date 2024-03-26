import numpy as np
import matplotlib.pyplot as plt
import LowerBounds
import utils

class Series_Q:
    def __init__(self, series, window_size, num_segments):
        self.Q = series
        self.window_size = window_size
        self.num_segments = num_segments
        self.keogh_upper_bounds, self.keogh_lower_bounds = self.lb_keogh_envelope()
        self.U, self.L = self.calculate_segment_bounds()
        self.paa_values = self.paa()

    def find_best_match(self, database):
        best_so_far = np.inf
        index_of_best_match = -1
        for i, candidate in enumerate(database):
            candidate = Series_Q(candidate, self.window_size, self.num_segments)
            bound = LowerBounds.LowerBounds(self, candidate)
            LB_dist = bound.lb_keogh()
            if LB_dist < best_so_far:
                true_dist = utils.dtw_distance(self.Q, candidate.Q)
                if true_dist < best_so_far:
                    best_so_far = true_dist
                    index_of_best_match = i
        return index_of_best_match, best_so_far
    
    # First we need to get the upper and lower envelopes.
    def lb_keogh_envelope(self):
        """
        Calculate the upper and lower envelope of a time series given a window size.

        Returns:
        - upper_bound: The upper envelope.
        - lower_bound: The lower envelope.
        """
        window_size = self.window_size
        upper_bound = np.empty(len(self.Q))
        lower_bound = np.empty(len(self.Q))

        for i in range(len(self.Q)):
            lower_bound[i] = np.min(self.Q[max(0, i - window_size):min(len(self.Q), i + window_size + 1)])
            upper_bound[i] = np.max(self.Q[max(0, i - window_size):min(len(self.Q), i + window_size + 1)])

        return upper_bound, lower_bound
    
    def plot_with_envelope(self):
        """
        Plot two time series and the envelope of the second series.
        """
        window_size = self.window_size
        
        plt.figure(figsize=(14, 6))
        plt.plot(self.Q, label='Candidate Series', linestyle='--')
        plt.fill_between(range(len(self.Q)), self.keogh_lower_bounds, self.keogh_upper_bounds, color='gray', alpha=0.5,
                         label='LB Keogh Envelope')
        plt.title('Time Series with LB Keogh Envelope')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def calculate_segment_bounds(self):
        """
        Calculate upper and lower bounds for a time series divided into segments.

        Returns:
        - upper_bounds: Upper bounds for each segment.
        - lower_bounds: Lower bounds for each segment.
        """
        
        assert len(self.keogh_upper_bounds) == len(self.keogh_lower_bounds)
        n = len(self.keogh_upper_bounds)
        seg_len = n // self.num_segments  # Length of each segment
        upper_bounds = np.empty(self.num_segments)
        lower_bounds = np.empty(self.num_segments)
        # Calculate bounds for each segment
        for i in range(1, self.num_segments + 1):
            start_idx = seg_len * (i - 1) + 1
            end_idx = seg_len * i
            segment_upper_bound = self.keogh_upper_bounds[start_idx:end_idx]
            segment_lower_bound = self.keogh_lower_bounds[start_idx:end_idx]

            upper_bounds[i - 1] = np.max(segment_upper_bound)
            lower_bounds[i - 1] = np.min(segment_lower_bound)
        remainder = n % self.num_segments
        if remainder:
            i = self.num_segments + 1
            start_idx = seg_len * (i - 1) + 1
            end_idx = n
            segment_upper_bound = self.keogh_upper_bounds[start_idx:end_idx]
            segment_lower_bound = self.keogh_lower_bounds[start_idx:end_idx]

            upper_bounds = np.concatenate((upper_bounds, [np.max(segment_upper_bound)]))
            lower_bounds = np.concatenate((lower_bounds, [np.min(segment_lower_bound)]))

        return upper_bounds, lower_bounds
    
    def plot_segment_bounds(self):
        """
        Plot the time series along with upper and lower bounds for each segment and Keogh envelope.
        """
        window_size = self.window_size
        segments = self.num_segments
        upper_bounds, lower_bounds = self.U, self.L
        n = len(self.Q)
        seg_len = np.ones(segments, dtype=int) * n // segments
        if n % segments != 0:
            seg_len = np.concatenate((seg_len, [n % segments]))
        plt.figure(figsize=(10, 6))
        plt.plot(self.Q, label='Time Series', color='blue')

        # Plot segment bounds
        for i in range(segments):
            start_idx = i * seg_len[i]
            end_idx = (i + 1) * seg_len[i]
            plt.fill_between(range(start_idx, end_idx), lower_bounds[i], upper_bounds[i], color='gray', alpha=0.5)
        if n % segments != 0:
            start_idx = n - n % segments
            end_idx = n
            plt.fill_between(range(start_idx, end_idx), lower_bounds[i], upper_bounds[i], color='gray', alpha=0.5)

        # Plot Keogh envelope
        plt.fill_between(range(len(self.Q)), self.keogh_lower_bounds, self.keogh_upper_bounds, color='orange', alpha=0.3, label='Keogh Envelope')

        plt.title('Time Series with Segment Bounds and Keogh Envelope')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def paa(self):
        """
        Calculate the Piecewise Aggregate Approximation (PAA) of a time series.

        Returns:
        - PAA representation of the series.
        """
        n = len(self.Q)
        seg_len = n // self.num_segments
        remainder = n % self.num_segments

        if remainder == 0:
            paa_values = np.mean(self.Q.reshape(-1, seg_len), axis=1)
        else:
            paa_values = np.concatenate([np.mean(self.Q[:seg_len * self.num_segments].reshape(-1, seg_len), axis=1),
                                         [np.mean(self.Q[seg_len * self.num_segments:])]])

        return paa_values
    
    
    
