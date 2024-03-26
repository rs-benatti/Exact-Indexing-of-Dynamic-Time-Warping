import numpy as np
class LowerBounds:
    """
    Controller created to define the lower bounding interactions between the query (Q) and a candidate Series (R)
    """
    def __init__(self, Q, R):
        assert Q.num_segments == R.num_segments
        assert Q.window_size == Q.window_size 
        self.window_size = Q.window_size
        self.num_segments = Q.num_segments
        self.Q = Q
        self.R = R

    def lower_bound_distance(self):
        return abs(self.Q .Q- self.R.Q)
    
    def lb_keogh(self):
        """
        Calculate the Lower Bounding Keogh distance between two time series.

        Returns:
        - LB_Keogh distance.
        """
        window_size = self.window_size
        LB_sum = 0
        # Create envelope of s2 with given window_size
        for i in range(len(self.Q.Q)):
            lower_bound = np.min(self.R.Q[max(0, i - window_size):min(len(self.R.Q), i + window_size + 1)])
            upper_bound = np.max(self.R.Q[max(0, i - window_size):min(len(self.R.Q), i + window_size + 1)])

            if self.Q.Q[i] > upper_bound:
                LB_sum += (self.Q.Q[i] - upper_bound) ** 2
            elif self.Q.Q[i] < lower_bound:
                LB_sum += (self.Q.Q[i] - lower_bound) ** 2

        return np.sqrt(LB_sum)
    
    def LB_PAA(self):
        """
        Calculate the Lower Bounding PAA (LB_PAA) distance between query series Q and candidate series c.

        Returns:
        - LB_PAA distance between the query series and the candidate series.
        """
        c = self.R.paa()
        num_segments = self.num_segments
        n = len(self.Q.Q)
        seg_len = np.ones(num_segments, dtype=int) * n // num_segments
        if n % num_segments != 0:
            seg_len = np.concatenate((seg_len, [n % num_segments]))

        U, L = self.Q.calculate_segment_bounds()
        N = num_segments

        sum = 0
        for i in range(len(U)):
            if c[i] > U[i]:
               sum += (seg_len[i]/N) * (c[i] - U[i])**2 
            elif c[i] < L[i]:
                sum +=(seg_len[i]/N) * (c[i] - L[i])**2
            else:
                0

        return np.sqrt(sum)