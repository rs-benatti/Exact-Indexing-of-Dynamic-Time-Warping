import Series_Q
import numpy as np

class MBRNode:
    def __init__(self, database, window_size, num_segments, min_child_size=5):
        """
        Initialize an MBR Node with the given number of segments.

        Args:
        - num_segments: The number of segments in each PAA representation within this node.
        """
        self.series_list = []
        self.database = [Series_Q.Series_Q(series, window_size, num_segments) for series in database]
        self.num_segments = num_segments
        self.window_size = window_size
        self.paa_representations = []
        self.min_child_size = 5
        for i, candidate in enumerate(self.database):
            self.series_list.append(candidate)
            self.paa_representations.append(candidate.paa_values)
        self.paa_representations = np.array(self.paa_representations)
        self.mbrs = self.compute_mbrs()

        # Check if the database size is less than the minimum child size
        if len(database) <= self.min_child_size:
            self.is_leaf = True
            self.children = []
            return

        # Divide the dataset into two parts
        mid = len(database) // 2
        left_data = database[:mid]
        right_data = database[mid:]

        # Create child nodes
        self.left_child = MBRNode(left_data, window_size, num_segments, min_child_size)
        self.right_child = MBRNode(right_data, window_size, num_segments, min_child_size)

        # Check if the child nodes meet the minimum size requirement
        if len(self.left_child.series_list) < self.min_child_size and len(self.right_child.series_list) < self.min_child_size:
            self.is_leaf = True
            self.children = []
        else:
            self.is_leaf = False
            self.children = [self.left_child, self.right_child]

        # Initialize min and max bounds for each segment
        self.min_bounds = [float('inf')] * num_segments
        self.max_bounds = [float('-inf')] * num_segments

    def compute_mbrs(self):
        """
        Compute Minimum Bounding Rectangles (MBRs) for each segment across all PAA representations.

        Returns:
        - List of MBRs, where each MBR is a tuple (min_value, max_value) for each segment.
        """
        num_segments = self.paa_representations.shape[1]
        mbrs = []

        # Compute MBRs for each segment
        for segment_idx in range(num_segments):
            segment_values = self.paa_representations[:, segment_idx]
            min_value = np.min(segment_values)
            max_value = np.max(segment_values)
            mbrs.append([min_value, max_value])
        mbrs = np.array(mbrs)
        self.mbrs = mbrs
        return mbrs

