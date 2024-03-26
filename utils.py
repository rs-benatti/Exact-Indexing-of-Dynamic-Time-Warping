import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dtw_distance(Q, R):
        n, m = len(Q), len(R)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, :] = np.inf
        dtw_matrix[:, 0] = np.inf
        dtw_matrix[0, 0] = 0

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(Q[i-1] - R[j-1])
                last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
                dtw_matrix[i, j] = cost + last_min

        return dtw_matrix[n, m]

def MINDIST(Q, L, H):
    """
    Calculate the MINDIST between a query Q and a minimum bounding rectangle (MBR) R.

    Parameters:
    - Q: The query series (PAA representation).
    - L: The lower endpoints of the MBR.
    - H: The higher endpoints of the MBR.

    Returns:
    - The MINDIST between Q and R.
    """
    # Initialize the sum of squared distances
    sum_of_squares = 0
    
    # Iterate through each dimension
    for q, l, h in zip(Q, L, H):
        if q < l:
            distance = l - q
        elif q > h:
            distance = q - h
        else:
            distance = 0
        sum_of_squares += distance ** 2
    
    # Return the square root of the sum of squares
    return np.sqrt(sum_of_squares)

def load_ucr_dataset(foldername, train, base_directory='UCR_TS_Archive_2015'):
    filename = base_directory + '/' + foldername + '/' + foldername + '_' + 'TRAIN' if train else base_directory + '/' + foldername + '/' + foldername + '_' +'TEST'
    data = pd.read_csv(filename, header=None)
    labels = data.values[:, 0]
    series = data.drop(columns=[0]).values
    return labels, series




