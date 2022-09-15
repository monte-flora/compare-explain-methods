import numpy as np

# Create a subsample of the data 
def subsampler(X, y, size):
    rs = np.random.RandomState(123)
    inds = rs.choice(len(X), size=size, replace=False)
    X_sub = X.iloc[inds, :]
    y_sub = y[inds]

    X_sub.reset_index(drop=True, inplace=True)
    
    return X_sub, y_sub