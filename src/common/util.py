import numpy as np
import sage

# Create a subsample of the data 
def subsampler(X, y, size):
    rs = np.random.RandomState(123)
    inds = rs.choice(len(X), size=size, replace=False)
    X_sub = X.iloc[inds, :]
    y_sub = y[inds]

    X_sub.reset_index(drop=True, inplace=True)
    
    return X_sub, y_sub

def normalize_importance(scores):
    """Min-max normalization of importance scores"""
    return scores / (np.percentile(scores, 99) - np.percentile(scores, 1))

# Compute SAGE
def compute_sage(model, X, y, background):
    """Compute SAGE"""
    # Set up an imputer to handle missing features
    random_state = np.random.RandomState(42)
    random_inds = np.random.choice(len(background), size=100, replace=False)
    try:
        X_rand = background.values[random_inds,:]
    except:
        X_rand = background[random_inds,:]
    
    # Set up the imputer. 
    imputer = sage.MarginalImputer(model.predict_proba, X_rand)

    # Set up an estimator. 
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')

    sage_values = estimator(X, y)
    
    return sage_values