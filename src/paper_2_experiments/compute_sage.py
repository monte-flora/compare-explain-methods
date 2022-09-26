#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append('/work/mflora/ROAD_SURFACE')

import sage 
import pandas as pd
from joblib import load
import numpy as np
import xarray as xr
import pickle

from skexplain.common.importance_utils import to_skexplain_importance
from wofs_ml_severe.wofs_ml_severe.common.multiprocessing_utils import run_parallel, to_iterator

from probsr_config import TARGET_COLUMN, PREDICTOR_COLUMNS, FIGURE_MAPPINGS
from calibration_classifier import CalibratedClassifier


# In[2]:

BASE_PATH = '/work/mflora/explainability_work/'
DATA_BASE_PATH = os.path.join(BASE_PATH, 'datasets')
MODEL_BASE_PATH = os.path.join(BASE_PATH, 'models')

# In[10]:


def compute_sage(model, X, y, background):
    """Compute SAGE"""
    # Set up an imputer to handle missing features
    random_state = np.random.RandomState(42)
    random_inds = np.random.choice(len(background), size=512, replace=False)
    try:
        X_rand = background.values[random_inds,:]
    except:
        X_rand = background[random_inds,:]
    
    # Set up the imputer. 
    imputer = sage.MarginalImputer(model.predict_proba, X_rand)

    # Set up an estimator. 
    estimator = sage.PermutationEstimator(imputer, 'cross entropy')

    print(np.shape(X))
    
    sage_values = estimator(X, y)
    
    return sage_values


# In[13]:


def subsampler(X,y, size=100000):
    random_state = np.random.RandomState(42)
    inds = random_state.choice(len(X), size=size, replace=False)
    
    X_sub = X.iloc[inds]
    y_sub = y[inds]
    
    X_sub.reset_index(drop=True, inplace=True)
    
    return X_sub, y_sub


# In[11]:


time='first_hour'
targets = ['tornado', 'severe_hail', 'severe_wind', 'road_surface']
opts = ['original', 'reduced']
def worker(target, opt):        
    if target=='road_surface':
        train_df = pd.read_csv('/work/mflora/ROAD_SURFACE/probsr_training_data.csv')
        if opt == 'original':
                calibrator =  load(os.path.join(MODEL_BASE_PATH, 'JTTI_ProbSR_RandomForest_Isotonic.pkl'))
                rf_orig = load(os.path.join(MODEL_BASE_PATH,'JTTI_ProbSR_RandomForest.pkl'))
                model = CalibratedClassifier(rf_orig, calibrator)
                X = train_df[PREDICTOR_COLUMNS].astype(float)
                y = train_df[TARGET_COLUMN].astype(float).values
        else:
                # Load Model
                model_name = os.path.join(MODEL_BASE_PATH, f'RandomForest_manualfeatures_12.joblib')
                data = load(model_name)
                model = data['model']
                X = train_df[data['features']].astype(float)
                y = train_df[TARGET_COLUMN].astype(float).values

    else:
        opt_tag = '' if opt == 'original' else 'L1_based_feature_selection_with_manual'
        df = pd.read_pickle(os.path.join(DATA_BASE_PATH, f'{time}_training_matched_to_{target}_0km_dataset'))
    
        # Load Model
        model_name = os.path.join(MODEL_BASE_PATH,
                                  f'LogisticRegression_first_hour_{target}_under_standard_{opt_tag}.pkl')
        
        data = load(model_name)
        model = data['model']
        X = df[data['features']].astype(float)
        y = df[f'matched_to_{target}_0km'].astype(float).values
    
    # Calculate SAGE values
    X_sub, y_sub = subsampler(X,y)
    sage_values = compute_sage(model, X_sub.values, y_sub, X)
    
    with open(os.path.join(DATA_BASE_PATH, f'sage_results_{opt}_{target}.nc'), 'wb') as f:
        pickle.dump(sage_values, f)


# In[15]:

#worker('road_surface', 'reduced')

# In[ ]:

run_parallel(worker, args_iterator=to_iterator(targets, opts), nprocs_to_use=8)

