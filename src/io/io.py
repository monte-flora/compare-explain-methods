###########################################################
# Scripts for loading an ML model and its training dataset
# Author: monte-flora 
# Email : monte.flora@noaa.gov
###########################################################

import pandas as pd
import os
from src.common.calibration_classifier import CalibratedClassifier
from joblib import load
from .probsr_config import PREDICTOR_COLUMNS, TARGET_COLUMN


def load_data_and_model(dataset, option, dataset_path, model_path, return_dates=False):
    """Load a X,y of a dataset
    
    Parameters
    ------------
    dataset : 'road_surface', 'tornado', 'severe_hail', 'severe_wind'
    option : 'original', 'reduced'
    dataset_path : path-like
        Base path to the ML dataset dir
    model_path : path-like 
        Base path to the ML model dir

    Returns
    ------------
    model, X, y
    """
    # For this study, we only used the FIRST HOUR dataset from Flora et al. (2021)
    TIME = 'first_hour'
    
    if dataset == 'road_surface':
        est_name = 'Random Forest'
        
        train_df = pd.read_csv(os.path.join(dataset_path, 'probsr_training_data.csv'))
        if option == 'original':
            calibrator =  load(os.path.join(model_path, 'JTTI_ProbSR_RandomForest_Isotonic.pkl'))
            rf_orig = load(os.path.join(model_path,'JTTI_ProbSR_RandomForest.pkl'))
            model = CalibratedClassifier(rf_orig, calibrator)
            X = train_df[PREDICTOR_COLUMNS].astype(float)
            y = train_df[TARGET_COLUMN].astype(float).values
        else:
            model_name = os.path.join(model_path, f'RandomForest_manualfeatures_12.joblib')
            data = load(model_name)
            model = data['model']
            X = train_df[data['features']].astype(float)
            y = train_df[TARGET_COLUMN].astype(float).values
    else:
        est_name = 'LogisticRegression'
        opt_tag = '' if option == 'original' else 'L1_based_feature_selection_with_manual'
     
        #df = pd.read_pickle(os.path.join(dataset_path, f'{TIME}_training_matched_to_{dataset}_0km_dataset'))
    
        df = pd.read_feather(os.path.join(dataset_path, 
                             f'original_{TIME}_training_matched_to_{dataset}_0km_data.feather'))
    
        # Load Model
        model_path = os.path.join(model_path,
                                  f'LogisticRegression_first_hour_{dataset}_under_standard_{opt_tag}.pkl')
        
        data = load(model_path)
        model = data['model']
        X = df[data['features']].astype(float)
        y = df[f'matched_to_{dataset}_0km'].astype(float).values
        dates = df['Run Date'].values
        fti = df['FCST_TIME_IDX'].values
     
    if return_dates:
        return (est_name, model), X, y, dates, fti
        
    else:    
        return (est_name, model), X, y   