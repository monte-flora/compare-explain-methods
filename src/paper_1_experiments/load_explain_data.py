import sys
sys.path.append('/home/monte.flora/python_packages/scikit-explain/')
sys.path.append('/home/monte.flora/python_packages/master/ml_workflow')
sys.path.append('/work/mflora/ROAD_SURFACE')
import skexplain
from os.path import join
import pickle
import numpy as np
import pandas as pd
import joblib
import xarray as xr


from probsr_config import PREDICTOR_COLUMNS, FIGURE_MAPPINGS, COLOR_DICT

def to_xarray(shap_data, estimator_name, feature_names=None):
    dataset={}
    
    shap_values = shap_data['shap_values']
    bias = shap_data['bias']
    
    dataset[f'shap_values__{estimator_name}'] = (['n_examples', 'n_features'], shap_values)
    dataset[f'bias__{estimator_name}'] = (['n_examples'], bias.astype(np.float64))
    dataset['X'] = (['n_examples', 'n_features'], shap_data['X'])
    dataset['y'] = (['n_examples'], shap_data['targets'])
    
    ds = xr.Dataset(dataset)
    #ds.attrs['features'] = feature_names
    
    return ds 


def load_explain(hazard,  return_pd=False):

    explainer = skexplain.ExplainToolkit()
    
    if hazard != 'road_surface':
        base_path = '/work/mflora/ML_DATA/'

        # shap results
        fname = join(base_path, 'SHAP_VALUES', f'shap_values_LogisticRegression_{hazard}_first_hour.pkl')
        with open(fname, 'rb') as f:
            shap_data = pickle.load(f)


        # ale results
        ale_fname = join(base_path,'ALE_RESULTS', f'ale_results_all_models_{hazard}_first_hour.nc')
        ale_results = explainer.load(ale_fname)
        
        feature_names = ale_results.attrs['features used']
        
        shap_ds = to_xarray(shap_data, estimator_name='LogisticRegression' )
        ##explainer.X = pd.DataFrame(shap_ds['X'], columns=feature_names)

        shap_vals = pd.DataFrame(shap_ds['shap_values__LogisticRegression'].values, columns=feature_names)
        X_shap = pd.DataFrame(shap_ds['X'].values, columns=feature_names)    
        
        

        # pd results
        pd_fname = join(base_path,'PD_RESULTS', f'pd_results_all_models_{hazard}_first_hour.nc')
        pd_results = explainer.load(pd_fname)
    
        # ti results 
        other_base_path = '/work/mflora/explainability_work/results'
        lime_fname = join(other_base_path,f'lime_{hazard}_original.nc')
        lime_results = explainer.load(lime_fname)
        lime_ds = explainer.load(lime_fname)
        #print(lime_ds)
        lime_vals = pd.DataFrame(lime_ds['lime_values__LogisticRegression'].values, columns=feature_names)
        X_lime = pd.DataFrame(lime_ds['X'].values, columns=feature_names)
    
        ti_vals=None
        X_ti = None 


    else:
        # Loading the road surface ALE, SHAP, and permutation importance. 
        base_path = '/work/mflora/ROAD_SURFACE'

        # ale_results
        ale_fname = join(base_path,'ale_results', 'ale_rf_original.nc')
        ale_results = explainer.load(ale_fname)

        feature_names = ale_results.attrs['features used']
        
        # pd results
        pd_fname = join(base_path,'pd_results', 'pd_rf_original.nc')
        pd_results = explainer.load(pd_fname)
        
        # shap results
        shap_fname = join(base_path,'shap_results', 'shap_rf_original.nc')
        with open(shap_fname, 'rb') as f:
            shap_data = pickle.load(f)
        
        shap_ds = to_xarray(shap_data, estimator_name='Random Forest' )
        shap_vals = pd.DataFrame(shap_ds['shap_values__Random Forest'].values, columns=PREDICTOR_COLUMNS)
        X_shap = pd.DataFrame(shap_ds['X'].values, columns=PREDICTOR_COLUMNS)    
    
        #ti results
        other_base_path = '/work/mflora/explainability_work/results'
        ti_fname = join(other_base_path,f'ti_{hazard}_original.nc')
        ti_ds = explainer.load(ti_fname)
        ti_vals = pd.DataFrame(ti_ds['tree_interpreter_values__Random Forest'].values, columns=PREDICTOR_COLUMNS)
        X_ti = pd.DataFrame(ti_ds['X'].values, columns=PREDICTOR_COLUMNS)
    
        #lime results
        lime_fname = join(other_base_path,f'lime_{hazard}_original.nc')
        lime_ds = explainer.load(lime_fname)
        #print(lime_ds)
        lime_vals = pd.DataFrame(lime_ds['lime_values__Random Forest'].values, columns=PREDICTOR_COLUMNS)
        X_lime = pd.DataFrame(lime_ds['X'].values, columns=PREDICTOR_COLUMNS)
    

    
    if return_pd:
        return ale_results, shap_ds, shap_vals, X_shap, pd_results, lime_vals, X_lime, ti_vals, X_ti 
    else:    
        return ale_results, shap_ds, shap_vals, X_shap
                 
                 