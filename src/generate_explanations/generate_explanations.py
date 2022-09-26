import sys, os 
from os.path import dirname
path = dirname(dirname(os.getcwd()))
sys.path.insert(0, path)
sys.path.insert(0, '/home/monte.flora/python_packages/scikit-explain')

import skexplain 
from skexplain.common.importance_utils import to_skexplain_importance
from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
from src.io.io import load_data_and_model
from src.common.util import subsampler, normalize_importance, compute_sage

import pickle
import shap
import itertools

# Constants. 
N_JOBS = 1
SIZE = 1000

BASE_PATH = '/work/mflora/explainability_work/'
DATA_BASE_PATH = os.path.join(BASE_PATH, 'datasets')
MODEL_BASE_PATH = os.path.join(BASE_PATH, 'models')
RESULTS_PATH = os.path.join(BASE_PATH, 'results')

DATASETS = ['road_surface','tornado', 'severe_wind', 'severe_hail', 'road_surface']
OPTIONS = ['original', 'reduced']

def func(dataset, option):    
    # Load the Data and Model 
    model, X, y = load_data_and_model(dataset, option, DATA_BASE_PATH, MODEL_BASE_PATH)
    # Subsample the dataset. 
    X_sub, y_sub = subsampler(X,y,SIZE)
    features = list(X.columns)
    est_name = model[0]

    if dataset=='road_surface':
        if option == 'original':
            estimator = (model[0], model[1].base_estimator_)
        else:
            estimator = (model[0], model[1].base_estimator.named_steps['model'])
        
        explainer = skexplain.ExplainToolkit(estimator, X_sub, y_sub)
    else:
        explainer = skexplain.ExplainToolkit(model, X_sub, y_sub)
    
    if dataset == 'road_surface':
        print('Computing treeinterpreter...')
        results = explainer.local_attributions('tree_interpreter', n_jobs=N_JOBS)

        ti_rank = to_skexplain_importance(results[f'tree_interpreter_values__{est_name}'].values, 
                                     estimator_name=est_name, 
                                     feature_names=features, 
                                     method ='tree_interpreter')

        # Sum the SHAP values for each feature and then save results. 
        explainer.save(os.path.join(RESULTS_PATH, f'ti_{dataset}_{option}.nc'), results)
        explainer.save(os.path.join(RESULTS_PATH, f'ti_rank_{dataset}_{option}.nc'), ti_rank)
        
    # For the LIME, we must provide the training dataset. We also denote any categorical features. 
    if dataset == 'road_surface':
        lime_kws = {'training_data' : X.values, 'categorical_names' : ['rural', 'urban']}
    else:
        lime_kws = {'training_data' : X.values}

    results = explainer.local_attributions('lime', lime_kws=lime_kws, n_jobs=N_JOBS)

    lime_rank = to_skexplain_importance(results[f'lime_values__{est_name}'].values, 
                                     estimator_name=est_name, 
                                     feature_names=features, 
                                     method ='lime')

    # Sum the SHAP values for each feature and then save results. 
    explainer.save(os.path.join(RESULTS_PATH, f'lime_{dataset}_{option}.nc'), results)
    explainer.save(os.path.join(RESULTS_PATH, f'lime_rank_{dataset}_{option}.nc'), lime_rank)
        
for dataset, option in  itertools.product(DATASETS, OPTIONS):
    print(f'Dataset : {dataset}... Option: {option}')
    func(dataset, option)   
    
#run_parallel(func, itertools.product(DATASETS, OPTIONS), n_jobs=N_JOBS, description='Compute TI/LIME')
    