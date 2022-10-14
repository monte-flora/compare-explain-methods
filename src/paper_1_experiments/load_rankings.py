import sys
sys.path.append('/home/monte.flora/python_packages/scikit-explain/')
sys.path.append('/home/monte.flora/python_packages/master/ml_workflow')
sys.path.append('/work/mflora/ROAD_SURFACE')
import skexplain
from os.path import join
import pickle
from glob import glob
import joblib

from probsr_config import PREDICTOR_COLUMNS, FIGURE_MAPPINGS, COLOR_DICT
from skexplain.common.importance_utils import to_skexplain_importance


# backward singlepass, forward multipass, coefs/gini, SHAP 
base_path = '/work/mflora/ML_DATA/'
other_base_path = '/work/mflora/explainability_work/results'

BASE_PATH = '/work/mflora/explainability_work/'
DATA_BASE_PATH = join(BASE_PATH, 'datasets')
MODEL_BASE_PATH = join(BASE_PATH, 'models')
RESULTS_PATH = join(BASE_PATH, 'results')


def rename(ds, direction):
    data_vars = [v for v in ds.data_vars if 'pass' in v]
    mapper = {v : f'{direction}_{v}' for v in data_vars}
    
    ds = ds.rename(mapper)
    
    return ds

def load_imp(hazard, option='original', adjust_scores=True):

    # Load the WoFS-ML-Severe Models
    if hazard != 'road_surface':
        base_path = '/work/mflora/ML_DATA/MODEL_SAVES'
        model_name = 'LogisticRegression'
        model_paths = glob(join(base_path, f'{model_name}_first_hour_{hazard}*'))
        if option == 'original':
            model_path = [m for m in model_paths if 'manual' not in m][0]
        else:
            model_path = [m for m in model_paths if 'manual' in m][0]
        model_data = joblib.load(model_path)

        model = model_data['model']
        feature_names = model_data['features']
    
    else:
        basePath = '/work/mflora/ROAD_SURFACE'
        # load the random forest
        if option == 'original':
            rf = joblib.load(join(basePath, 'JTTI_ProbSR_RandomForest.pkl'))
            feature_names = PREDICTOR_COLUMNS
            
        else:
            model_name = join(MODEL_BASE_PATH,'RandomForest_manualfeatures_12.joblib')
            data = joblib.load(model_name)
            rf = data['model'] 
            feature_names = data['features']
        
        
    
    explainer = skexplain.ExplainToolkit()

    if hazard == 'road_surface':
        name = 'Random Forest'
    else:
        name = 'LogisticRegression'
    
    base_path = '/work/mflora/ML_DATA/'
    
    # permutation results
    if hazard == 'road_surface': 
        basePath = '/work/mflora/ROAD_SURFACE'
        bsp_fname = join(basePath,'permutation_importance', f'perm_imp_{option}_backward.nc')
        fsp_fname = join(basePath,'permutation_importance', f'perm_imp_{option}_forward.nc')
    else:    
        path = join(base_path, 'permutation_importance')
        opt = '' if option == 'original' else 'L1_based'
        bsp_fname = join(path, f'perm_imp_{hazard}_first_hour_{opt}_backward.nc' )
        fsp_fname = join(path, f'perm_imp_{hazard}_first_hour_{opt}_forward.nc' ) 
        #bsp_fname = join(path, f'permutation_importance_{hazard}_first_hour_training_norm_aupdcbackward.nc' )
        #fmp_fname = join(path, f'permutation_importance_{hazard}_first_hour_training_norm_aupdcforward.nc' )   

    bsp = explainer.load(bsp_fname)
    fsp = explainer.load(fsp_fname)
    
    bsp = rename(bsp, 'backward')
    fsp = rename(fsp, 'forward')
    
    
    if adjust_scores:
        # Turn the permutation importances into proper importance scores. 
        # Backward multipass and forward singlepass: 1 + (permute score - original score)
        original_score = bsp[f'original_score__{name}'].values
        scores = 1.0 + (bsp[f'backward_multipass_scores__{name}'].values - original_score)
        bsp[f'backward_multipass_scores__{name}'] = (['n_vars_multipass', 'n_permute'], scores)
    
        original_score = fsp[f'original_score__{name}'].values
        scores = 1.0 + (fsp[f'forward_singlepass_scores__{name}'].values - original_score)
        fsp[f'forward_singlepass_scores__{name}'] = (['n_vars_singlepass', 'n_permute'], scores)
  
        # Backward singlepass and forward multipass: original_score - permuted score
        original_score = bsp[f'original_score__{name}'].values
        scores = original_score - bsp[f'backward_singlepass_scores__{name}'].values
        bsp[f'backward_singlepass_scores__{name}'] = (['n_vars_singlepass', 'n_permute'], scores)
    
        original_score = fsp[f'original_score__{name}'].values
        scores = original_score - fsp[f'forward_multipass_scores__{name}'].values
        fsp[f'forward_multipass_scores__{name}'] = (['n_vars_multipass', 'n_permute'], scores)

    
    bmp = bsp.copy()
    fmp = fsp.copy()

    # ALE     
    if hazard == 'road_surface':
        ale_var_fname = join(basePath,'ale_results', f'ale_var_rf_{option}.nc')
        ale_var = explainer.load(ale_var_fname)
        
    else:
        # ale results
        opt = '' if option == 'original' else 'L1_based_feature_selection_with_manual'
        ale_fname = join(base_path,'ALE_RESULTS', f'ale_var_results_all_models_{hazard}_first_hour{opt}.nc')
        ale_var = explainer.load(ale_fname)
    
    
    if hazard == 'road_surface':
        # load the random forest
        if option == 'original':
            gini_values = rf.feature_importances_
           
        else:
            gini_values = rf.base_estimator.named_steps['model'].feature_importances_
          
        gini_rank = to_skexplain_importance(gini_values,
                                       estimator_name='Random Forest', 
                                       feature_names=feature_names, 
                                         method = 'gini')
        
        
    else:
        coefs = model.base_estimator.named_steps['model'].coef_[0]
        coef_rank = to_skexplain_importance(coefs,
                                       estimator_name=name, 
                                       feature_names=feature_names, 
                                        method = 'coefs')

    # shap results
    if hazard == 'road_surface':
        fname = join(basePath,'shap_results', f'shap_rf_{option}.nc')
    else:
        fname = join(base_path, 'SHAP_VALUES', f'shap_values_LogisticRegression_{hazard}_first_hour{opt}.pkl')
    
    with open(fname, 'rb') as f:
        shap_data = pickle.load(f)
        shap_vals = shap_data['shap_values']
    
    shap_rank = to_skexplain_importance(shap_vals, 
                                      estimator_name=name, 
                                      feature_names=feature_names, 
                                      method ='shap_sum', )
    
    
    # sage 
    #sage_opt = 'reduced' if ('L1_based' in opt or opt=='reduced') else 'original'
    sage_fname = join(DATA_BASE_PATH,f'sage_results_{option}_{hazard}.nc')
    with open(sage_fname, 'rb') as f:
        sage_results = pickle.load(f)
    
    sage_rank = to_skexplain_importance(sage_results,
                                     estimator_name=name, 
                                     feature_names=feature_names, 
                                     method = 'sage')
    
    # tree interpreter
    if hazard == 'road_surface':
        ti_fname = join(other_base_path,f'ti_rank_{hazard}_{option}.nc')
        ti_rank = explainer.load(ti_fname)

    
    # lime
    lime_fname = join(other_base_path,f'lime_rank_{hazard}_{option}.nc')
    lime_rank = explainer.load(lime_fname)

    
    if hazard == 'road_surface':
        return ([bsp, bmp, fsp, fmp, sage_rank, gini_rank, shap_rank, ale_var, lime_rank, ti_rank],
                ['backward_singlepass', 'backward_multipass', 'forward_singlepass', 'forward_multipass', 'sage', 'gini', 
                 'shap_sum', 'ale_variance', 'lime', 'tree_interpreter'], name,  feature_names)
                
        #return ([bsp, bsp, fsp, fmp,  gini_rank, shap_rank, ale_var],
        #        ['singlepass', 'multipass', 'singlepass', 'multipass',  'gini', 
        #         'shap_sum', 'ale_variance',], name)
        
        
    else:
        return ([bsp, bmp, fsp, fmp, sage_rank, coef_rank, shap_rank, ale_var, lime_rank],
               ['backward_singlepass', 'backward_multipass', 'forward_singlepass', 'forward_multipass', 'sage', 'coefs', 
                 'shap_sum', 'ale_variance', 'lime'], name, feature_names)
    
        #return ([bsp, bsp, fsp, fmp, coef_rank, shap_rank, ale_var],
        #        ['singlepass', 'multipass', 'singlepass', 'multipass',  'coefs', 
        #         'shap_sum', 'ale_variance'], name)
    
    