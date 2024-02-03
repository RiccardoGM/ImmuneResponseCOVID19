# Import statements
import pandas as pd
import numpy as np

# Sklearn
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Add path to custom modules
import sys
import os
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
sys.path.append('/Users/riccardo/Documents/OneDrive - SISSA/TriesteProject/Covid19/')

# Import custom modules
from Modules import CustomClasses as cc, Parameters


# ---- # ---- # ---- # Functions # ---- # ---- # ---- #

def preprocessing(Data, target_train, Data_test=pd.DataFrame(), target_test=None, standardization='dafault', 
                  imputation='knn', std_parameters_dict={}, fix_outliers=False, do_imputation=True):
    
    '''
       This function pre-processes data for training and testing.
       Input: 
             1) Data: pandas DataFrame with all covariates ready for preprocessing.
             2) target_train: string corresponding to target column in Data used for training.
             3) target_test: string corresponding to target column in Data used for testing.
             4) Data_test: pandas DataFrame with all covariates ready for preprocessing.
             5) standardization: string, 'default', 'PowerTransformer', 'QuantileTransformer'
             6) imputation: string, 'knn' or 'mean'
             7) nan_masking: False or float indicating the fraction of nans allowed in each row.
             8) random_state: seed for random state.
             9) std_parameters: dictionary of std parameters to use for standardization.
             10) fix_outliers: whether to fix outliers or not in the PowerTransformer.
    '''
    
    if target_test==None:
        target_test = target_train
    
    ## Create local copy
    Data_local = Data.copy()
    
    ## Variables
    Targets = [target_train, target_test]
    Features = [feature for feature in Data_local.columns if feature not in Targets and feature != 'ID']    
    Features_cat = [feature for feature in Features if (np.isin(Data_local[feature].dropna().unique(), [0, 1]).all())]
    Features_noncat = [feature for feature in Features if feature not in Features_cat]
    n_noncat_features = len(Features_noncat)
    
    ## Datasets from input data
    Data_X = Data_local[Features].copy().astype(float)
    Data_Y = Data_local[target_train].copy().astype(float)
    Data_ID = Data_local[['ID']].copy()

    ## Test set
    if not Data_test.empty:
        Data_X_test = Data_test[Features].copy().astype(float)
        Data_ID_test = Data_test[['ID']].copy()
        intersection = set.intersection(set(Data_test.columns), set(Targets))
        flag_test_target = False
        if len(intersection)>0:
            flag_test_target = True
            Data_Y_test = Data_test[target_test].copy().astype(float)
    
    ## Perform imputation
    if do_imputation:
        ## Standardization (before imputation)
        if n_noncat_features>0:
            X = Data_X.loc[:, Features_noncat].values.copy()
            X_mean = np.nanmean(X, axis=0)
            X_std = np.nanstd(X, axis=0)
            X = (X - X_mean) / X_std
            Data_X.loc[:, Features_noncat] = X
            # Test set
            if not Data_test.empty:
                X = Data_X_test.loc[:, Features_noncat].values.copy()
                X = (X - X_mean) / X_std
                Data_X_test.loc[:, Features_noncat] = X
                
        ## Imputation
        if imputation=='knn':
            Imputer_knn = KNNImputer(n_neighbors=10, weights='uniform')
            Data_X = pd.DataFrame(Imputer_knn.fit_transform(Data_X), columns=Features)
            # Test set
            if not Data_test.empty:
                Data_X_test = pd.DataFrame(Imputer_knn.transform(Data_X_test), columns=Features)
        elif imputation=='mean':
            Imputer_mean = SimpleImputer(strategy='mean')
            Data_X = pd.DataFrame(Imputer_mean.fit_transform(Data_X), columns=Features)
            # Test set
            if not Data_test.empty:
                Data_X_test = pd.DataFrame(Imputer_mean.transform(Data_X_test), columns=Features)
        else:
            raise ValueError('Imputation not understood. Allowed values: \'knn\' (default), \'mean\'.')

        ## Reverse standardization
        if n_noncat_features>0:
            X = Data_X.loc[:, Features_noncat].values.copy()
            X = X * X_std + X_mean
            Data_X.loc[:, Features_noncat] = X
            # Test set
            if not Data_test.empty:
                X = Data_X_test.loc[:, Features_noncat].values.copy()
                X = X * X_std + X_mean
                Data_X_test.loc[:, Features_noncat] = X

    ## Standardization
    ss_0 = StandardScaler()
    if standardization=='StandardScaler':
        standardizer = StandardScaler()
    elif standardization=='PowerTransformer':
        standardizer = PowerTransformer()
    else:
        raise ValueError('Standardization not understood. Allowed values: \'StandardScaler\', \'PowerTransformer\'.')
    std_parameters = pd.Series(index=Features_noncat, dtype=float)
    if n_noncat_features>0:
        X = Data_X.loc[:, Features_noncat].values.copy()
        # Standardize
        X = ss_0.fit_transform(X) # Apply StandardScaler first
        standardizer.fit(X) 
        if not pd.DataFrame(std_parameters_dict).empty:
            if 'PowerTransformer' in std_parameters_dict.keys():
                std_parameters_values = std_parameters_dict['PowerTransformer']
                standardizer.lambdas_ = std_parameters_values
        X = standardizer.transform(X)
        
        # Fix outliers
        if fix_outliers:
            of_method = 'zscore'
            ql = -3.
            qh = 3.
            of = cc.OutliersFixer(method=of_method)
            X = of.fit_transform(X, ql, qh)
            # re-standardize
            ss = StandardScaler()
            X = ss.fit_transform(X)
        # re-assign data
        Data_X.loc[:, Features_noncat] = X
        # Test set
        if not Data_test.empty:
            X = Data_X_test.loc[:, Features_noncat].values.copy()
            # standardize
            X = ss_0.transform(X)
            X = standardizer.transform(X)
            # fix outliers
            if fix_outliers:
                of = cc.OutliersFixer(method=of_method)
                X = of.fit_transform(X, ql, qh) # refit: detect outliers within test set
                # re-standardize
                X = ss.transform(X)
            Data_X_test.loc[:, Features_noncat] = X
        
    ## Return preprocessed datasets
    returns = {}
    df_train = pd.DataFrame(Data_X, columns=Features)
    df_train[target_train] = Data_Y
    df_train['ID'] = Data_ID
    returns['Training set'] = {'X': Data_X,
                               'Y': Data_Y,
                               'ID': Data_ID, 
                               'DataFrame': df_train}
        
    if standardization=='PowerTransformer':
        returns['Training set']['Std_Parameters'] = np.array([np.nan] * len(Features_noncat)) #standardizer.lambdas_
    else:
        returns['Training set']['Std_Parameters'] = np.array([np.nan] * len(Features_noncat))
        
    # Test set
    if not Data_test.empty:
        df_test = pd.DataFrame(Data_X_test, columns=Features)
        df_test[target_test] = np.nan
        df_test['ID'] = Data_ID_test
        returns['Test set'] = {'X': Data_X_test,
                               'ID': Data_ID_test, 
                               'DataFrame': df_test}
        if flag_test_target:
            returns['Test set']['Y'] = Data_Y_test
            df_test[target_test] = Data_Y_test
        
    return returns


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def data_dict(Data, target_train, Data_test=pd.DataFrame(), target_test=None, random_state=None):
    
    ''' This function returns a data dictionary as in preprocessing, without performing the preprocessing.'''
    
    if target_test==None:
        target_test = target_train
    
    ## Create local copy
    Data_local = Data.copy()
    
    ## Variables
    Targets = [target_train, target_test]
    Features = [feature for feature in Data_local.columns if feature not in Targets and feature != 'ID']    
    Features_cat = [feature for feature in Features if (np.isin(Data_local[feature].dropna().unique(), [0, 1]).all())]
    Features_noncat = [feature for feature in Features if feature not in Features_cat]
    
    ## Datasets from input data
    Data_X = Data_local[Features].copy().astype(float)
    Data_Y = Data_local[target_train].copy().astype(float)
    Data_ID = Data_local[['ID']].copy()

    ## Test set
    if not Data_test.empty:
        Data_X_test = Data_test[Features].copy().astype(float)
        Data_ID_test = Data_test[['ID']].copy()
        intersection = set.intersection(set(Data_test.columns), set(Targets))
        flag_test_target = False
        if len(intersection)>0:
            flag_test_target = True
            Data_Y_test = Data_test[target_test].copy().astype(float)
        
    ## Return preprocessed datasets
    returns= {'Training set': {'X': Data_X,
                               'Y': Data_Y,
                               'ID': Data_ID, 
                               'Std_Parameters': np.array([np.nan] * len(Features_noncat))
                               }}
        
    # Test set
    if not Data_test.empty:
        returns['Test set'] = {'X': Data_X_test,
                               'ID': Data_ID_test}
        if flag_test_target:
            returns['Test set']['Y'] = Data_Y_test
            
    return returns


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def models_prediction(Data, test_size, models_dict, target_train, target_test=None, min_NPV=0.97, min_NPV_Models=False, 
                      random_state=None, standardization='default', imputation='knn', do_nan_masking_univ=True, 
                      do_nan_masking=False, nan_masking=None, do_nan_masking_groupwise=False, groups=None, hyperp_dict={},
                      do_preprocessing=True, fix_outliers=False, do_imputation=True, pca_var_threshold=0.05):

    ''' Return predictions for a single train-test split, for each model in test_size '''
    
    if target_test==None:
        target_test = target_train
    
    Results = {}
    
    ## Logistic regression models
    if 'LR' in models_dict.keys():
        Results['LR'] = {}
        
        for set_name in models_dict['LR'].keys():
            columns = models_dict['LR'][set_name]            
            
            ## Look for hyperparameters
            C_reg = None
            std_parameters_dict = {}
            if not pd.DataFrame(hyperp_dict).empty:
                if 'LR' in hyperp_dict.keys():
                    if set_name in hyperp_dict['LR'].keys():
                        if 'C' in hyperp_dict['LR'][set_name].keys():
                            C_reg = hyperp_dict['LR'][set_name]['C']
                        if 'std_parameters' in hyperp_dict['LR'][set_name].keys():
                            if standardization=='PowerTransformer':
                                std_parameters_dict = {'PowerTransformer': hyperp_dict['LR'][set_name]['std_parameters']}
            
            
            ## Nan masking
            if len(columns)>1:
                if do_nan_masking:
                    nan_masking_val = nan_masking
                    do_nan_masking_groupwise_val = do_nan_masking_groupwise
                else:
                    nan_masking_val = 0.99
            elif len(columns)==1:
                do_nan_masking_groupwise_val = False
                if do_nan_masking_univ:
                    nan_masking_val = 0.0 
                else:
                    nan_masking_val = 1.
            else:
                raise ValueError('N. of columns need to be larger than zero.')
            Data_masked = nan_masking_fc(Data, 
                                         columns,
                                         nan_masking_val,
                                         do_nan_masking_groupwise_val, 
                                         groups)
            
            ## Train-test split            
            Data_train, Data_test = split_fc(Data_masked, 
                                             columns, 
                                             target_train, 
                                             target_test,
                                             test_size)


            ## Preprocessing
            #standardization: 'default' or 'PowerTransformer'
            Data_train = Data_train[columns+[target_train, 'ID']]
            Data_test = Data_test[columns+[target_test, 'ID']]
            if do_preprocessing:
                Preprocessed_data_dict = preprocessing(Data_train,
                                                       target_train,
                                                       target_test=target_test,
                                                       Data_test=Data_test,
                                                       standardization=standardization, 
                                                       imputation=imputation, # 'default' or 'PowerTransformer'
                                                       std_parameters_dict=std_parameters_dict,
                                                       fix_outliers=fix_outliers, 
                                                       do_imputation=do_imputation)
            else:
                Preprocessed_data_dict = data_dict(Data_train,
                                                   target_train,
                                                   target_test=target_test,
                                                   Data_test=Data_test)
            
            
            ## Training set
            X_train = Preprocessed_data_dict['Training set']['X'].values
            y_train = Preprocessed_data_dict['Training set']['Y'].values.ravel()
            ID_train = Preprocessed_data_dict['Training set']['ID'].values.ravel()

            
            ## Test set
            X_test = Preprocessed_data_dict['Test set']['X'].values
            ID_test = Preprocessed_data_dict['Test set']['ID'].values.ravel()
            y_test = Preprocessed_data_dict['Test set']['Y'].values.ravel()
            
            
            ## Apply pca to non-categorical data
            Data_pca = Data_train[columns].values
            idx_cat = [i for i in range(Data_pca.shape[1]) if set(Data_pca[:, i]).issubset(set([0, 1, np.nan]))]
            idx_num = [i for i in range(Data_pca.shape[1]) if i not in idx_cat]
            X_train_nocat = X_train[:, idx_num]
            pca = PCA(svd_solver='full')
            pca.fit(X_train_nocat)
            mask_components = pca.explained_variance_ratio_.ravel()<pca_var_threshold
            pca.components_[mask_components] = 0
            X_train_nocat = pca.transform(X_train_nocat)
            X_train = np.hstack((X_train[:, idx_cat], X_train_nocat))
            if 'Test set' in Preprocessed_data_dict.keys():
                X_test_nocat = X_test[:, idx_num]
                X_test_nocat = pca.transform(X_test_nocat)
                X_test = np.hstack((X_test[:, idx_cat], X_test_nocat))
                      
                        
            ## Hyperparameters grid search
            LR = LogisticRegression()
            if C_reg==None:
                grid = Parameters.hyperparameters_grid_LR
                score = Parameters.score
                n_splits = Parameters.n_splits_gridsearch
                hyperparameters = best_hyppar_gridsearch(X_train, y_train, LR, grid, score, n_splits)
                C_reg = hyperparameters['C']
            else:
                hyperparameters = {'C': C_reg, 'class_weight': 'balanced', 'penalty': 'l2', 'max_iter': 1000}
            LR = LogisticRegression(**hyperparameters)
            
            
            ## Prediction
            y_train_LR = LR.fit(X_train, y_train).predict(X_train)
            y_test_LR = LR.predict(X_test)
            value_train_LR = LR.decision_function(X_train)
            value_test_LR = LR.decision_function(X_test)
            coefficients_LR = LR.coef_.reshape(-1,)
            bias_LR = LR.intercept_
            coefficients_LR_projected = list(pca.inverse_transform(coefficients_LR[len(idx_cat):]))
            for i, idx in enumerate(idx_cat):
                coefficients_LR_projected.insert(idx, coefficients_LR[0, i])
            coefficients_LR_projected = np.array(coefficients_LR_projected)

            
            ## Prediction min NPV models
            if min_NPV_Models:
                threshold = best_threshold_class0(y_pred=y_train_LR,
                                                  value_pred=value_train_LR,
                                                  y_target=y_train,
                                                  min_NPV=min_NPV,
                                                  fixed_threshold=False)
                if pd.notnull(threshold):
                    y_train_LR_0 = np.zeros_like(y_train_LR)
                    y_train_LR_0[value_train_LR>threshold] = 1
                    y_test_LR_0 = np.zeros_like(y_test_LR)
                    y_test_LR_0[value_test_LR>threshold] = 1
                else:
                    y_train_LR_0 = np.zeros_like(y_train_LR) * np.nan
                    y_test_LR_0 = np.zeros_like(y_test_LR) * np.nan
                    #print('Threshold not found - try reducing min_NPV.')
            
            
            ## Save results
            Results['LR'][set_name] = {'Train': y_train_LR.ravel(),
                                       'Train_value': value_train_LR.ravel(),
                                       'Train_Labels': y_train,
                                       'ID_train': ID_train,
                                       'Test': y_test_LR.ravel(),
                                       'Test_value': value_test_LR.ravel(),
                                       'Test_Labels': y_test,
                                       'ID_test': ID_test,
                                       'Weights': coefficients_LR_projected.ravel(),
                                       'Bias': bias_LR.ravel(), 
                                       'C': C_reg, 
                                       'Std_Parameters': Preprocessed_data_dict['Training set']['Std_Parameters']}
            
            if min_NPV_Models:
                Results['LR'][set_name+'_minNPV'] = {'Train': y_train_LR_0.ravel(),
                                                     'Train_value': value_train_LR.ravel(),
                                                     'Train_Labels': y_train,
                                                     'ID_train': ID_train,
                                                     'Test': y_test_LR_0.ravel(),
                                                     'Test_value': value_test_LR.ravel(),
                                                     'Test_Labels': y_test, 
                                                     'ID_test': ID_test}

    return Results


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def best_threshold_class0(y_pred, value_pred, y_target, min_NPV=0.97, fixed_threshold=False):

    '''
       This function returns the value threshold giving the best specificity for the given min_NPV.
       Input: 
             1) y_pred: 1D np.array (n_samples) with class predictions.
             2) value_pred: 1D np.array (n_samples) with classifier value for each prediction.
             3) y_target: 1D np.array (n_samples) with class targets.
             4) min_NPV: float in (0, 1); minimum required negative predictive value.
             5) fixed_threshold: boolean; whether to set NPV=min_NPV.
    '''
    
    start = max(value_pred)
    best_threshold = threshold = start
    stop = min(value_pred)
    
    y_pred_0 = np.zeros_like(y_pred)
    y_pred_0[value_pred<=threshold] = 0
    y_pred_0[value_pred>threshold] = 1
    
    score = recall_score(1-y_target, 1-y_pred_0)
    best_score = 0
    
    delta_th = 0.005
    #threshold = threshold - delta_th
    
    error = 1e-2
    
    while threshold > stop:
        y_pred_0[value_pred<=threshold] = 0
        y_pred_0[value_pred>threshold] = 1
        score = recall_score(1-y_target, 1-y_pred_0)
        control_score = precision_score(1-y_target, 1-y_pred_0)
        if score > best_score:
            if fixed_threshold:
                if abs(control_score - min_NPV)<error:
                    best_score = score
                    best_threshold = threshold
            else:
                if control_score > min_NPV:
                    best_score = score
                    best_threshold = threshold
        threshold = threshold - delta_th
        
    if best_score>0:
        return best_threshold
    else:
        return None


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #
    
def best_hyppar_gridsearch(X, y, model, grid, score, n_splits, random_state=None):
    
    if random_state!=None:
        shuffle=True
    else:
        shuffle=False
    
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=grid, 
                               scoring=score,
                               cv=StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle), 
                               n_jobs=1) # can set n_jobs=-1 (use all cores) with env: anaconda
    grid_search.fit(X, y)
    
    return grid_search.best_params_


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def nan_masking_fc(Data, columns, nan_masking=None, do_nan_masking_groupwise=False, groups=[], return_row_mask=False):
    
    # assumes at least one row has data
    
    Data_X = Data[columns].copy()
    
    if nan_masking!=None:
        if nan_masking>=0 and nan_masking<=1:
            if do_nan_masking_groupwise==False:
                nan_mask = pd.isnull(Data_X.values)
                ratio_nans_rows = np.sum(nan_mask, axis=1)/nan_mask.shape[1]
                mask_rows = ratio_nans_rows<=nan_masking
                Data_masked = Data.loc[mask_rows, :]
            else:
                if len(groups)>0:
                    mask_rows = np.ones(Data_X.shape[0], dtype=bool)
                    for group in groups:
                        group_columns = [col for col in columns if col in group]
                        group_mask = np.sum(pd.isnull(Data_X[group_columns]), axis=1)/Data_X[group_columns].shape[1]<=nan_masking
                        if len(group_columns)>0:
                            mask_rows = mask_rows & group_mask  
                    Data_masked = Data.loc[mask_rows, :]
                else:
                    raise ValueError('Groups not found.')
        else:
            raise ValueError('Nan masking not understood. Admitted values: False, or float in [0, 1].')
            
    if return_row_mask:
        return Data_masked, mask_rows
    else:
        return Data_masked


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #

def split_fc(Data, columns, target_train, target_test, test_size, ignore_sex=False, max_iter=50):
    
    flag = True
    counter = 0
    
    if ('ID' in Data.columns) and ('ID' not in columns):
        columns = columns + ['ID']
        
    stratification_columns = [target_train, target_test]
    if not ignore_sex:
        if 'sex' in columns:
            stratification_columns = stratification_columns + ['sex']
        
    while(flag):
        X_train, X_test, y_train, y_test = train_test_split(Data[columns].values, 
                                                            Data[[target_train, target_test]].values, 
                                                            test_size=test_size, 
                                                            stratify=Data[stratification_columns].values, 
                                                            shuffle=True)
        y_train, y_test = y_train[:, 0], y_test[:, 1]
        n_nans_col_train = np.sum(np.sum(pd.notnull(X_train), axis=0) == 0)
        n_nans_col_test = np.sum(np.sum(pd.notnull(X_test), axis=0) == 0)
        counter = counter + 1
        if n_nans_col_train==0 and n_nans_col_test==0 and counter<max_iter:
            flag = False
        else:
            raise ValueError('It was not possible to find a split with no nans in each column.')
    
    Data_train = pd.DataFrame(X_train, columns=columns)
    Data_train[target_train] = y_train
    Data_test = pd.DataFrame(X_test, columns=columns)
    Data_test[target_test] = y_test
    
    return Data_train, Data_test