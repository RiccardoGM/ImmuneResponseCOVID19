## Import statements
import pandas as pd
import numpy as np
import datetime
from datetime import date
import re
import sys

# Sklearn
from scipy.stats import kurtosistest, pearsonr, spearmanr
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Custom classes
import sys
import os
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
sys.path.append('/Users/riccardo/Documents/OneDrive - SISSA/TriesteProject/Covid19/')
from Modules import CustomClasses as cc


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
                      do_preprocessing=True, fix_outliers=False, do_imputation=True):

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
            
            
            ## Apply pca
            if '#PCA' in set_name:
                substring = set_name.partition('#')[2]
                n_components = int(re.search(r'\d+', substring).group())
                
                # Apply PCA only to non-categorical data
                Data_pca = Data_train[columns].values
                idx_cat = [i for i in range(Data_pca.shape[1]) if set(Data_pca[:, i]).issubset(set([0, 1, np.nan]))]
                idx_num = [i for i in range(Data_pca.shape[1]) if i not in idx_cat]
                X_train_nocat = X_train[:, idx_num]
                pca = PCA(n_components=n_components, svd_solver='full')
                X_train_nocat = pca.fit_transform(X_train_nocat)
                X_train = np.hstack((X_train[:, idx_cat], X_train_nocat))
                
                if 'Test set' in Preprocessed_data_dict.keys():
                    X_test_nocat = X_test[:, idx_num]
                    X_test_nocat = pca.transform(X_test_nocat)
                    X_test = np.hstack((X_test[:, idx_cat], X_test_nocat))
                      
                        
            ## Hyperparameters grid search
            LR = LogisticRegression()
            if C_reg==None:
                grid = {'C': [1e-4, 1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1e0, 5*1e0, 1e1], 
                        'class_weight': ['balanced'],
                        'penalty': ['l2'],
                        'max_iter': [1000]}
                score = 'f1'
                n_splits = 3
                hyperparameters = best_hyppar_gridsearch(X_train, y_train, LR, grid, score, n_splits)
                C_reg = hyperparameters['C']
            else:
                hyperparameters = {'C': C_reg, 
                                   'class_weight': 'balanced',
                                   'penalty': 'l2',
                                   'max_iter': 1000}
            LR = LogisticRegression(**hyperparameters)
            
            
            ## Prediction
            y_train_LR = LR.fit(X_train, y_train).predict(X_train)
            y_test_LR = LR.predict(X_test)
            value_train_LR = LR.decision_function(X_train)
            value_test_LR = LR.decision_function(X_test)
            coefficients_LR = LR.coef_
            bias_LR = LR.intercept_
            if '#PCA' in set_name:
                coefficients_LR_projected = list(pca.inverse_transform(coefficients_LR[0, len(idx_cat):]))
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
                                       'Weights': coefficients_LR.ravel(),
                                       'Bias': bias_LR.ravel(), 
                                       'C': C_reg, 
                                       'Std_Parameters': Preprocessed_data_dict['Training set']['Std_Parameters']}
            if '#PCA' in set_name:
                Results['LR'][set_name]['Weights_projected'] = coefficients_LR_projected.ravel()
            
            if min_NPV_Models:
                Results['LR'][set_name+'_minNPV'] = {'Train': y_train_LR_0.ravel(),
                                                     'Train_value': value_train_LR.ravel(),
                                                     'Train_Labels': y_train,
                                                     'ID_train': ID_train,
                                                     'Test': y_test_LR_0.ravel(),
                                                     'Test_value': value_test_LR.ravel(),
                                                     'Test_Labels': y_test, 
                                                     'ID_test': ID_test}
                               
    if 'SVC' in models_dict.keys():
        Results['SVC'] = {}
        
        for set_name in models_dict['SVC'].keys():
            #print('SVC -', set_name)
            
            '''
            columns = models_dict['SVC'][set_name]
            
            ## Preprocessing
            #standardization: 'default' or 'PowerTransformer'
            Data_train = df_train[columns+[target_train, 'ID']]
            Data_test = df_test[columns+[target_test, 'ID']]
            
            nan_masking_val = nan_masking
            if nan_masking_univ==True:
                if len(columns)==1:
                    nan_masking_val = 0.0
                    do_nan_masking_groupwise = None
                    groups = None
                    
            Preprocessed_data_dict = preprocessing_V3(Data_train,
                                                      target_train,
                                                      target_test=target_test,
                                                      Data_test=Data_test,
                                                      standardization=standardization,
                                                      imputation=imputation, 
                                                      nan_masking=nan_masking_val, 
                                                      do_nan_masking_groupwise=do_nan_masking_groupwise, 
                                                      groups=groups) # 'default' or 'PowerTransformer' 
            '''
            
            
            columns = models_dict['SVC'][set_name]
            
            
            ## Hyperparameters
            C_reg = None
            std_parameters_dict = {}
            
            
            ## Nan masking
            if len(columns)>1:
                if do_nan_masking:
                    nan_masking_val = nan_masking
                else:
                    nan_masking_val = 0.99
            elif len(columns)==1:
                do_nan_masking_groupwise = False
                if do_nan_masking_univ:
                    nan_masking_val = 0.0 
                else:
                    nan_masking_val = 1.
            else:
                raise ValueError('N. of columns need to be larger than zero.')
            Data_masked = nan_masking_fc(Data, 
                                         columns,
                                         nan_masking_val,
                                         do_nan_masking_groupwise, 
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
                Preprocessed_data_dict = preprocessing_V4(Data_train,
                                                          target_train,
                                                          target_test=target_test,
                                                          Data_test=Data_test,
                                                          standardization=standardization, 
                                                          imputation=imputation, # 'StandardScaler' or 'PowerTransformer'
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
            
            
            ## Apply pca
            if '#PCA' in set_name:
                substring = set_name.partition('#')[2]
                n_components = int(re.search(r'\d+', substring).group())
                
                # Apply PCA only to non-categorical data
                Data_pca = Data_train[columns].values
                idx_cat = [i for i in range(Data_pca.shape[1]) if set(Data_pca[:, i]).issubset(set([0, 1, np.nan]))]
                idx_num = [i for i in range(Data_pca.shape[1]) if i not in idx_cat]
                X_train_nocat = X_train[:, idx_num]
                pca = PCA(n_components=n_components, svd_solver='full')
                X_train_nocat = pca.fit_transform(X_train_nocat)
                X_train = np.hstack((X_train[:, idx_cat], X_train_nocat))
                
                if 'Test set' in Preprocessed_data_dict.keys():
                    X_test_nocat = X_test[:, idx_num]
                    X_test_nocat = pca.transform(X_test_nocat)
                    X_test = np.hstack((X_test[:, idx_cat], X_test_nocat))
                    
                    
            ## Hyperparameters grid search
            SVClass = SVC()
            grid = {'C': [1e-4, 1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1e0, 5*1e0, 1e1], 
                    'kernel': ['rbf'],
                    'class_weight': ['balanced']}
            score = 'f1'
            n_splits = 7
            hyperparameters = best_hyppar_gridsearch(X_train, y_train, SVClass, grid, score, n_splits)
            C_reg = hyperparameters['C']
            SVClass = SVC(**hyperparameters)
            
            
            ## Prediction
            #SVClass = SVC(C=1., class_weight='balanced', gamma='auto', kernel='rbf')
            y_train_SVClass = SVClass.fit(X_train, y_train).predict(X_train)
            y_test_SVClass = SVClass.predict(X_test)
            value_train_SVClass = SVClass.decision_function(X_train)
            value_test_SVClass = SVClass.decision_function(X_test)
            #coefficients_SVClass = SVClass.coef_
            #bias_SVClass = SVClass.intercept_
            
            
            ## Prediction min NPV models
            if min_NPV_Models:
                threshold = best_threshold_class0(y_pred=y_train_SVClass,
                                                  value_pred=value_train_SVClass,
                                                  y_target=y_train,
                                                  min_NPV=min_NPV,
                                                  fixed_threshold=False)
                if pd.notnull(threshold):
                    y_train_SVClass_0 = np.zeros_like(y_train_SVClass)
                    y_train_SVClass_0[value_train_SVClass>threshold] = 1
                    y_test_SVClass_0 = np.zeros_like(y_test_SVClass)
                    y_test_SVClass_0[value_test_SVClass>threshold] = 1
                else:
                    y_train_SVClass_0 = np.zeros_like(y_train_SVClass) * np.nan
                    y_test_SVClass_0 = np.zeros_like(y_test_SVClass) * np.nan
                    #print('Threshold not found - try reducing min_NPV.')
            
            
            # Save results
            Results['SVC'][set_name] = {'Train': y_train_SVClass.ravel(),
                                        'Train_value': value_train_SVClass.ravel(),
                                        'Train_Labels': y_train,
                                        'ID_train': ID_train,
                                        'Test': y_test_SVClass.ravel(),
                                        'Test_value': value_test_SVClass.ravel(),
                                        'Test_Labels': y_test,
                                        'ID_test': ID_test,
                                        'C': C_reg,
                                        'Std_Parameters': Preprocessed_data_dict['Training set']['Std_Parameters']}
                
            if min_NPV_Models:
                Results['SVC'][set_name+'_minNPV'] = {'Train': y_train_SVClass_0.ravel(),
                                                      'Train_value': value_train_SVClass.ravel(),
                                                      'Train_Labels': y_train,
                                                      'ID_train': ID_train,
                                                      'Test': y_test_SVClass_0.ravel(),
                                                      'Test_value': value_test_SVClass.ravel(),
                                                      'Test_Labels': y_test, 
                                                      'ID_test': ID_test}
                
                
    if 'RFC' in models_dict.keys():
        Results['RFC'] = {}
        
        for set_name in models_dict['RFC'].keys():
            print('RFC -', set_name)
            columns = models_dict['RFC'][set_name]
            
            ## Preprocessing
            #standardization: 'default' or 'PowerTransformer'
            Data_train = df_train[columns+[target_train, 'ID']]
            Data_test = df_test[columns+[target_test, 'ID']]
            
            nan_masking_val = nan_masking
            if nan_masking_univ==True:
                if len(columns)==1:
                    nan_masking_val = 0.0
                    do_nan_masking_groupwise = None
                    groups = None
                    
            Preprocessed_data_dict = preprocessing_V3(Data_train,
                                                      target_train,
                                                      target_test=target_test,
                                                      Data_test=Data_test,
                                                      standardization=standardization, 
                                                      imputation=imputation, 
                                                      nan_masking=nan_masking_val,
                                                      do_nan_masking_groupwise=do_nan_masking_groupwise, 
                                                      groups=groups) # 'default' or 'PowerTransformer'             
            ## Training set
            X_train = Preprocessed_data_dict['Training set']['X'].values
            y_train = Preprocessed_data_dict['Training set']['Y'].values.ravel()
            ID_train = Preprocessed_data_dict['Training set']['ID'].values.ravel()

            ## Test set
            X_test = Preprocessed_data_dict['Test set']['X'].values
            ID_test = Preprocessed_data_dict['Test set']['ID'].values.ravel()
            y_test = Preprocessed_data_dict['Test set']['Y'].values.ravel()
            
            ## Prediction
            RFC = RandomForestClassifier(n_estimators=500, min_samples_split = 20, class_weight='balanced', random_state=random_state)
            y_train_RFC = RFC.fit(X_train, y_train).predict(X_train)
            y_test_RFC = RFC.predict(X_test)
            #coefficients_RFC = RFC.coef_
            #bias_RFC = RFC.intercept_
            
            # Save results
            Results['RFC'][set_name] = {'Train': y_train_LR.ravel(),
                                        'Train_Labels': y_train,
                                        'Test': y_test_LR.ravel(),
                                        'Test_Labels': y_test}
                                       #'Weights': coefficients_RFC.ravel(),
                                       #'Bias': bias_RFC.ravel()}

    return Results


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def models_prediction_reg(df_train, df_test, models_dict, target_train, target_test=None, min_NPV=0.97, min_NPV_Models=False, random_state=None, standardization='default', imputation='knn', nan_masking=None, nan_masking_univ=True):

    '''
        Similar to models_prediction, but here we do regressions instead of classifications.
    '''
    
    if target_test==None:
        target_test = target_train
    
    Results = {}
    
    if 'RidR' in models_dict.keys():
        Results['RidR'] = {}
        
        for set_name in models_dict['RidR'].keys():
            #print('LR -', set_name)
            columns = models_dict['RidR'][set_name]
            
            ## Preprocessing
            #standardization: 'default' or 'PowerTransformer'
            Data_train = df_train[columns+[target_train, 'ID']]
            Data_test = df_test[columns+[target_test, 'ID']]
            
            nan_masking_val = nan_masking
            if nan_masking_univ==True:
                if len(columns)==1:
                    nan_masking_val = 0.0
                    
            Preprocessed_data_dict = preprocessing_V3(Data_train,
                                                      target_train,
                                                      target_test=target_test,
                                                      Data_test=Data_test,
                                                      standardization=standardization, 
                                                      imputation=imputation,  # 'default' or 'PowerTransformer' 
                                                      nan_masking=nan_masking_val)
            
            ## Training set
            X_train = Preprocessed_data_dict['Training set']['X'].values
            y_train = Preprocessed_data_dict['Training set']['Y'].values.ravel()
            ID_train = Preprocessed_data_dict['Training set']['ID'].values.ravel()

            ## Test set
            X_test = Preprocessed_data_dict['Test set']['X'].values
            ID_test = Preprocessed_data_dict['Test set']['ID'].values.ravel()
            y_test = Preprocessed_data_dict['Test set']['Y'].values.ravel()
            
            ## Apply pca
            if '#PCA' in set_name:
                substring = set_name.partition('#')[2]
                n_components = int(re.search(r'\d+', substring).group())
                pca = PCA(n_components=n_components, svd_solver='full')
                X_train = pca.fit_transform(X_train)
                if 'Test set' in Preprocessed_data_dict.keys():
                    X_test = pca.transform(X_test)
                                       
            ## Hyperparameters grid search
            model = Ridge()
            grid = {'alpha': [1e-4, 1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1e0, 5*1e0, 1e1], 
                    'tol': [1e-4]}
            score = 'neg_mean_squared_error'
            n_splits = 3
            hyperparameters = best_hyppar_gridsearch(X_train, y_train, model, grid, score, n_splits)
            C_reg = hyperparameters['alpha']
            model = Ridge(**hyperparameters)
            
            ## Prediction
            y_train_ridge = model.fit(X_train, y_train).predict(X_train)
            y_train_ridge = np.round(y_train_ridge)
            y_test_ridge = model.predict(X_test)
            y_test_ridge = np.round(y_test_ridge)
            coefficients_ridge = model.coef_
            bias_ridge = model.intercept_
            
            ## Prediction min NPV models
            # This needs changes... FIX!
            if min_NPV_Models:
                value_train_LR = LR.decision_function(X_train)
                value_test_LR = LR.decision_function(X_test)
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
            
            # Save results
            Results['RidR'][set_name] = {'Train': y_train_ridge.ravel(),
                                       'Train_Labels': y_train,
                                       'Test': y_test_ridge.ravel(),
                                       'Test_Labels': y_test,
                                       'Weights': coefficients_ridge.ravel(),
                                       'Bias': bias_ridge.ravel(), 
                                       'C': C_reg, 
                                       'Std_Parameters': Preprocessed_data_dict['Training set']['Std_Parameters']}
            
            if min_NPV_Models:
                Results['RidR'][set_name+'_minNPV'] = {'Train': y_train_LR_0.ravel(),
                                                     'Train_Labels': y_train,
                                                     'Test': y_test_LR_0.ravel(), 
                                                     'Test_Labels': y_test}

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


def age_masking(X_train, y_train, age_train, age_test):
    
    '''
       This function returns data masked by age. The lower bound is set according to min(age_test). 
       Input: 
             1) X_train: 2D np.array (n_samples x n_features).
             2) y_train: 1D np.array (n_samples).
             3) age_train: 1D np.array (n_samples).
             4) age_test: model trained (n_samples).
    '''
    
    
    ## Create local copies
    X_train_c = X_train.copy()
    y_train_c = y_train.copy()
    
    
    ## Define lower bound
    min_test_age = min(age_test)
    if min_test_age>90:
        #lower_bound = 85
        lower_bound = 85
    else:
        lower_bound = min_test_age - 5
    #elif min_test_age>=65:
    #    lower_bound = min_test_age - 5
    #elif min_test_age>=60:
    #    lower_bound = 60
    #else:
    #    lower_bound = min_test_age 
    
    
    ## Define upper bound
    max_test_age = max(age_test)
    upper_bound = max_test_age + 200 #20
    
    ## Apply masking
    age_mask = age_train >= lower_bound
    age_mask = age_mask & (age_train <= upper_bound)
    X_train_c = X_train_c[age_mask, :]
    y_train_c = y_train_c[age_mask]
    
    
    return X_train_c, y_train_c


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def classification_plot2D(X_1, X_2, y, classifier, X_1_test=np.array([]), X_2_test=np.array([]), y_test=np.array([]), levels=[-2, -0.6, -0.3, 0, 0.3, 0.6, 2]):
    
    '''
       This function produces a scatterplot of the (2D) training data, 
       with a heatmap of the passed classifier.
       Input: 
             1) X_1: 1D np.array with feature 1 values.
             2) X_2: 1D np.array with feature 2 values.
             3) y: 1D np.array with target values.
             4) classifier: model trained on X_1, X_2 and y.
             5) X_1_test: 1D np.array with feature 1 test values.
             6) X_2_test: 1D np.array with feature 2 test values.
             7) y: 1D np.array with target test values.
             8) levels: 1D list levels for contour plot.
    '''
    
    #SetPlotParams()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    
    ## Define grid
    h = .05 # .02  # step size in the mesh
    x_1_min, x_1_max = X_1.min() - 2.5, X_1.max() + 2.5
    x_2_min, x_2_max = X_2.min() - 2.5, X_2.max() + 2.5
    xx_1, xx_2 = np.meshgrid(np.arange(x_1_min, x_1_max, h),
                         np.arange(x_2_min, x_2_max, h))
    X = np.concatenate((X_1.reshape(-1, 1), X_2.reshape(-1, 1)), axis=1)


    ## Compute values for heatmap
    if hasattr(classifier, "decision_function"):
        Z = classifier.decision_function(np.c_[xx_1.ravel(), xx_2.ravel()])
    else:
        Z = classifier.predict_proba(np.c_[xx_1.ravel(), xx_2.ravel()])[:, 1]
    
    
    ## Put the result into a color plot
    Z = Z.reshape(xx_1.shape)
    min_lev = Z.min()
    max_lev = Z.max()
    levels = levels
    
    '''
    if min_lev<-2 or max_lev>2:
        max_lev_abs = max(abs(min_lev), abs(max_lev))
        new_leves_top = []
        new_leves_bottom = []
        if max_lev_abs/2. >= 2:
            new_leves_top = new_leves_top + [(max_lev_abs+2)/2.]
            new_leves_bottom = [-(max_lev_abs+2)/2.] + new_leves_bottom
        new_leves_top = new_leves_top + [max_lev_abs]
        new_leves_bottom = [-max_lev_abs] + new_leves_bottom
        levels = new_leves_bottom + levels
        levels = levels + new_leves_top
    '''
    
    min_lev = min(levels)
    max_lev = max(levels)
        
    ax.contourf(xx_1, xx_2, Z, cmap=cm, levels=levels, alpha=.5, vmin=min_lev, vmax=max_lev) 
    ax.contour(xx_1, xx_2, Z, levels=[0], colors=['black'], alpha=.4)

    
    ## Plot the training points
    ax.scatter(X_1, X_2, c=y, cmap=cm_bright, edgecolors='face', s=7, alpha=0.7)

    ## Plot the test points
    if len(X_1_test)>0 and len(X_2_test)>0:
        if len(y_test)>0:
            ax.scatter(X_1_test, X_2_test, c=y_test, vmin=0., vmax=1., cmap=cm_bright, marker='x', edgecolors='face', s=80, alpha=1)
        else:
            ax.scatter(X_1_test, X_2_test, color='black', marker='x', s=80, linewidth=2., alpha=1)
    
    ax.set_xlim(x_1_min, x_1_max)
    ax.set_ylim(x_2_min, x_2_max)
    #ax.set_xticks(())
    #ax.set_yticks(())
    
    plt.tight_layout()
    plt.show() 
    

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def reshape_datafromfile(Data_test):
    
    '''
       This function checks if the uploaded file (assigned to Data_test) containes the features used to make predictions, then organizes Data_test as a pandas dataframe with: n_rows = n_patients, n_col = n_features.
       NB: this reshaping is not strictly required to make predictions about patients.
       Input: 
             1) Data_test: pandas DataFrame with no header, n_rows = n_features (including ID), n_col = n_patients.
    '''
    
    Features = FeaturesTestSet
    
    
    ## Create local copy
    Data_test_local = Data_test.copy()
    
    
    ## Check features
    if not set(Data_test_local.index).issubset(Features):
        error_message = 'Formato file non valido. Caricare file Excel Workbook (.xlsx) con le seguenti righe:\n'+'\n'.join(Features)
        raise Exception(error_message)
      
    
    ## Reshape data
    try:
        Data_test_local.loc['ID', :] = Data_test_local.loc['ID', :].values.astype(int).astype(str)
    except:
        Data_test_local.loc['ID', :] = Data_test_local.loc['ID', :].values.astype(str)
    ID_test = Data_test_local.loc['ID', :].values
    Data_test_local.drop(index='ID', inplace=True)
    Data_test_local = pd.DataFrame(Data_test_local.values.transpose(), columns=Data_test_local.index.values, index=ID_test)
    
    
    ## Return data
    return Data_test_local


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def get_patient_data(Data_test):
    
    '''
       This function asks the ID of the patient to test, then looks for ID in Data_test and checks the associated values.
       It throws an error if any of the values is not numeric, positive or equal to zero.
       It returns the data associated with ID if no errors occurred.
       Input: 
             1) Data_test: pandas DataFrame with n_rows = n_patients (indexed by ID), n_col = n_features.
             2) The user inserts the ID string.
    '''
    
    ## Create local copy
    Data_test_local = Data_test.copy()
    
    
    ## Check ID
    ID = input('Inserire ID\n')
    flag = True
    while flag:        
        if ID in Data_test_local.index:
            x_testing = Data_test_local.loc[[ID], :]
            flag = False
        elif ID in ['exit', '\'exit\'', 'Exit']:
            sys.exit('Sessione terminata.')
            flag = False
        else:
            message = 'Inserire nuovo ID o inserire \'exit\' per terminare la sessione\n'
            ID = input(message)
            
    
    ## Check values
    error_message = 'Input non valido: controllare valori citofluorimetrici del paziente con ID='+ ID \
    + '.\nSono ammessi solo valori numerici positivi (gli zeri vengono interpretati come valori mancanti).'
    for element in x_testing.values.ravel():
        if isfloat(element) or pd.isnull(element):
            if isfloat(element):
                if element<0:
                    raise ValueError(error_message)
        else:
            raise ValueError(error_message)
                
    
    ## Convert to float
    x_testing = x_testing.astype(float)
    
    
    ## Sex
    flag = True
    while flag:
        sex = input('Inserire sesso (M, F)\n')
        if sex in ('M', 'F'):
            flag = False
            if sex == 'M':
                value = 0
            else:
                value = 1
    x_testing['sex'] = value

        
    ## Age
    flag = True
    while flag:
        #value = input('Inserire data di nascita gg/mm/aaaa (esempio: 25/12/1960)\n')        
        #match = re.match('^[0-9]{2}/[0-9]{2}/[0-9]{4}$', value)
        #if match or value=='':
        #    if value=='':
        #        value = np.nan
        #        flag = False
        #    else:
        #        datetime_obj = datetime.datetime.strptime(value, '%d/%m/%Y')
        #        day, month, year = datetime_obj.day, datetime_obj.month, datetime_obj.year
        #        if day>0 and day<32:
        #            if month>0 and month<13:
        #                if year>1890 and year<2020:
        #                    value = calculate_age(datetime_obj)
        #                    flag = False
        inserted_value = input('Inserire anno di nascita (esempio: 1940)\n')
        if isfloat(inserted_value):
            year = int(inserted_value)
            if year>1889 and year<2021:
                datetime_str = '01/07/' + inserted_value
                datetime_obj = datetime.datetime.strptime(datetime_str, '%d/%m/%Y')
                value = calculate_age(datetime_obj)
                flag = False
            else:
                print('Valore inserito fuori dal range [1890, 2020]')
        elif inserted_value == '':
            value = np.nan
            flag = False
                
    x_testing['age'] = value
    
    
    ## ID
    x_testing['ID'] = ID
        
    ## Return test set as pandas dataframe
    return x_testing, year, sex
    

# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# ---- # ---- # ---- # ---- # ---- # ---- # ---- # ---- #


def calculate_age(birth_date):
    
    '''
       This function calculates current age from datetime object birth_date.
    '''
    
    born = birth_date
    today = date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    
    return age


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