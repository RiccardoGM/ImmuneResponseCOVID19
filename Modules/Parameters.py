## --- Parameters and methods --- ##

# Minimum NPV
min_NPV_Models = True # whether to eval. min NPV models
min_NPV = 0.97

# Min % no nans per column
perc_nonans = 10
perc_nonans_univ = 10

# Nan masking row-wise
do_nan_masking = True
do_nan_masking_univ = True
do_nan_masking_groupwise = True # apply nan-masking group-wise for each row
nan_masking = 0.5 # max % nans allowed per row or per group

# N samples for average
N_av = 100 #100

# Imputation
imputation_method = 'knn'
imputation_method_univ = 'mean'

# Standardization
std_method = 'PowerTransformer' # 'PowerTransformer' or 'StandardScaler'
z_score_th = 3.
std_cat_variables = True

# PCA 
pca_var_threshold = 0.05

# Preprocessing
do_preprocessing_multiv = True # Always True
do_preprocessing_univ = True # True or False (False: no standardization)

# Train-test
test_size = 0.30
ignore_sex = False

# Age
age_min = 30 # 30 or 70
age_max = 100 # 100 or 70
if age_max<=70:
    ignore_sex = True # Required to split data in this stratum

# Delta onset
donset_min = 0 # 0 or 11
donset_max = 30 # 10 or 30

# Target
train_target = 'OTI+death'
test_target = train_target

# Regulariser
find_regulariser_before_average = False
hyperparameters_grid_LR = {'C': [1e-4, 1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1e0, 5*1e0, 1e1], 
                           'class_weight': ['balanced'],
                           'penalty': ['l2'],
                           'max_iter': [1000]}
score = 'f1'
n_splits_gridsearch = 3



## --- Main variables --- ##

# Variables of interest
immunecells_set = ['NK/uL', 'B CD19/uL', 'T CD3/uL', 'T CD4/uL', '% T CD4 HLADR+', 'T CD8/uL', '% T CD8 HLADR+', 
                   'WBC/uL', 'Granulo/uL', 'Mono/uL', 'Mono DR IF', 'Mono DR %', 'Lymph/uL', 'RTE/uL', 'RTE % CD4']
cytokines_set = ['IFNGC', 'IL10', 'IL1B', 'IL2R', 'IL6', 'IL8', 'IP10']
demographics_set = ['age', 'sex', 'delta_onset']
scores_set = ['CCI (charlson comorbidity index)', 'SOFA', 'NEWS', 'qCSI', '4C']
biomarkers_set = ['proADM', 'LDH', 'CRP']
output_set = ['death', 'OTI+death', 'OTI+ICU+death', 'WHO=>3']
allinput_set = demographics_set + immunecells_set + cytokines_set + biomarkers_set + scores_set

# Variables of multivariate models
FC_set = ['Granulo/uL', 'Mono/uL', 'Mono DR IF', 'RTE % CD4', 'T CD3/uL', 'B CD19/uL']
Dem_set = ['age', 'sex', 'delta_onset']
CK_set = ['IL2R', 'IL6', 'IL8', 'IL10', 'IP10']
BM_set = ['proADM', 'LDH', 'CRP']

# Comorbidity variables
comorbidities_set = ['obesity', 'dyslipidemia', 'CVDs', 'diabetes', 'COPD', 'CKI', 'hepatopathy',
                     'hypertension', 'tumor', 'oncohematology', 'autoimmunity', 'immunosuppressed']



## --- Input files --- ##

file_name_inpatients = 'DataInpatients_anonymized.xlsx' # Data filtered by CCI
file_name_outpatients = 'DataOutpatients_anonymized.xlsx'



## --- Main directories --- ##

import os
this_path = os.path.abspath('') 
parent_dir = os.path.dirname(this_path)
path_results = parent_dir + '/Results/'
path_figures = parent_dir + '/Figures/'
path_datasets = parent_dir + '/Data/'
path_setsdescription = parent_dir + '/DatasetsDescription/'



## --- Directory stat. analysis --- ##

foldername_statresults = 'DescriptiveStatistics/'



## --- Experiment description --- ##

# Delta onset
exp_description = '_DOnsetMin#%d#Max#%d' % (donset_min, donset_max)

# Age
exp_description = exp_description + '_AgeMin#%d#Max#%d' % (age_min, age_max)

# Target
if train_target==test_target:
    if train_target=='merged_death':
        str_name = 'death'
    else:
        str_name = train_target
    if '_' in str_name:
        str_name.replace('_', '-')
    exp_description = exp_description + '_Target#%s' % str_name
else:
    if train_target=='merged_death':
        str_name1 = 'death'
    else:
        str_name1 = train_target
    if test_target=='merged_death':
        str_name2 = 'death'
    else:
        str_name2 = test_target
    if '_' in str_name1:
        str_name1.replace('_', '-')
    if '_' in str_name2:
        str_name2.replace('_', '-')
    exp_description = exp_description + '_TargetTrain#%s#Test#%s' % (str_name1, str_name2)

# Std method
exp_description = exp_description + '_Std#%s' % (std_method)



## --- Directory multiv. --- ##

exp_multiv_description = ''

# Nan masking
if do_nan_masking:
    if do_nan_masking_groupwise:
        exp_multiv_description = exp_multiv_description + '_NansRowGroupwise#%d' % int(100*nan_masking)
    else:
        exp_multiv_description = exp_multiv_description + '_NansRow#%d' % int(100*nan_masking)
    
# PCA % var. threshold
exp_multiv_description = exp_multiv_description + '_PCAPercVarTh#%d' % int(pca_var_threshold*100)

foldername_multiv = 'MultivModels' + exp_description + exp_multiv_description + '/'



## --- Directory univ. --- ##

exp_univ_description = ''

# Nan masking
if do_nan_masking_univ:
    exp_univ_description = exp_univ_description + '_NanMask#True'
else:
    exp_univ_description = exp_univ_description + '_NanMask#False'

# Preprocessing
if not do_preprocessing_univ:
    exp_univ_description = exp_univ_description + '_Std#False'

foldername_univ = 'UnivModels' + exp_description + exp_univ_description + '/'



## --- Colors --- ##

pink = '#E38A8A'
light_green = '#A3F0A3'
green = '#7FB285'
pine_green = '#136F63'
light_pine_green = '#79DDD0'
violet = '#A888BF'
light_red_purp = '#FF616D'
red_purp = '#F73B5C'
light_blue = '#3C8DAD'
queen_blue = '#456990'
dark_blue = '#125D98'
orange = '#F5A962'
yellow = '#FFD966'
lavander = '#D5C6E0'
african_violet = '#B084CC'
dark_liver = '#56494C'
dark_purple = '#30011E'
grey_green = '#5B7B7A'
coral = '#FF8360'
dark_coral = '#FF6F47'
rose = '#FF69A5'
electricblue = '#7DF9FF'
blue_green = '#77CFBF'
dark_dark_blue = '#0B385B'
dark_blue_green = '#40b5a0'
eton_blue = '#96c8a2'
light_grey = '#DDDDDD'
grey_1 = '#AAAAAA'
grey_2 = '#DDDDDD'