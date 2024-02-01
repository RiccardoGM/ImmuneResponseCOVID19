## Minimum NPV
min_NPV_Models = True
min_NPV = 0.97

## Correlation threshod
corr_th = 0.8
corr_th_univ = 0.99

## Min % no nans per column
perc_nonans = 10
perc_nonans_univ = 10

## Nan masking row-wise
do_nan_masking = True
do_nan_masking_univ = True
do_nan_masking_groupwise = True # if to apply nan-masking group-wise for each row
nan_masking = 0.5 # max % nans allowed per row or per group

## Reference time
ref_time = 't0'

## N samples for average
N_av = 100

## Imputation
imputation_method = 'knn'
imputation_method_univ = 'mean'

## Standardization
std_method = 'PowerTransformer' # 'PowerTransformer' or 'StandardScaler'
z_score_th = 3.

## PCA 
pca_var_threshold = 0.05

## Train-test
test_size = 0.30

## Age
age_min = 30 # 30
age_max = 100 # 100

## Delta onset
donset_masking = True
donset_min = 0 # 0
donset_max = 30 # 30

## Target
train_target = 'IOT+death' # death, merged_death, infectious_complications, IOT+death, IOT+ICU+death, IOT+death+WHOge4
test_target = train_target

## Plot
plot_minNPV_models = False

## Feature selection
use_manual_selection = True

## Dataset to use
use_CCIWHOMasked_dataset = True

## Regulariser
find_regulariser_before_average = True

# ------- #

## Paths
path_results = '/Users/riccardo/Documents/GitHub/COVID19Classification/Results/'
path_figures = '/Users/riccardo/Documents/GitHub/COVID19Classification/Figures/'
path_datasets = '/Users/riccardo/Documents/GitHub/COVID19Classification/Data/'

## --- Folder basic stat --- ##
foldername_basicstatresults = 'BasicStatAnalysis/'


## --- Experiment description --- ##
exp_description = ''

# Delta onset
if donset_masking:
    if donset_min:
        exp_description = exp_description + '_DOnsetMin#%d' % donset_min
        if donset_max:
            exp_description = exp_description + '#Max#%d' % donset_max
    elif donset_max:
        exp_description = exp_description + '_DOnsetMax#%d' % donset_max

# Age
if age_min:
    exp_description = exp_description + '_AgeMin#%d' % age_min
    if age_max:
        exp_description = exp_description + '#Max#%d' % age_max
elif age_max:
    exp_description = exp_description + '_AgeMax#%d' % age_max

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

# Dataset
if use_CCIWHOMasked_dataset:
    exp_description = exp_description + '_DatasetMaskedByCCIWHO'
    
# Regulariser
if find_regulariser_before_average:
    exp_description = exp_description + '_RegFixed#True'
else:
    exp_description = exp_description + '_RegFixed#False'

## --- Folder multiv. --- ##
exp_multiv_description = ''

# CorrTh
exp_multiv_description = exp_multiv_description + '_CorrTh#%s' % (str(corr_th).replace('0.', ''))

# Nan masking
if do_nan_masking:
    if do_nan_masking_groupwise:
        exp_multiv_description = exp_multiv_description + '_NansRowGroupwise#%d' % int(100*nan_masking)
    else:
        exp_multiv_description = exp_multiv_description + '_NansRow#%d' % int(100*nan_masking)
    

if do_nan_masking_univ:
    exp_multiv_description = exp_multiv_description + '_UnivNanMask#True'
else:
    exp_multiv_description = exp_multiv_description + '_UnivNanMask#False'
    
# PCA % var. threshold
exp_multiv_description = exp_multiv_description + '_PCAPercVarTh#%d' % int(pca_var_threshold*100)
    
foldername_multiv = 'MultivModels' + exp_description + exp_multiv_description + '/'


## --- Folder univ. --- ##
exp_univ_description = ''
# Nan masking
if do_nan_masking_univ:
    exp_univ_description = exp_univ_description + '_NanMask#True'
else:
    exp_univ_description = exp_univ_description + '_NanMask#False'

foldername_univ = 'UnivModels' + exp_description + exp_univ_description + '/'
