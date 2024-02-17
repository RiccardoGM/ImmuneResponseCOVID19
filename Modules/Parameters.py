## Minimum NPV
min_NPV_Models = True
min_NPV = 0.97

## Correlation threshod
#corr_th = 0.8
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
std_cat_variables = True

## PCA 
pca_var_threshold = 0.05

## Train-test
test_size = 0.30
ignore_sex = False

## Age
age_min = 30 # 30 or 70
age_max = 100 # 100 or 70
if age_max<=70:
    ignore_sex = True # Condition required to split data in this stratum

## Delta onset
donset_min = 0 # 0 or 11
donset_max = 30 # 30 or 10

## Target
train_target = 'IOT+death' # death, merged_death, infectious_complications, IOT+death, IOT+ICU+death, IOT+death+WHOge4
test_target = train_target

## Plot
plot_minNPV_models = False

## Feature selection
use_manual_selection = True

## Dataset to use
use_CCIMasked_dataset = True

## Regulariser
find_regulariser_before_average = False
hyperparameters_grid_LR = {'C': [1e-4, 1e-3, 1e-2, 5*1e-2, 1e-1, 5*1e-1, 1e0, 5*1e0, 1e1], 
                           'class_weight': ['balanced'],
                           'penalty': ['l2'],
                           'max_iter': [1000]}
score = 'f1'
n_splits_gridsearch = 3

## Variables of interest
immunecells_set = ['NK/uL', 'B CD19/uL', 'T CD3/uL', 'T CD4/uL', '% T CD4 HLADR POS', 'T CD8/uL', '% T CD8 HLADR POS', 
                   'WBC/uL', 'NeutroBaEu/uL', 'Mono/uL', 'MONO DR IFI', 'Mono DR %', 'Linfo/uL', 'LRTE/uL', 'LRTE % dei CD4']
cytokines_set = ['IFNGC', 'IL10', 'IL1B', 'IL2R', 'IL6', 'IL8', 'IP10']
demographics_set = ['age', 'sex', 'delta_onset']
scores_set = ['CCI (charlson comorbidity index)', 'SOFA', 'NEWS', 'qCSI', '4 C score']
biomarkers_set = ['PROADM', 'LDH', 'PCR']
output_set = ['merged_death', 'IOT+death', 'IOT+ICU+death', 'WHO=>3']
allinput_set = demographics_set + immunecells_set + cytokines_set + biomarkers_set + scores_set

## Variables of multivariate models
FC_set = ['NeutroBaEu/uL', 'Mono/uL', 'MONO DR IFI', 'LRTE % dei CD4', 'T CD3/uL', 'B CD19/uL']
Dem_set = ['age', 'sex', 'delta_onset']
CK_set = ['IL2R', 'IL6', 'IL8', 'IL10', 'IP10']
BM_set = ['PROADM', 'LDH', 'PCR']

## Other variables
comorbidities_set = ['obesity', 'dyslipidemia', 'cardiovascular_disease', 'diabetes', 'BPCO', 'IRC', 'hepatopathy',
                     'hypertension', 'tumor', 'oncohematology', 'autoimmunity', 'immunosuppressed']
TC_set = ['parenchima_width', 'TC1_level']
therapies_set = ['cortisone']
hometherapies_set = ['home_therapy#immunosuppressants', 'home_therapy#immunomodulators', 'home_therapy#steroids']

## File names
file_name_inpatients = 'DataInpatients_CCIMasked.xlsx'
file_name_outpatients = 'DataOutpatients.xlsx'


# ------- #

## Paths
path_results = '/Users/riccardo/Documents/GitHub/COVID19Classification/Results/'
path_figures = '/Users/riccardo/Documents/GitHub/COVID19Classification/Figures/'
path_datasets = '/Users/riccardo/Documents/GitHub/COVID19Classification/Data/'
path_setsdescription = '/Users/riccardo/Documents/GitHub/COVID19Classification/DatasetsDescription/'


## --- Folder basic stat --- ##
foldername_basicstatresults = 'BasicStatAnalysis/'


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

# Dataset
if use_CCIMasked_dataset:
    exp_description = exp_description + '_DatasetMaskedByCCI'
    
# Regulariser
if find_regulariser_before_average:
    exp_description = exp_description + '_RegFixed#True'
else:
    exp_description = exp_description + '_RegFixed#False'

## --- Folder multiv. --- ##
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


## --- Folder univ. --- ##

exp_univ_description = ''

# Nan masking
if do_nan_masking_univ:
    exp_univ_description = exp_univ_description + '_NanMask#True'
else:
    exp_univ_description = exp_univ_description + '_NanMask#False'

foldername_univ = 'UnivModels' + exp_description + exp_univ_description + '/'
