# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import sys
import os

# Add path to custom modules
os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'
sys.path.append('/Users/riccardo/Documents/GitHub/COVID19Classification/')
print(sys.version)

# Import custom modules
from Modules import Parameters

## ---------- ##

# Parameters
file_name_inpatients = Parameters.file_name_inpatients
file_name_outpatients = Parameters.file_name_outpatients
min_age = Parameters.age_min 
max_age = Parameters.age_max
min_donset = Parameters.donset_min
max_donset = Parameters.donset_max
z_score_th = Parameters.z_score_th
path_import_inpatients = Parameters.path_datasets + file_name_inpatients
path_import_outpatients = Parameters.path_datasets + file_name_outpatients

# Variables of interest
immunecells_set = Parameters.immunecells_set
cytokines_set = Parameters.cytokines_set
demographics_set = Parameters.demographics_set
scores_set = Parameters.scores_set
biomarkers_set = Parameters.biomarkers_set
output_set = Parameters.output_set
allinput_set = Parameters.allinput_set

## ---------- ##

def data_preprocessing(min_age=min_age, max_age=max_age, min_donset=min_donset, max_donset=max_donset, z_score_th=z_score_th,
                       path_import_inpatients=path_import_inpatients, path_import_outpatients=path_import_outpatients):

    # Import data of hostpitalized patients
    DataInpatients = pd.read_excel(path_import_inpatients, engine='openpyxl')
    DataInpatients.drop(columns=['Unnamed: 0'], inplace=True)

    # Import data of non-hostpitalized patients
    DataOutpatients = pd.read_excel(path_import_outpatients, engine='openpyxl')
    DataOutpatients.drop(columns=['Unnamed: 0'], inplace=True)

    # Format nan/nat to none
    DataInpatients = DataInpatients.where(DataInpatients.notnull().values, -1e100)
    DataInpatients = DataInpatients.where(DataInpatients.values!=-1e100, np.nan)
    DataOutpatients = DataOutpatients.where(DataOutpatients.notnull().values, -1e100)
    DataOutpatients = DataOutpatients.where(DataOutpatients.values!=-1e100, np.nan)

    # Add neutro* variable
    new_idx = DataInpatients.columns.get_loc('WBC/uL') + 1
    v = DataInpatients['WBC/uL'].values - (DataInpatients['Mono/uL'].values + DataInpatients['Linfo/uL'].values)
    v[v<0] = 0
    DataInpatients.insert(loc=new_idx, column='NeutroBaEu/uL', value=v)
    new_idx = DataOutpatients.columns.get_loc('WBC/uL') + 1
    v = DataOutpatients['WBC/uL'].values - (DataOutpatients['Mono/uL'].values + DataOutpatients['Linfo/uL'].values)
    v[v<0] = 0
    DataOutpatients.insert(loc=new_idx, column='NeutroBaEu/uL', value=v)

    # Add delta_onset column
    dates = [date for date in DataInpatients.columns if 'date#' in date]    
    ref_date_str = '01-01-2020'
    ref_date = pd.Timestamp(ref_date_str)
    d_dates = {}
    for date in dates:
        d_dates[date] = np.array([(element-ref_date).days if pd.notnull(element) else np.nan for element in DataInpatients[date]])
    d_dates_df = pd.DataFrame(d_dates, columns=dates, index=DataInpatients['ID'])
    date_flowcyt_exam = d_dates_df['date#flowcyt_exam'].values
    date_onset = d_dates_df['date#onset'].values    
    delta_onset_df = pd.DataFrame(date_flowcyt_exam - date_onset, columns=['delta_onset'], index=d_dates_df.index)
    v = delta_onset_df.values
    DataInpatients['delta_onset'] = v
    DataOutpatients['delta_onset'] = np.nan

    # Filter by age
    age = np.round(DataInpatients['age'].values)
    age_mask = (age >= min_age) & (age < max_age)
    DataInpatients = DataInpatients.loc[age_mask, :]

    # Filter by delta_onset
    donset = np.round(DataInpatients['delta_onset'].values)
    donset_mask = (donset >= min_donset) & (donset <= max_donset) & (pd.notnull(donset))
    DataInpatients = DataInpatients.loc[donset_mask, :]

    # Final dataset
    std = StandardScaler()
    ptr = PowerTransformer()
    for name in allinput_set:
        z = ptr.fit_transform(std.fit_transform(DataInpatients[name].values.reshape(-1, 1)))
        is_outlier = abs(z) > z_score_th
        n_outliers = sum(is_outlier)
        if n_outliers:
            DataInpatients[name].where(is_outlier.reshape(-1,)==False, inplace=True)

    # Reset indices
    DataInpatients.reset_index(inplace=True)
    DataOutpatients.reset_index(inplace=True)
    
    return DataInpatients[['ID']+allinput_set+output_set], DataOutpatients[['ID']+allinput_set+output_set]