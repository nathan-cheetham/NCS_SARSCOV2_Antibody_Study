# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:52:12 2022

@author: k2143494

'Within-between' twin pair generalised linear mixed effects regression models to test associations between antibody levels following SARS-CoV-2 vaccination and various factors within TwinsUK cohort https://doi.org/10.7554/eLife.80428
"""

import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt 
plt.rc("font", size=12)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#%% Set parameters
within_var = 'DiffToMean' # DiffToMean // AsIs . specify whether to use individual variable 'as-is', or use the 'difference to mean' for the 'within pair' coefficient


#%% Create functions
### Function to summarise statsmodel results in dataframe
# From https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
def results_summary_to_dataframe(results, round_dp_coeff):
    '''take the result of an statsmodel results table and transforms it into a dataframe'''
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    results_df = pd.DataFrame({"pvals":pvals,
                               "coeff":coeff,
                               "conf_lower":conf_lower,
                               "conf_higher":conf_higher
                                })
    
    # Reorder columns
    results_df = results_df[["coeff","conf_lower","conf_higher","pvals"]]
        
    # Highlight variables where confidence intervals are both below 1 or both above 1
    results_df.loc[(results_df['pvals'] < 0.05)
                        ,'Significance'] = 'Significant, *, p < 0.05'
    results_df.loc[(results_df['pvals'] < 0.01)
                        ,'Significance'] = 'Significant, **, p < 0.01'
    results_df.loc[(results_df['pvals'] < 0.001)
                        ,'Significance'] = 'Significant, ***, p < 0.001'
    
    round_dp = round_dp_coeff
    
    results_df['tidy_string'] = results_df['coeff'].round(round_dp).astype(str) + ' (' + results_df['conf_lower'].round(round_dp).astype(str) + ', ' + results_df['conf_higher'].round(round_dp).astype(str) + '),' 
    
    results_df.loc[(results_df['pvals'] >= 0.05)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(2).astype(str)
    results_df.loc[(results_df['pvals'] < 0.05) &
                   (results_df['pvals'] >= 0.01)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(2).astype(str) + '*'
    results_df.loc[(results_df['pvals'] < 0.01) &
                   (results_df['pvals'] >= 0.001)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(3).astype(str) + '**'
    results_df.loc[(results_df['pvals'] < 0.001) &
                   (results_df['pvals'] >= 0.0001)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p = ' + results_df['pvals'].round(4).astype(str) + '***'
    results_df.loc[(results_df['pvals'] < 0.0001)
                   ,'tidy_string'] = results_df['tidy_string'] + ' p < 0.0001***'
    
    
    return results_df


#%% Load data 
# Twins Covid study results combined with demographics (i.e. final combined and processed dataset)
combined_flat = pd.read_csv(r"TwinsCovid_CoPE_Antibody_Antigen_flat.csv")
col_list = combined_flat.columns.to_list() # save column names


#%% Processing
# -----------------------------------------------------------------------------
# Convert age to numeric field - 'NoDataAvailable' to NaN
combined_flat['age'] = pd.to_numeric(combined_flat['age'], errors = 'coerce')

# -----------------------------------------------------------------------------
# Add number of weeks since 1st and 2nd vaccination
combined_flat['WeeksSinceVacc_1'] = (combined_flat['DaysSinceVacc_1']/7).apply(np.floor) # round down
combined_flat['WeeksSinceVacc_2'] = (combined_flat['DaysSinceVacc_2']/7).apply(np.floor) # round down
combined_flat['WeeksSinceVacc_3'] = (combined_flat['DaysSinceVacc_3']/7).apply(np.floor) # round down

# Add number of weeks since 1st and 2nd vaccination
combined_flat['TwoWeeksSinceVacc_1'] = (combined_flat['DaysSinceVacc_1']/14).apply(np.floor) # round down
combined_flat['TwoWeeksSinceVacc_2'] = (combined_flat['DaysSinceVacc_2']/14).apply(np.floor) # round down
combined_flat['TwoWeeksSinceVacc_3'] = (combined_flat['DaysSinceVacc_3']/14).apply(np.floor) # round down

# -----------------------------------------------------------------------------
# Group Vaccine name as Oxford/Pfizer/Other
# Vaccine 1
vacc_other_list = ['Moderna', 'Other - please specify', 'Janssen/ Johnson & Johnson', 'Novavax', 'Valneva']
combined_flat.loc[(combined_flat['Vaccine_1_name'] == 'Oxford AstraZeneca'),'Vaccine_1_name_grouped'] = 'Oxford AstraZeneca'
combined_flat.loc[(combined_flat['Vaccine_1_name'] == 'Pfizer BioNTech'),'Vaccine_1_name_grouped'] = 'Pfizer BioNTech'
combined_flat.loc[(combined_flat['Vaccine_1_name'].isin(vacc_other_list)),'Vaccine_1_name_grouped'] = 'Other'
combined_flat['Vaccine_1_name_grouped'] = combined_flat['Vaccine_1_name_grouped'].fillna(combined_flat['Vaccine_1_name'])

# Vaccine 2
combined_flat.loc[(combined_flat['Vaccine_2_name'] == 'Oxford AstraZeneca'),'Vaccine_2_name_grouped'] = 'Oxford AstraZeneca'
combined_flat.loc[(combined_flat['Vaccine_2_name'] == 'Pfizer BioNTech'),'Vaccine_2_name_grouped'] = 'Pfizer BioNTech'
combined_flat.loc[(combined_flat['Vaccine_2_name'].isin(vacc_other_list)),'Vaccine_2_name_grouped'] = 'Other'
combined_flat['Vaccine_2_name_grouped'] = combined_flat['Vaccine_2_name_grouped'].fillna(combined_flat['Vaccine_2_name'])

# Vaccine 3
vacc_other_list = ['Oxford AstraZeneca', 'Other - please specify', 'Janssen/ Johnson & Johnson', 'Novavax', 'Valneva']
combined_flat.loc[(combined_flat['Vaccine_3_name'] == 'Moderna'),'Vaccine_3_name_grouped'] = 'Moderna'
combined_flat.loc[(combined_flat['Vaccine_3_name'] == 'Pfizer BioNTech'),'Vaccine_3_name_grouped'] = 'Pfizer BioNTech'
combined_flat.loc[(combined_flat['Vaccine_3_name'].isin(vacc_other_list)),'Vaccine_3_name_grouped'] = 'Other'
combined_flat['Vaccine_3_name_grouped'] = combined_flat['Vaccine_3_name_grouped'].fillna(combined_flat['Vaccine_3_name'])


# -----------------------------------------------------------------------------
# Create wider grouping of self-reported had covid ever 
# Group self-reported positive test
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport'].str.contains('2. SuspectedCovid'))
                  , 'HadCovid_Ever_SelfReport_grouped'] = '2. SuspectedCovid'
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport'].str.contains('3. PositiveCovid'))
                  , 'HadCovid_Ever_SelfReport_grouped'] = '3. PositiveCovid'

combined_flat['HadCovid_Ever_SelfReport_grouped'] = combined_flat['HadCovid_Ever_SelfReport_grouped'].fillna(combined_flat['HadCovid_Ever_SelfReport'])

# Binary grouping as positive confirmed has too few < 250 results to present
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport_grouped'].isin(['2. SuspectedCovid', '3. PositiveCovid']))
                  , 'HadCovid_Ever_SelfReport_binary'] = '2-3. SuspectedOrPositiveCovid'

combined_flat['HadCovid_Ever_SelfReport_binary'] = combined_flat['HadCovid_Ever_SelfReport_binary'].fillna(combined_flat['HadCovid_Ever_SelfReport_grouped'])

# Rename original variable to distinguish naming
combined_flat = combined_flat.rename(columns = {'HadCovid_Ever_SelfReport':'HadCovid_Ever_SelfReport_original'}) 


# -----------------------------------------------------------------------------
# Create count of comorbidites. Use subdomains only
col_list = ['Domain_CancerAny', 'Domain_ChronicLungDisease', 'Domain_Osteoporosis', 'Domain_DiabetesType2','SubDomain_HeartFailure', 'SubDomain_AtrialFibrillation', 'SubDomain_CoronaryHeartDisease','SubDomain_Hypertension', 'SubDomain_AnxietyStressDisorder', 'SubDomain_Depression', 'SubDomain_Stroke', 'SubDomain_Alzheimers', 'SubDomain_RheumatoidArthritis','SubDomain_HighCholesterol', 'SubDomain_Asthma', 'SubDomain_DiabetesType1', 'SubDomain_Epilepsy', 'SubDomain_LiverCirrhosis']
# Sum col list to get multi-morbidity count
nan_fill_text = 'NoDataAvailable'
combined_flat['SubDomain_MultimorbidityCount_Ungrouped'] = combined_flat[col_list].replace({nan_fill_text:np.nan,
                                                                                            'Yes':1,
                                                                                            'No':0
                                                                                            }).astype(float).astype('Int64').sum(skipna = True, axis = 1)

# Count 'NoDataAvailable'
combined_flat['SubDomain_MultimorbidityCount_NoDataCount'] = combined_flat[col_list].replace({nan_fill_text:1,
                                                                                            'Yes':0,
                                                                                            'No':0
                                                                                            }).astype(float).astype('Int64').sum(skipna = True, axis = 1)

test = combined_flat[col_list + ['SubDomain_MultimorbidityCount_Ungrouped','SubDomain_MultimorbidityCount_NoDataCount']]

# Exclude individuals with more than half of data missing - 9+/18, as skews to lower numbers
max_missing = 9
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_NoDataCount'] >= max_missing)
                  ,'SubDomain_MultimorbidityCount_Ungrouped'] = np.nan # Really 'insufficient data'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_NoDataCount'] >= max_missing)
                  ,'SubDomain_MultimorbidityCount_Grouped'] = nan_fill_text # Really 'insufficient data'

# Group into categorical variable
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Ungrouped'] == 0)
                  ,'SubDomain_MultimorbidityCount_Grouped'] = '0'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Ungrouped'] >= 1) 
                  & (combined_flat['SubDomain_MultimorbidityCount_Ungrouped'] <= 3)
                  ,'SubDomain_MultimorbidityCount_Grouped'] = '1-3'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Ungrouped'] >= 4)
                  ,'SubDomain_MultimorbidityCount_Grouped'] = '4+'

combined_flat['SubDomain_MultimorbidityCount_Ungrouped'] = combined_flat['SubDomain_MultimorbidityCount_Ungrouped'].fillna(nan_fill_text)

test = combined_flat[col_list + ['SubDomain_MultimorbidityCount_Ungrouped','SubDomain_MultimorbidityCount_Grouped']]


# -----------------------------------------------------------------------------
# Combine certain comorbidities
# Heart condition (CHD or Heart Failure)
combined_flat.loc[(combined_flat['SubDomain_HeartFailure'] == 'NoDataAvailable') 
                  & (combined_flat['SubDomain_CoronaryHeartDisease'] == 'NoDataAvailable')
                  ,'SubDomain_HeartDiseaseOrFailure'] = 'NoDataAvailable'

combined_flat.loc[((combined_flat['SubDomain_HeartFailure'] == 'No') & (combined_flat['SubDomain_CoronaryHeartDisease'] == 'No'))
                  | ((combined_flat['SubDomain_HeartFailure'] == 'No') & (combined_flat['SubDomain_CoronaryHeartDisease'] == 'NoDataAvailable'))
                  | ((combined_flat['SubDomain_HeartFailure'] == 'NoDataAvailable') & (combined_flat['SubDomain_CoronaryHeartDisease'] == 'No'))
                  ,'SubDomain_HeartDiseaseOrFailure'] = 'No'

combined_flat.loc[(combined_flat['SubDomain_HeartFailure'] == 'Yes') | (combined_flat['SubDomain_CoronaryHeartDisease'] == 'Yes')
                  ,'SubDomain_HeartDiseaseOrFailure'] = 'Yes'

# Anxiety or Depression
combined_flat.loc[(combined_flat['SubDomain_AnxietyStressDisorder'] == 'NoDataAvailable') 
                  & (combined_flat['SubDomain_Depression'] == 'NoDataAvailable')
                  ,'SubDomain_AnxietyOrDepression'] = 'NoDataAvailable'

combined_flat.loc[((combined_flat['SubDomain_AnxietyStressDisorder'] == 'No') & (combined_flat['SubDomain_Depression'] == 'No'))
                  | ((combined_flat['SubDomain_AnxietyStressDisorder'] == 'No') & (combined_flat['SubDomain_Depression'] == 'NoDataAvailable'))
                  | ((combined_flat['SubDomain_AnxietyStressDisorder'] == 'NoDataAvailable') & (combined_flat['SubDomain_Depression'] == 'No'))
                  ,'SubDomain_AnxietyOrDepression'] = 'No'

combined_flat.loc[(combined_flat['SubDomain_AnxietyStressDisorder'] == 'Yes') | (combined_flat['SubDomain_Depression'] == 'Yes')
                  ,'SubDomain_AnxietyOrDepression'] = 'Yes'

test = combined_flat[['SubDomain_AnxietyStressDisorder','SubDomain_Depression','SubDomain_AnxietyOrDepression']]

# Combine diabetes type I and II
combined_flat.loc[(combined_flat['SubDomain_DiabetesType1'] == 'NoDataAvailable') 
                  & (combined_flat['Domain_DiabetesType2'] == 'NoDataAvailable')
                  ,'SubDomain_DiabetesAny'] = 'NoDataAvailable'

combined_flat.loc[((combined_flat['SubDomain_DiabetesType1'] == 'No') & (combined_flat['Domain_DiabetesType2'] == 'No'))
                  | ((combined_flat['SubDomain_DiabetesType1'] == 'No') & (combined_flat['Domain_DiabetesType2'] == 'NoDataAvailable'))
                  | ((combined_flat['SubDomain_DiabetesType1'] == 'NoDataAvailable') & (combined_flat['Domain_DiabetesType2'] == 'No'))
                  ,'SubDomain_DiabetesAny'] = 'No'

combined_flat.loc[(combined_flat['SubDomain_DiabetesType1'] == 'Yes') | (combined_flat['Domain_DiabetesType2'] == 'Yes')
                  ,'SubDomain_DiabetesAny'] = 'Yes'

# -----------------------------------------------------------------------------
# Create count of comorbidites - SELECTED COMORBIDITIES COMMON TO MOST OTHER COHORTS
col_list = ['SubDomain_AnxietyOrDepression', 'SubDomain_DiabetesAny', 'SubDomain_HeartDiseaseOrFailure', 'Domain_CancerAny', 'SubDomain_Hypertension']
# Sum col list to get multi-morbidity count
nan_fill_text = 'NoDataAvailable'
combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] = combined_flat[col_list].replace({nan_fill_text:np.nan,
                                                                                            'Yes':1,
                                                                                            'No':0
                                                                                            }).astype(float).astype('Int64').sum(skipna = True, axis = 1)

# Count 'NoDataAvailable'
combined_flat['SubDomain_MultimorbidityCount_Selected_NoDataCount'] = combined_flat[col_list].replace({nan_fill_text:1,
                                                                                            'Yes':0,
                                                                                            'No':0
                                                                                            }).astype(float).astype('Int64').sum(skipna = True, axis = 1)


# Exclude individuals with 'too much' comorbidtiy data missing - 2+/5, as skews to lower numbers
max_missing = 2
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_NoDataCount'] >= max_missing)
                  ,'SubDomain_MultimorbidityCount_Selected_Ungrouped'] = np.nan # Really 'insufficient data'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_NoDataCount'] >= max_missing)
                  ,'SubDomain_MultimorbidityCount_Selected_Grouped'] = nan_fill_text # Really 'insufficient data'

# Group into categorical variable
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] == 0)
                  ,'SubDomain_MultimorbidityCount_Selected_Grouped'] = '0'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] == 1)
                  ,'SubDomain_MultimorbidityCount_Selected_Grouped'] = '1'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] == 2)
                  ,'SubDomain_MultimorbidityCount_Selected_Grouped'] = '2'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] == 3)
                  ,'SubDomain_MultimorbidityCount_Selected_Grouped'] = '3'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] >= 4)
                  ,'SubDomain_MultimorbidityCount_Selected_Grouped'] = '4+'

combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] = combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'].fillna(nan_fill_text)

test = combined_flat[col_list + ['SubDomain_MultimorbidityCount_Selected_Ungrouped','SubDomain_MultimorbidityCount_Selected_Grouped']]



# -----------------------------------------------------------------------------
# Create ordinal variables for ordered categoricals
# E.g. self-rated health, frailty index
# Pre-pandemic self-rated health
for n in range (1,6,1):
    combined_flat.loc[(combined_flat['PrePandemicHealth_Earliest'].str.contains(str(n)))
                    ,'PrePandemicHealth_Earliest_Ordinal'] = n
    
# Frailty index categories
for n in range (1,5,1):
    combined_flat.loc[(combined_flat['FrailtyIndexOngoingCat'].str.contains(str(n)))
                    ,'FrailtyIndexOngoingCat_Ordinal'] = n


# -----------------------------------------------------------------------------
# Do log transformation of frailty index score to make distribution more normal
# Convert to numeric field
combined_flat['FrailtyIndexOngoingScore'] = pd.to_numeric(combined_flat['FrailtyIndexOngoingScore'], errors = 'coerce')

combined_flat['FrailtyIndexOngoingScore_Log'] = np.log(combined_flat['FrailtyIndexOngoingScore'])


# -----------------------------------------------------------------------------
# Thriva 2, 3 vaccinations
data_thriva2_3vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                           & (combined_flat['WeeksSinceVacc_3'] >= 2)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()

number_vaccinations = 3

if number_vaccinations == 1:
    sample_data = data_thriva1_1vacc.copy()
elif number_vaccinations == 3:
    sample_data = data_thriva2_3vacc.copy()

# -----------------------------------------------------------------------------
# Calculate z-score (difference to mean in units of standard deviations) for frailty index score
sample_data['FrailtyIndexOngoingScore_ZScore'] = scipy.stats.zscore(np.array(sample_data['FrailtyIndexOngoingScore']), nan_policy="omit") 

# delete values where frailty score is 0 - gives infinity log value
sample_data.loc[(sample_data['FrailtyIndexOngoingScore'] == 0)
            ,'FrailtyIndexOngoingScore_Log'] = np.nan

sample_data['FrailtyIndexOngoingScore_Log_ZScore'] = scipy.stats.zscore(np.array(sample_data['FrailtyIndexOngoingScore_Log']), nan_policy="omit") 

test = sample_data[['FrailtyIndexOngoingScore','FrailtyIndexOngoingScore_Log','FrailtyIndexOngoingScore_Log_ZScore']]

# Merge z-score to main table
combined_flat = pd.merge(combined_flat, sample_data[['StudyNumber','FrailtyIndexOngoingScore_ZScore','FrailtyIndexOngoingScore_Log_ZScore']], how = 'left', on = 'StudyNumber')




#%% Create dummy variables from un-ordered categoricals
# -----------------------------------------------------------------------------
# List of categorical input variables
var_input_all = [
                 'sex', 
                 'edu_bin_combined', 
                 'Vaccine_3_name_grouped',
                 'NaturalInfection_WideCDC_Interpretation_MaxToDate', 
                 'ShieldingFlag', 
                 'MedicationFlag_ED_Immunosuppressant',
                 'SubDomain_Depression',
                 'SubDomain_RheumatoidArthritis',
                 'SubDomain_MultimorbidityCount_Selected_Grouped'
                 ]

# Create dummy variables
dummy_var_list_full = []
for var in var_input_all:
    combined_flat[var] = combined_flat[var].fillna('NaN') # fill NaN with 'No data' so missing data can be distinguished from 0 results
    cat_list ='var'+'_'+var # variable name
    cat_list = pd.get_dummies(combined_flat[var], prefix=var) # create binary variable of category value
    combined_flat = combined_flat.join(cat_list) # join new column to dataframe
 
combined_flat_col_list = combined_flat.columns.to_list() # save column names 



#%% Calculate twin pair average and difference variables to use as variables in within-between models
# Specify variables to calculate mean and difference to mean for
calc_cols = [
             'PrePandemicHealth_Earliest_Ordinal', 
             'FrailtyIndexOngoingScore_Log_ZScore',

             # Categorical binary variables
             'ShieldingFlag_yes',
             'MedicationFlag_ED_Immunosuppressant_Yes',
             'SubDomain_RheumatoidArthritis_Yes',
             'edu_bin_combined_nvq4/nvq5/degree or equivalent',
             'SubDomain_MultimorbidityCount_Selected_Grouped_3',
             
             
             ]

# loop through all columns used as dependent and independent variables in model
for n in range(0,len(calc_cols),1):
    col = calc_cols[n] 
    
    # Slice to select rows without missing data
    data_slice = combined_flat[(combined_flat[col] != 'NoDataAvailable')]
    data_slice = data_slice[~(data_slice[col].isnull())]
    
    data_slice[col] = data_slice[col].astype(float) # cast data to specified type
    
    # Group by study number to get max value for each individual
    col_grouped_bystudynumber = data_slice.groupby(['FamilyNumber','StudyNumber'])[col].max()
    
    # Group by family number to calculate twin pair average
    col_grouped_byfamily = col_grouped_bystudynumber.groupby('FamilyNumber').mean()
    col_grouped_byfamily = col_grouped_byfamily.rename(col + '_FamilyMean')
    
    # # Group by family number to calculate twin pair average
    # col_grouped = data_slice.groupby('FamilyNumber')[col].mean()
    # col_grouped = col_grouped.rename(col + '_FamilyMean')
    
    # Merge to original dataframe to add twin pair average as additional column
    combined_flat = pd.merge(combined_flat, col_grouped_byfamily, how = 'left', left_on = 'FamilyNumber', right_index = True)
    
    # Calculate individual value - family mean value
    combined_flat[col + '_DiffToMean'] = pd.to_numeric(combined_flat[col], errors='coerce') - combined_flat[col + '_FamilyMean']
    
    
test = combined_flat[['FamilyNumber','StudyNumber',
                      'edu_bin_combined_nvq4/nvq5/degree or equivalent',
                      'edu_bin_combined_nvq4/nvq5/degree or equivalent' + '_DiffToMean',
                      'edu_bin_combined_nvq4/nvq5/degree or equivalent' + '_FamilyMean']]


#%% Specify dependent and independent variables
# -----------------------------------------------------------------------------
# List of control variables that are continuous, and should be converted to float type
control_cols_continuous = ['age',
                           'WeeksSinceVacc_3',]

# List of continuous test variables
cols_continuous = ['PrePandemicHealth_Earliest_Ordinal', 
                 'FrailtyIndexOngoingScore_Log_ZScore',
            
                 ]

# List of categorical variables original names we want to test in models 
cols_categorical  = ['sex',
                    'edu_bin_combined',
                    'Vaccine_3_name_grouped',
                    'NaturalInfection_WideCDC_Interpretation_MaxToDate',
                    'ShieldingFlag',
                    'MedicationFlag_ED_Immunosuppressant',
                    'SubDomain_RheumatoidArthritis',
                    
                    'PrePandemicEmploymentStatus_Earliest',
                    'FrailtyIndexOngoingCat',
                    'SubDomain_MultimorbidityCount_Selected_Grouped',
                    ]

# list of dummy variables representing categorical values to include in models
cols_categorical_dummy = ['sex_M',
                         'edu_bin_combined_nvq4/nvq5/degree or equivalent',                        
                         'Vaccine_3_name_grouped_Moderna',
                         'Vaccine_3_name_grouped_Other',                         
                         'NaturalInfection_WideCDC_Interpretation_MaxToDate_2. Evidence of natural infection',                
                         'ShieldingFlag_yes',
                         'MedicationFlag_ED_Immunosuppressant_Yes',
                         'SubDomain_RheumatoidArthritis_Yes',
                         'SubDomain_MultimorbidityCount_Selected_Grouped_3'
                         
                         ]

# List of categories which correspond to missing data that we want to remove from model sample data
missing_list = ['NoDataAvailable', 
                '0.0 Unknown - individual did not complete CoPE', 
                '1. Possible evidence of natural infection (N negative, S positive, Vaccination status unknown)',
                '0.1 Unknown - Answer not provided in CoPE']


# -----------------------------------------------------------------------------
# Generate sets of control/adjustment variables 
### 3 vaccines
control_cols_vacc3_agesex_vaccinfo_infection_string = 'Adjusted for: age, sex, weeks since vaccination, vaccine received, serology-based infection status'
control_cols_vacc3_agesex_vaccinfo_infection = ['age', 
                'sex_M', 
                'WeeksSinceVacc_3',                         
                'Vaccine_3_name_grouped_Moderna',
                'Vaccine_3_name_grouped_Other', 
                'NaturalInfection_WideCDC_Interpretation_MaxToDate_2. Evidence of natural infection',
                ]

control_cols_list = [
                     [control_cols_vacc3_agesex_vaccinfo_infection,control_cols_vacc3_agesex_vaccinfo_infection_string],
                                          ]

# -----------------------------------------------------------------------------
# Select test variables and run models
# test_col_list = cols_continuous + cols_categorical
test_col_list = [
                'controls_only', # for controls only
             
                'FrailtyIndexOngoingScore_Log_ZScore',
                'PrePandemicHealth_Earliest_Ordinal', 
                 
                  # Categorical binary variables
                  'ShieldingFlag_yes',
                  'MedicationFlag_ED_Immunosuppressant_Yes',
                  'SubDomain_RheumatoidArthritis_Yes',
                  'edu_bin_combined_nvq4/nvq5/degree or equivalent',
                  'SubDomain_MultimorbidityCount_Selected_Grouped_3'
                 ]



#%% Filter data to get analysis sample
# -----------------------------------------------------------------------------
# Thriva 2, 3 vaccinations
data_thriva2_3vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                           & (combined_flat['WeeksSinceVacc_3'] >= 2)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()

sample_data = data_thriva2_3vacc.copy()



#%% Add difference columns for value after filtering for vaccination
# Specify variables to calculate mean and difference to mean for
col = 'Value'

# Slice to select rows without missing data
data_slice = sample_data[(sample_data[col] != 'NoDataAvailable')]
data_slice = data_slice[~(data_slice[col].isnull())]

data_slice[col] = data_slice[col].astype(float) # cast data to specified type

# Group by study number to get max value for each individual
col_grouped_bystudynumber = data_slice.groupby(['FamilyNumber','StudyNumber'])[col].max()

# Group by family number to calculate twin pair average
col_grouped_byfamily = col_grouped_bystudynumber.groupby('FamilyNumber').mean()
col_grouped_byfamily = col_grouped_byfamily.rename(col + '_FamilyMean')

# Merge to original dataframe to add twin pair average as additional column
sample_data = pd.merge(sample_data, col_grouped_byfamily, how = 'left', left_on = 'FamilyNumber', right_index = True)

# Calculate individual value - family mean value
sample_data[col + '_DiffToMean'] = pd.to_numeric(sample_data[col], errors='coerce') - sample_data[col + '_FamilyMean']
    
test = sample_data[['FamilyNumber','StudyNumber',
                      'Value',
                      'Value' + '_DiffToMean',
                      'Value' + '_FamilyMean']]


#%% Setup dependent variables for models
y_col_metric_list = ['Value']
test = sample_data[y_col_metric_list]

#%% Specify parameters and run models
# Create empty lists to fill with results tables
result_OLS_list = []
result_GLME_individual_list = []
result_GLME_withinbetween_list = []

result_OLS_filtered_list = []
result_GLME_individual_filtered_list = []
result_GLME_withinbetween_filtered_list = []

# Specify which metric is used as outcome variable
y_col_metric = y_col_metric_list[0]

# Specify which control variable set to use
control_cols = control_cols_list[0][0]
control_cols_string = control_cols_list[0][1]
if number_vaccinations == 1:
    control_cols_idx_list = [0,1,2]
elif number_vaccinations == 3:
    control_cols_idx_list = [3,4,5]

# Filter for zygosity
zygosity_filter_list = [['MZ', 'DZ', 'NoDataAvailable', 'UZ'], ['MZ'], ['DZ']]
zygosity_filter_list_string = ['No filter', 'MZ', 'DZ']


# Loop through control variable combinations
for n in range(0,len(control_cols_idx_list),1):
    control_cols_idx = control_cols_idx_list[n]
    control_cols = control_cols_list[control_cols_idx][0]
    control_cols_string = control_cols_list[control_cols_idx][1]
    
    # Loop through zygosity filter
    for zygosity in range(0,len(zygosity_filter_list),1):
        zygosity_list = zygosity_filter_list[zygosity]
        zygosity_string = zygosity_filter_list_string[zygosity]
    
    
        # Loop through outcome variable list
        for y_col_metric in y_col_metric_list:
            
            # Loop through test column list
            for col in test_col_list:
                print('testing: ' + col)
                #%% Data preparation
                # Take a copy of data before applying filters
                sample_data_filtered = sample_data.copy()
                
                # filter for selected zygosity
                sample_data_filtered = sample_data_filtered[(sample_data_filtered['ACTUAL_ZYGOSITY'].isin(zygosity_list))]
                
                
                # Drop rows with missing values from control columns
                control_cols_copy = control_cols.copy()
                for control_col in control_cols_copy:
                    # print('testing: ' + control_col)
                    sample_data_filtered = sample_data_filtered[~(sample_data_filtered[control_col].isin(missing_list))]
                    sample_data_filtered = sample_data_filtered[~(sample_data_filtered[control_col].isnull())]
                
                    if control_col in control_cols_continuous:
                        # cast continuous data to float type
                        sample_data_filtered[control_col] = sample_data_filtered[control_col].astype(float)
                
                    
                if col == 'controls_only':
                    test_cols = []
                    x_cols = control_cols
                
                elif col in control_cols: # if test var is in control cols, just run controls and ignore results
                    test_cols = []
                    x_cols = control_cols
                
                else:
                    # -------------------------------------------------------------------------
                    # Drop rows with missing values in test column
                    sample_data_filtered = sample_data_filtered[~(sample_data_filtered[col].isin(missing_list))]
                    sample_data_filtered = sample_data_filtered[~(sample_data_filtered[col].isnull())]
                    
                    # -------------------------------------------------------------------------
                    # Add matching dummy variables to test column list 
                    test_cols = []
                    if col in cols_categorical:    
                        for col_dummy in cols_categorical_dummy:
                            if col in col_dummy:
                                test_cols.append(col_dummy) 
                        
                    elif col not in cols_categorical:
                        test_cols.append(col) 
                    
                        # cast continuous data to float type
                        sample_data_filtered[col] = sample_data_filtered[col].astype(float)
                    
                    # Final list of input variables
                    x_cols = control_cols + test_cols
                
                
                
                
                # -------------------------------------------------------------------------
                # Specify dependent variable
                y_col = y_col_metric  
                
                # Drop rows with missing values in outcome dependent variable column
                sample_data_filtered = sample_data_filtered[~(sample_data_filtered[y_col].isin(missing_list))]
                sample_data_filtered = sample_data_filtered[~(sample_data_filtered[y_col].isnull())]
                y_data = sample_data_filtered[y_col]
            
                
                # Final check - drop any columns from input variable list where all values = 0 - will get singular matrix otherwise
                x_cols_copy = x_cols.copy()
                for x_col in x_cols_copy:
                    # print(x_cols)
                    if (sample_data_filtered[x_col].sum() == 0):
                        # print('removing: ' + x_col)
                        x_cols.remove(x_col)
            
            
            
                #%% 1. Generalised linear fixed effect model - treat all individuals separately
                # Example - Dependent variable, y, is 'Value' column - anti-S value, Independent variables: Age, Sex
                x_data = sample_data_filtered[x_cols]
                x_data = sm.add_constant(x_data)
                
                model = sm.OLS(y_data,x_data)
                results = model.fit()
                model_summary = results.summary()
                # print(model_summary)
                
                result_OLS = results_summary_to_dataframe(results,2)
                
                # Add extra details
                result_OLS['TestVar'] = col
                result_OLS['NumVacc'] = number_vaccinations
                result_OLS['Outcome'] = y_col_metric
                result_OLS['ControlVar'] = control_cols_string
                result_OLS['Zygosity'] = zygosity_string
                result_OLS['SampleSize'] = sample_data_filtered.shape[0]
                
                # filter for test var only
                if (col == 'controls_only'):
                    result_OLS_filtered = result_OLS[~(result_OLS.index.str.contains('const'))]
                else:
                    result_OLS_filtered = result_OLS[(result_OLS.index.str.contains(result_OLS['TestVar'].max()))]
                
                # Append to list
                result_OLS_list.append(result_OLS)
                result_OLS_filtered_list.append(result_OLS_filtered)
                
                
                #%% Count number of twins with data for each family and use to exclude unpaired twins for models that include FamilyNumber as random effect
                StudyNumber_count = sample_data_filtered.groupby('FamilyNumber')['StudyNumber'].count()
                StudyNumber_count = StudyNumber_count.rename('StudyNumber_FamilyCount')
                
                # Join count to main dataframe
                sample_data_filtered = pd.merge(sample_data_filtered, StudyNumber_count, how = 'left', left_on = 'FamilyNumber', right_index = True)
                
                sample_data_filtered = sample_data_filtered[(sample_data_filtered['StudyNumber_FamilyCount'] >= 2)]
                
                
                # Drop any columns from input variable list where all values = 0 - will get singular matrix otherwise
                x_cols_copy = x_cols.copy()
                for x_col in x_cols_copy:
                    # print(x_cols)
                    if (sample_data_filtered[x_col].sum() == 0):
                        # print('removing: ' + x_col)
                        x_cols.remove(x_col)
                
                
                #%% 2. Generalised linear mixed effects model - cluster based on family
                # Include family number as a random effect - allow intercept (and slope if converges) to vary for each family
                # Generate equation from x_col list
                formula = y_col + ' ~ '  
                for n in range(0,len(x_cols),1):
                    formula_col = x_cols[n]
                    if n < len(x_cols)-1:
                        formula = formula + formula_col + ' + '
                    elif n == len(x_cols)-1:
                        formula = formula + formula_col
                
                # Random intercept only
                model = sm.MixedLM(endog = sample_data_filtered[y_col], exog = sample_data_filtered[x_cols], groups = sample_data_filtered['FamilyNumber']).fit()
                
                model_summary = model.summary()
                # print(model_summary)
                
                result_GLME_individual = results_summary_to_dataframe(model,2)
                # Add extra details
                result_GLME_individual['TestVar'] = col
                result_GLME_individual['NumVacc'] = number_vaccinations
                result_GLME_individual['Outcome'] = y_col_metric
                result_GLME_individual['ControlVar'] = control_cols_string
                result_GLME_individual['Zygosity'] = zygosity_string
                result_GLME_individual['SampleSize'] = sample_data_filtered.shape[0]
                
                # filter for test var only
                if (col == 'controls_only'):
                    result_GLME_individual_filtered = result_GLME_individual[~(result_GLME_individual.index.str.contains('Group Var'))]
                else:
                    result_GLME_individual_filtered = result_GLME_individual[(result_GLME_individual.index.str.contains(result_GLME_individual['TestVar'].max()))]
                
                # Append to list
                result_GLME_individual_list.append(result_GLME_individual)
                result_GLME_individual_filtered_list.append(result_GLME_individual_filtered)
                
                
                #%% 3. Generalised linear mixed effects model, including within and between pair coefficents - to better understand origin of effects
                # specify independent variables - changing test column to within and between pair variables where desired
                test_cols_within_between = []
                for cols in test_cols:
                    if cols in calc_cols:
                        test_cols_within_between.append(cols + '_FamilyMean')
                        
                        # Choose what to use as the variable to test 'within pair' variation - either the regular variable 'as is', or the 'difference to pair mean'. Both options should give same coefficients for controls and within coefficient, but will affect the coefficient of the 'between' pair variable. 'As is' is suggested to be better as can interpret both individual and pair mean together. Explained here https://statisticalhorizons.com/between-within-contextual-effects/
                        if within_var == 'DiffToMean':
                            test_cols_within_between.append(cols + '_DiffToMean')
                        elif within_var == 'AsIs':
                            test_cols_within_between.append(cols)    
                        
                        x_cols = control_cols + test_cols_within_between
                        
                    elif cols not in calc_cols:
                        x_cols = control_cols + test_cols
                                
                # Drop any columns from input variable list where all values = 0 - will get singular matrix otherwise
                x_cols_copy = x_cols.copy()
                for x_col in x_cols_copy:
                    # print(x_cols)
                    if (len(sample_data_filtered[(sample_data_filtered[x_col] > 0)][x_col]) == 0):
                        # print('removing: ' + x_col)
                        x_cols.remove(x_col)
                
                
                # Generate equation from x_col list
                formula = y_col + ' ~ '  
                for n in range(0,len(x_cols),1):
                    formula_col = x_cols[n]
                    if n < len(x_cols)-1:
                        formula = formula + formula_col + ' + '
                    elif n == len(x_cols)-1:
                        formula = formula + formula_col
                
                # Random intercept only
                model = sm.MixedLM(endog = sample_data_filtered[y_col], exog = sample_data_filtered[x_cols], groups = sample_data_filtered['FamilyNumber']).fit()
                
                # Wald test to compare difference between family mean (between) and diff to family mean (within) coefficients
                print(cols + ', ' + zygosity_string + ', ' + y_col_metric)
                print(control_cols_string)
                print(str(cols + '_FamilyMean = ' + cols + '_DiffToMean'))
                print(model.wald_test(str(cols + '_FamilyMean = ' + cols + '_DiffToMean')))
                
                model_summary = model.summary()
                # print(model_summary)
                
                result_GLME_withinbetween = results_summary_to_dataframe(model,2)
                # Add extra details
                result_GLME_withinbetween['TestVar'] = col
                result_GLME_withinbetween['NumVacc'] = number_vaccinations
                result_GLME_withinbetween['Outcome'] = y_col_metric
                result_GLME_withinbetween['ControlVar'] = control_cols_string
                result_GLME_withinbetween['Zygosity'] = zygosity_string
                result_GLME_withinbetween['SampleSize'] = sample_data_filtered.shape[0]
                
                # filter for test var only
                if (col == 'controls_only'):
                    result_GLME_withinbetween_filtered = result_GLME_withinbetween[~(result_GLME_withinbetween.index.str.contains('Group Var'))]
                else:
                    result_GLME_withinbetween_filtered = result_GLME_withinbetween[(result_GLME_withinbetween.index.str.contains(result_GLME_withinbetween['TestVar'].max()))]
                
                # Append to list
                result_GLME_withinbetween_list.append(result_GLME_withinbetween)
                result_GLME_withinbetween_filtered_list.append(result_GLME_withinbetween_filtered)


#%% Append tables together
# -----------------------------------------------------------------------------
### GLFE as individuals - Generalised linear fixed effect model
# full
master_result_OLS = result_OLS_list[0]
for i in range(1,(len(result_OLS_list))):
    master_result_OLS = master_result_OLS.append(result_OLS_list[i])

# filtered for test variables only
master_result_OLS_filtered = result_OLS_filtered_list[0]
for i in range(1,(len(result_OLS_filtered_list))):
    master_result_OLS_filtered = master_result_OLS_filtered.append(result_OLS_filtered_list[i])

master_result_OLS['Model'] = 'OLS - as individuals'
master_result_OLS_filtered['Model'] = 'OLS - as individuals'
    
# -----------------------------------------------------------------------------
### GLME as individuals
# full
master_result_GLME_individual = result_GLME_individual_list[0]
for i in range(1,(len(result_GLME_individual_list))):
    master_result_GLME_individual = master_result_GLME_individual.append(result_GLME_individual_list[i])

# filtered for test variables only
master_result_GLME_individual_filtered = result_GLME_individual_filtered_list[0]
for i in range(1,(len(result_GLME_individual_filtered_list))):
    master_result_GLME_individual_filtered = master_result_GLME_individual_filtered.append(result_GLME_individual_filtered_list[i])

master_result_GLME_individual['Model'] = 'GLME - as individuals'
master_result_GLME_individual_filtered['Model'] = 'GLME - as individuals'

# -----------------------------------------------------------------------------
### GLME within-between
# full
master_result_GLME_withinbetween = result_GLME_withinbetween_list[0]
for i in range(1,(len(result_GLME_withinbetween_list))):
    master_result_GLME_withinbetween = master_result_GLME_withinbetween.append(result_GLME_withinbetween_list[i])

# filtered for test variables only
master_result_GLME_withinbetween_filtered = result_GLME_withinbetween_filtered_list[0]
for i in range(1,(len(result_GLME_withinbetween_filtered_list))):
    master_result_GLME_withinbetween_filtered = master_result_GLME_withinbetween_filtered.append(result_GLME_withinbetween_filtered_list[i])

master_result_GLME_withinbetween['Model'] = 'GLME - within-between'
master_result_GLME_withinbetween_filtered['Model'] = 'GLME - within-between'

# Combine GLME unfiltered tables
master_result_combined = master_result_OLS.append(master_result_GLME_individual)
master_result_combined = master_result_combined.append(master_result_GLME_withinbetween)

# Combine GLME filtered tables
master_result_filtered_combined = master_result_OLS_filtered.append(master_result_GLME_individual_filtered)
master_result_filtered_combined = master_result_filtered_combined.append(master_result_GLME_withinbetween_filtered)

xxx


#%% Export results
if export_csv == 1:
    master_result_filtered_combined.to_csv('TwinsCovid_GLMER_Vacc3.csv')