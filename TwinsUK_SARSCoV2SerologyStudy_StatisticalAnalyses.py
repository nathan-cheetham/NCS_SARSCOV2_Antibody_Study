# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 11:34:06 2021

@author: k2143494, Nathan Cheetham, Department of Twin Research, King's College London, UK, nathan.cheetham@kcl.ac.uk

Logistic regression and other statistical analyses to test associations between antibody levels following SARS-CoV-2 vaccination and various factors within TwinsUK cohort https://doi.org/10.7554/eLife.80428
"""

#%% Import packages
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import mannwhitneyu
import pymannkendall as mk
import matplotlib.pyplot as plt 
plt.rc("font", size=12)
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
import math
import statsmodels.stats.api as sms
import statsmodels.api as sm

import seaborn as sns
import dexplot as dxp
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from datetime import datetime
from pySankey.sankey import sankey
import matplotlib as mpl
mpl.rc('hatch', color='k', linewidth=1.5)

do_analysis_logreg = 0
do_postvaccinf = 1

export_csv = 0

#%% Define functions
# -----------------------------------------------------------------------------
def gen_dummy_var_list(var_list, dummy_var_list):
    """ Function to identify dummy variables from list of variable names, based on partial string match """
    matching_dummy_var_list = [] 
    for var_name in var_list: # for each variable 
        # generate list of dummy variables
        for var_dummy in dummy_var_list: 
            if (var_name in var_dummy) is True:
                matching_dummy_var_list.append(var_dummy)  
    return matching_dummy_var_list

# -----------------------------------------------------------------------------
def sm_logreg_summary(x_data, y_data, CI_alpha, plot_roc, do_robust_se, cluster_df, cluster_col):
    """Function to generate summary table of variables and fitted coefficients and odds ratios for logistic regression using STATS MODELS"""
    # add cluster label column to x data set if including cluster correlations in covariance calculation
    if do_robust_se == 'cluster':
        x_data[cluster_col] = cluster_df[cluster_col] 
        
    train_data = 0
    if train_data == 1:
        # Split data into test and train set
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, stratify = y_data, test_size = 0.25, random_state = 0)
    elif train_data != 1:
        # Perform logistic regression without splitting - i.e. use whole dataset - performance metrics are not meaningful
        x_train = x_data
        x_test = x_data
        y_train = y_data
        y_test = y_data
    
    # add constant - default for sklearn but not statsmodels
    x_train = sm.add_constant(x_train) 
    x_test = sm.add_constant(x_test)
    
    # Include cluster correlation due to Twin pairs in the model (calculates a different covariance matrix and produces different standard errors - 'robust') https://www.vincentgregoire.com/standard-errors-in-python/
    max_iterations = 20000
    solver_method = 'bfgs'
    model = sm.Logit(y_train, x_train, use_t = True) # set up model
    
    if do_robust_se == 'cluster':
        model_fit = model.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='cluster',
                          cov_kwds={'groups': np.asarray(x_train[cluster_col])},
                          use_t=True)
    
    elif do_robust_se == 'HC3':
        model_fit = model.fit(maxiter = max_iterations, 
                          method = solver_method,  
                          cov_type='HC3',
                          use_t=True)
    else:
        model_fit = model.fit(maxiter = max_iterations, 
                              method = solver_method, 
                              use_t=True) # fit model
       
    logreg_sm_summary1 = model_fit.summary()
    logreg_sm_summary2 = model_fit.summary2()
    logreg_sm_score = model.score(model_fit.params)
    
    # Evaluate model performance
    y_pred_proba = model_fit.predict(x_test) # generate prediction probabilities for test data
    y_pred = y_pred_proba.round() # round to get binary classification
    fit_conf_matrix = metrics.confusion_matrix(y_test, y_pred) # generate confusion matrix to evaluate model predictions
    fit_accuracy = metrics.accuracy_score(y_test, y_pred)
    fit_precision = metrics.precision_score(y_test, y_pred)
    fit_recall = metrics.recall_score(y_test, y_pred)
    
    # Receiver operating characteristic (ROC) curve (Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity.)
    fpr, tpr, thresholds = metrics.roc_curve(y_true = y_test, y_score = y_pred_proba, drop_intermediate = False)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    if plot_roc == 1:
        fig1 = plt.figure()
        plt.plot(fpr,tpr,'--.',label="data 1, AUC="+str(auc))
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve: Logistic Regression Model (stats models)')
        plt.show()
    
    fit_metrics_summary = [auc,y_pred,fit_conf_matrix,fit_accuracy,fit_precision,fit_precision,fit_recall]
     
    # Extract coefficients and convert to Odds Ratios
    sm_coeff = model_fit.params
    sm_se = model_fit.bse
    sm_pvalue = model_fit.pvalues
    sm_coeff_CI = model_fit.conf_int(alpha=CI_alpha)
    sm_OR = np.exp(sm_coeff)
    sm_OR_CI = np.exp(sm_coeff_CI)
    
    # Summarise fitted parameters in table
    sm_summary = pd.DataFrame({'Variable': sm_coeff.index
                                , 'Coefficients': sm_coeff
                                , 'Standard Error': sm_se
                                , 'P-value': sm_pvalue
                                , 'Coefficient C.I. (lower)': sm_coeff_CI[0]
                                , 'Coefficient C.I. (upper)': sm_coeff_CI[1]
                                , 'Odds ratio': sm_OR
                                , 'OR C.I. (lower)': sm_OR_CI[0]
                                , 'OR C.I. (upper)': sm_OR_CI[1]
                                , 'OR C.I. error (lower)': np.abs(sm_OR - sm_OR_CI[0])
                                , 'OR C.I. error (upper)': np.abs(sm_OR - sm_OR_CI[1])})
    
    sm_summary = sm_summary.reset_index(drop = True)
      
    # Add number of observations for given variable in input and outcome datasets
    x_train_count = x_train.sum()
    x_train_count.name = "x_train count"
    sm_summary = pd.merge(sm_summary,x_train_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # join x_train and y_train
    x_y_train = x_train.copy()
    x_y_train['y_train'] = y_train
    # Count observation where y_train = 1
    y_train_count = x_y_train[x_y_train['y_train'] == 1].sum()
    y_train_count.name = "y_train = 1 count"
    sm_summary = pd.merge(sm_summary,y_train_count, how = 'left', left_on = 'Variable', right_index = True)
    
    # Highlight variables where confidence intervals are both below 1 or both above 1
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR > 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR > 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] > 1.0)
                        & (sm_summary['OR C.I. (upper)'] > 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR > 1), ***, p < 0.001'
    
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.05)
                        ,'Significance'] = 'Significant (OR < 1), *, p < 0.05'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.01)
                        ,'Significance'] = 'Significant (OR < 1), **, p < 0.01'
    sm_summary.loc[(sm_summary['OR C.I. (lower)'] < 1.0)
                        & (sm_summary['OR C.I. (upper)'] < 1.0)
                        & (sm_summary['P-value'] < 0.001)
                        ,'Significance'] = 'Significant (OR < 1), ***, p < 0.001'
        
    return sm_summary, fit_metrics_summary



#%% Load data 
# TwinsUK various COVID study results combined with demographics and health variables
combined_flat = pd.read_csv(r"TwinsCovid_CoPE_Antibody_Antigen_flat.csv")
col_list = combined_flat.columns.to_list() # save column names


#%% Processing covariates
#------------------------------------------------------------------------------
# Convert age to numeric field - 'NoDataAvailable' to NaN
combined_flat['age'] = pd.to_numeric(combined_flat['age'], errors = 'coerce')

#------------------------------------------------------------------------------
# Calculate days between CoPE 5 response and Thriva 2 sampling
combined_flat.loc[(combined_flat['StudyName'] == 'Thriva #2')
                  , 'DaysBetween_Thriva2_CoPE5'] = (pd.to_datetime(combined_flat['ItemDate'], errors = 'coerce') - pd.to_datetime(combined_flat['ItemDate_Vaccine_Status_CoPE5'], errors = 'coerce')).dt.days


# -----------------------------------------------------------------------------
# Add number of weeks since 1st and 2nd vaccination
combined_flat['WeeksSinceVacc_1'] = (combined_flat['DaysSinceVacc_1']/7).apply(np.floor) # round down
combined_flat['WeeksSinceVacc_2'] = (combined_flat['DaysSinceVacc_2']/7).apply(np.floor) # round down
combined_flat['WeeksSinceVacc_3'] = (combined_flat['DaysSinceVacc_3']/7).apply(np.floor) # round down

# Add number of weeks since 1st and 2nd vaccination
combined_flat['TwoWeeksSinceVacc_2'] = (combined_flat['DaysSinceVacc_2']/14).apply(np.floor) # round down

# -----------------------------------------------------------------------------
# add basic grouping of Thriva #1 results - 0-0.8 neg, 0.8-250, 250
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 0.0)
                  & (combined_flat['Value'] < 250.0)
                  ,'ValueGrouped_Binary'] =  '1. 0.4-250 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 250.0)
                  ,'ValueGrouped_Binary'] =  '2. 250+ U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'].isnull())
                  ,'ValueGrouped_Binary'] =  '0. void'

# add basic grouping of Thriva #1 results - 0-0.8 neg, 0.8-250, 250
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 0.0)
                  & (combined_flat['Value'] < 0.8)
                  ,'ValueGrouped'] =  '1. 0.4-0.8 U/mL (negative)'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 0.8)
                  & (combined_flat['Value'] < 250.0)
                  ,'ValueGrouped'] =  '2. 0.8-250 U/mL (positive)'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 250.0)
                  ,'ValueGrouped'] =  '3. 250+ U/mL (positive, capped)'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'].isnull())
                  ,'ValueGrouped'] =  '0. void'

# add more detailed grouping of Thriva results - groups of 50
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 0.0)
                  & (combined_flat['Value'] < 0.8)
                  ,'ValueGrouped_50s'] =  '1. 0.4-0.8 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 0.8)
                  & (combined_flat['Value'] < 25.0)
                  ,'ValueGrouped_50s'] =  '2. 0.8-25 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 25)
                  & (combined_flat['Value'] < 50.0)
                  ,'ValueGrouped_50s'] =  '2.2 25-50 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 50)
                  & (combined_flat['Value'] < 100.0)
                  ,'ValueGrouped_50s'] =  '3. 50-100 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 100)
                  & (combined_flat['Value'] < 150.0)
                  ,'ValueGrouped_50s'] =  '4. 100-150 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 150)
                  & (combined_flat['Value'] < 200.0)
                  ,'ValueGrouped_50s'] =  '5. 150-200 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 200)
                  & (combined_flat['Value'] < 250.0)
                  ,'ValueGrouped_50s'] =  '6. 200-250 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 250.0)
                  ,'ValueGrouped_50s'] =  '7. 250+ U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'].isnull())
                  ,'ValueGrouped_50s'] =  '0. void'

# add more detailed grouping of Thriva results - groups of 100
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 0.0)
                  & (combined_flat['Value'] < 50.0)
                  ,'ValueGrouped_100s'] =  '1. 0.4-50 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 50)
                  & (combined_flat['Value'] < 150.0)
                  ,'ValueGrouped_100s'] =  '2. 50-150 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 150)
                  & (combined_flat['Value'] < 250.0)
                  ,'ValueGrouped_100s'] =  '3. 150-250 U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'] >= 250.0)
                  ,'ValueGrouped_100s'] =  '4. 250+ U/mL'
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & (combined_flat['Value'].isnull())
                  ,'ValueGrouped_100s'] =  '0. void'

# Divide value by 100 to make easier to interpret
combined_flat.loc[(combined_flat['DataItem'] == 'Antibody_S')
                  & (combined_flat['StudyName'] == 'Thriva')
                  & ~(combined_flat['Value'].isnull())
                  ,'Value_DividedBy100'] = combined_flat['Value'].div(100) 

# -----------------------------------------------------------------------------
# Generate percentile numbers of antibody level
# Thriva 1, 1 vaccination
data_slice = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                           & (combined_flat['WeeksSinceVacc_1'] >= 4)
                           & (combined_flat['WeeksSinceVacc_1'] <= 11)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()
data_slice['Value_decile_thriva1vacc1'] = pd.qcut(data_slice['Value'], 10, labels = False, duplicates = 'drop') + 1
data_slice['Value_quintile_thriva1vacc1'] = pd.qcut(data_slice['Value'], 5, labels = False, duplicates = 'drop') + 1
data_slice['Value_percentile_thriva1vacc1'] = pd.qcut(data_slice['Value'], 100, labels = False, duplicates = 'drop') + 1
# merge 
combined_flat = pd.merge(combined_flat, data_slice[['StudyNumber','Value_decile_thriva1vacc1','Value_quintile_thriva1vacc1','Value_percentile_thriva1vacc1']], how = 'left', on = 'StudyNumber')

# Thriva 1, 2 vaccinations
data_slice = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                           & (combined_flat['WeeksSinceVacc_2'] >= 2)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()
data_slice['Value_decile_thriva1vacc2'] = pd.qcut(data_slice['Value'], 10, labels = False, duplicates = 'drop') + 1
data_slice['Value_quintile_thriva1vacc2'] = pd.qcut(data_slice['Value'], 5, labels = False, duplicates = 'drop') + 1
data_slice['Value_percentile_thriva1vacc2'] = pd.qcut(data_slice['Value'], 100, labels = False, duplicates = 'drop') + 1
# merge 
combined_flat = pd.merge(combined_flat, data_slice[['StudyNumber','Value_decile_thriva1vacc2','Value_quintile_thriva1vacc2','Value_percentile_thriva1vacc2']], how = 'left', on = 'StudyNumber')

# Thriva 2, 2 vaccinations
data_slice = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                           & (combined_flat['Result_Vaccine_Status_CoPE5'] == '2 vaccines received')
                           & (combined_flat['WeeksSinceVacc_2'] >= 2)
                           & (combined_flat['age'] >= 18)
                           & (combined_flat['DaysBetween_Thriva2_CoPE5'] <= 7) # sampled for thriva 2 no more than 7 days after completing cope5, in case of vaccination shortly after cope5
                           ].copy().reset_index()
data_slice['Value_decile_thriva2vacc2'] = pd.qcut(data_slice['Value'], 10, labels = False, duplicates = 'drop') + 1
data_slice['Value_quintile_thriva2vacc2'] = pd.qcut(data_slice['Value'], 5, labels = False, duplicates = 'drop') + 1
data_slice['Value_percentile_thriva2vacc2'] = pd.qcut(data_slice['Value'], 100, labels = False, duplicates = 'drop') + 1
# merge 
combined_flat = pd.merge(combined_flat, data_slice[['StudyNumber','Value_decile_thriva2vacc2','Value_quintile_thriva2vacc2','Value_percentile_thriva2vacc2']], how = 'left', on = 'StudyNumber')

# Thriva 2, 3 vaccinations
data_slice = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                           & (combined_flat['WeeksSinceVacc_3'] >= 2)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()
data_slice['Value_decile_thriva2vacc3'] = pd.qcut(data_slice['Value'], 10, labels = False, duplicates = 'drop') + 1
data_slice['Value_quintile_thriva2vacc3'] = pd.qcut(data_slice['Value'], 5, labels = False, duplicates = 'drop') + 1
data_slice['Value_percentile_thriva2vacc3'] = pd.qcut(data_slice['Value'], 100, labels = False, duplicates = 'drop') + 1
# merge 
combined_flat = pd.merge(combined_flat, data_slice[['StudyNumber','Value_decile_thriva2vacc3','Value_quintile_thriva2vacc3','Value_percentile_thriva2vacc3']], how = 'left', on = 'StudyNumber')





#%% Identification of post-vaccination infection
# -----------------------------------------------------------------------------
# Identify individuals with probable infection as before/after Thriva #1 or Thriva #2 (based on self-reported symptoms, antigen tests and lab antibody tests)
col_list = combined_flat.columns.to_list()

# Generate columns with Thriva #1 and #2 participation dates
# Thriva 1
combined_flat_slice = combined_flat[(combined_flat['StudyName'] == 'Thriva') 
                                    & (combined_flat['DataItem'] == 'Antibody_S')]
columns = ['StudyNumber', 'DataItem','ItemDate']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).last()
max_status = max_status.rename(columns = {'ItemDate':'ItemDate_Thriva1'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['ItemDate_Thriva1'], how = 'left', left_on = 'StudyNumber', right_index=True)

# Thriva 2
combined_flat_slice = combined_flat[(combined_flat['StudyName'] == 'Thriva #2') 
                                    & (combined_flat['DataItem'] == 'Antibody_S')]
columns = ['StudyNumber', 'DataItem','ItemDate']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).last()
max_status = max_status.rename(columns = {'ItemDate':'ItemDate_Thriva2'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['ItemDate_Thriva2'], how = 'left', left_on = 'StudyNumber', right_index=True)



# -----------------------------------------------------------------------------
# Identify individuals with probable post-vaccination infection (based on self-reported symptoms, antigen tests and lab antibody tests)
col_list = combined_flat.columns.to_list()

# Work with a copy of dataset 
combined_flat_slice = combined_flat.copy()

#### Identify controls - set values as defaults, then replace in cases where individuals meet infection criteria
# Minimum: People who participated in Thriva 1 AND EITHER Thriva 2 OR CoPE 5. Without these, people won't have had the opportunity to record any infection happening after Thriva 1
# Flag for post-vaccination infection but unknown exact vaccination status
combined_flat_slice.loc[~(combined_flat_slice['ItemDate_Thriva1'].isnull())
                        & (~(combined_flat_slice['ItemDate_Thriva2'].isnull())
                         | ~(combined_flat_slice['ItemDate_Vaccine_Status_CoPE5'].isnull()))
                        , 'PostVaccInfection_UnknownVaccine_Flag'] = '0.1 No infection recorded'
# Flag for post-vaccination infection whilst once vaccinated
combined_flat_slice.loc[~(combined_flat_slice['ItemDate_Thriva1'].isnull())
                        & (~(combined_flat_slice['ItemDate_Thriva2'].isnull())
                         | ~(combined_flat_slice['ItemDate_Vaccine_Status_CoPE5'].isnull()))
                        , 'PostVaccInfection_1Vaccine_Flag'] = '0.1 No infection recorded'
# Flag for post-vaccination infection whilst twice vaccinated
combined_flat_slice.loc[~(combined_flat_slice['ItemDate_Thriva1'].isnull())
                        & (~(combined_flat_slice['ItemDate_Thriva2'].isnull())
                         | ~(combined_flat_slice['ItemDate_Vaccine_Status_CoPE5'].isnull()))
                        , 'PostVaccInfection_2Vaccine_Flag'] = '0.1 No infection recorded'
# Flag for post-vaccination infection whilst 3 times vaccinated
combined_flat_slice.loc[~(combined_flat_slice['ItemDate_Thriva1'].isnull())
                        & (~(combined_flat_slice['ItemDate_Thriva2'].isnull())
                         | ~(combined_flat_slice['ItemDate_Vaccine_Status_CoPE5'].isnull()))
                        , 'PostVaccInfection_3Vaccine_Flag'] = '0.1 No infection recorded'

# Fill in missing values of flag columns
combined_flat_slice['PostVaccInfection_UnknownVaccine_Flag'] = combined_flat_slice['PostVaccInfection_UnknownVaccine_Flag'].fillna('0.0 Insufficient data')
combined_flat_slice['PostVaccInfection_1Vaccine_Flag'] = combined_flat_slice['PostVaccInfection_1Vaccine_Flag'].fillna('0.0 Insufficient data')
combined_flat_slice['PostVaccInfection_2Vaccine_Flag'] = combined_flat_slice['PostVaccInfection_2Vaccine_Flag'].fillna('0.0 Insufficient data')
combined_flat_slice['PostVaccInfection_3Vaccine_Flag'] = combined_flat_slice['PostVaccInfection_3Vaccine_Flag'].fillna('0.0 Insufficient data')


#### Identify Cases - identify post-vaccination infections
#### 0. Where NaturalInfection_WideCDC_Interpretation_MaxToDate at time of Thriva #1 is no evidence, but later changed to evidence (at Jan-22, only way this could happen is from positive anti-N antibody at Thriva 2), & vaccination status at time of Thriva #1 is at least once vaccinated
combined_flat_slice.loc[(combined_flat_slice['StudyName'] == 'Thriva')
                        & (combined_flat_slice['NaturalInfection_WideCDC_Interpretation_MaxToDate'] == '0. No evidence of natural infection')
                        & (combined_flat_slice['NaturalInfection_WideCDC_Interpretation_Final'] == '2. Evidence of natural infection') # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'].isin(['2.1 Vaccinated once', '2.2 Vaccinated twice', '2.3 Vaccinated 3 times']))
                        , 'PostVaccInfection_UnknownVaccine_Flag'] = '1.0 Change in serology based natural infection status after Thriva #1 whilst vaccinated, but date of infection unknown (after Thriva #1)'


#### 1. Where self-report symptom start date is after vaccination date
# Infection whilst vaccination #1
# First symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                        , 'PostVaccInfection_1Vaccine_Flag'] = '1.1 Self-reported 1st symptoms whilst once vaccinated (without antigen or antibody confirmation)'
# Date
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                        , 'PostVaccInfection_1Vaccine_Symptom_1_Date'] = combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']
# 2nd symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                  , 'PostVaccInfection_1Vaccine_Flag'] = '1.2 Self-reported 2nd symptoms whilst once vaccinated (without antigen or antibody confirmation)'
# Date
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                        , 'PostVaccInfection_1Vaccine_Symptom_2_Date'] = combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']


# Infection whilst vaccination #2
# First symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                        , 'PostVaccInfection_2Vaccine_Flag'] = '1.1 Self-reported 1st symptoms whilst twice vaccinated (without antigen or antibody confirmation)'
# Date
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                        , 'PostVaccInfection_2Vaccine_Symptom_1_Date'] = combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']
# 2nd symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                  , 'PostVaccInfection_2Vaccine_Flag'] = '1.2 Self-reported 2nd symptoms whilst twice vaccinated (without antigen or antibody confirmation)'
# Date
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                        , 'PostVaccInfection_2Vaccine_Symptom_2_Date'] = combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']

# Infection whilst vaccination #3
# First symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                        , 'PostVaccInfection_3Vaccine_Flag'] = '1.1 Self-reported 1st symptoms whilst 3 times vaccinated (without antigen or antibody confirmation)'
# Date
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                        , 'PostVaccInfection_3Vaccine_Symptom_1_Date'] = combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']
# 2nd symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                  , 'PostVaccInfection_3Vaccine_Flag'] = '1.2 Self-reported 2nd symptoms whilst 3 times vaccinated (without antigen or antibody confirmation)'
# Date
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                        , 'PostVaccInfection_3Vaccine_Symptom_2_Date'] = combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']



#### 2. Where self-report symptom start date is after vaccination date AND first positive anti-N lab test is also after symptom start date, giving a high probability of confirming that the infection was after vaccination
# Infection whilst vaccination #1
# First symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                        & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                        , 'PostVaccInfection_1Vaccine_Flag'] = '2.1 Self-reported 1st symptoms whilst once vaccinated (+ first positive anti-N after symptom start)'
# Date already collected under 1.

# 2nd symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                        & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                  , 'PostVaccInfection_1Vaccine_Flag'] = '2.2 Self-reported 2nd symptoms whilst once vaccinated (+ first positive anti-N after symptom start)'

# Infection whilst vaccination #2
# First symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                        & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                        , 'PostVaccInfection_2Vaccine_Flag'] = '2.1 Self-reported 1st symptoms whilst twice vaccinated (+ first positive anti-N after symptom start)'
# 2nd symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                        & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                  , 'PostVaccInfection_2Vaccine_Flag'] = '2.2 Self-reported 2nd symptoms whilst twice vaccinated (+ first positive anti-N after symptom start)'


# Infection whilst vaccination #3
# First symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                        & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_1_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                        , 'PostVaccInfection_3Vaccine_Flag'] = '2.1 Self-reported 1st symptoms whilst 3 times vaccinated (+ first positive anti-N after symptom start)'
# 2nd symptom
combined_flat_slice.loc[(combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                        & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse'])
                        & (combined_flat_slice['ItemDate'] == combined_flat_slice['ItemDate_Symptom_2_Start_EarliestResponse']) # Specify row where date matches symptom date to get vaccine status current filter to work
                        & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                  , 'PostVaccInfection_3Vaccine_Flag'] = '2.2 Self-reported 2nd symptoms whilst 3 times vaccinated (+ first positive anti-N after symptom start)'



#### 3. Where any self-reported positive antigen test is after vaccination date
antigen_selfreport_cols = ['Antigen_SelfReport_1', 
                                    'Antigen_SelfReport_2',
                                    'Antigen_SelfReport_3',
                                    'Antigen_SelfReport_4',
                                    'Antigen_SelfReport_5',
                                    'Antigen_SelfReport_6',
                                    'Antigen_SelfReport_7',
                                    'Antigen_SelfReport_8',
                                    'Antigen_SelfReport_9',
                                    'Antigen_SelfReport_10']
# Infection whilst vaccination #1
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                  , 'PostVaccInfection_1Vaccine_Flag'] = '3. Self-reported positive antigen test whilst once vaccinated'
# Test date
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                  , 'PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date'] = combined_flat_slice['ItemDate']

# Infection whilst vaccination #2
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                  , 'PostVaccInfection_2Vaccine_Flag'] = '3. Self-reported positive antigen test whilst twice vaccinated'
# Test date
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                  , 'PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date'] = combined_flat_slice['ItemDate']

# Infection whilst vaccination #3
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                  , 'PostVaccInfection_3Vaccine_Flag'] = '3. Self-reported positive antigen test whilst 3 times vaccinated'
# Test date
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                  , 'PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date'] = combined_flat_slice['ItemDate']


#### 4. Where any self-reported positive antigen test is after vaccination date AND first positive N is also after antigen positive date, can confirm post vaccination infection
# Infection whilst vaccination #1
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_1_EarliestResponse'])
                  & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                  , 'PostVaccInfection_1Vaccine_Flag'] = '4. Self-reported positive antigen test whilst once vaccinated (+ first positive anti-N after antigen positive date)'
# Infection whilst vaccination #2
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_2_EarliestResponse'])
                  & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                  , 'PostVaccInfection_2Vaccine_Flag'] = '4. Self-reported positive antigen test whilst twice vaccinated (+ first positive anti-N after antigen positive date)'
# Infection whilst vaccination #3
combined_flat_slice.loc[(combined_flat_slice['DataItem'].isin(antigen_selfreport_cols))
                  & ~(combined_flat_slice['ItemDate'].str.contains('99.', na = False))
                  & (combined_flat_slice['Result'] == 'Positive')
                  & (combined_flat_slice['ItemDate'] > combined_flat_slice['ItemDate_Vaccine_3_EarliestResponse'])
                  & (combined_flat_slice['ItemDate_N_pos_earliest'] > combined_flat_slice['ItemDate'])
                  & (combined_flat_slice['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                  , 'PostVaccInfection_3Vaccine_Flag'] = '4. Self-reported positive antigen test whilst 3 times vaccinated (+ first positive anti-N after antigen positive date)'

# Specify aggregation type
agg_dict = {'ItemDate_Vaccine_1_EarliestResponse': 'first',
            'ItemDate_Vaccine_2_EarliestResponse': 'first',
            'ItemDate_Vaccine_3_EarliestResponse': 'first',
            'ItemDate_Thriva1' : 'first',
            'ItemDate_Thriva2' : 'first',
            'ItemDate_Vaccine_Status_CoPE5' : 'first',
            'PostVaccInfection_UnknownVaccine_Flag' : 'max',
            'PostVaccInfection_1Vaccine_Flag' : 'max',
            'PostVaccInfection_1Vaccine_Symptom_1_Date' : 'first',
            'PostVaccInfection_1Vaccine_Symptom_2_Date' : 'first',
            'PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date' : 'first', # Earliest test that meets criteria
            'PostVaccInfection_2Vaccine_Flag' : 'max',
            'PostVaccInfection_2Vaccine_Symptom_1_Date' : 'first',
            'PostVaccInfection_2Vaccine_Symptom_2_Date' : 'first',
            'PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date' : 'first',
            'PostVaccInfection_3Vaccine_Flag' : 'max',
            'PostVaccInfection_3Vaccine_Symptom_1_Date' : 'first',
            'PostVaccInfection_3Vaccine_Symptom_2_Date' : 'first',
            'PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date' : 'first'
            }
# Group by and aggregate
combined_flat_slice_grouped = combined_flat_slice.groupby(['StudyNumber']).agg(agg_dict)
combined_flat_slice_grouped = combined_flat_slice_grouped.reset_index()

# -----------------------------------------------------------------------------
# Create column which gives likely date of infection
# 1 vaccination at time of infection
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|2.1'))
                   , 'PostVaccInfection_1Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Symptom_1_Date']
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Flag'].str.contains('1.2|2.2'))
                   , 'PostVaccInfection_1Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Symptom_2_Date']
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Flag'].str.contains('3.|4.'))
                   , 'PostVaccInfection_1Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date']

# 2 vaccination at time of infection
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|2.1'))
                   , 'PostVaccInfection_2Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Symptom_1_Date']
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Flag'].str.contains('1.2|2.2'))
                   , 'PostVaccInfection_2Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Symptom_2_Date']
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Flag'].str.contains('3.|4.'))
                   , 'PostVaccInfection_2Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date']

# 3 vaccination at time of infection
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Flag'].isin(['1.1 Self-reported 1st symptoms whilst 3 times vaccinated (without antigen or antibody confirmation)', '2.1 Self-reported 1st symptoms whilst 3 times vaccinated (+ first positive anti-N after symptom start)']))
                                , 'PostVaccInfection_3Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Symptom_1_Date']

combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Flag'].isin(['1.2 Self-reported 2nd symptoms whilst 3 times vaccinated (without antigen or antibody confirmation)', '2.2 Self-reported 2nd symptoms whilst 3 times vaccinated (+ first positive anti-N after symptom start)']))
                   , 'PostVaccInfection_3Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Symptom_2_Date']

combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Flag'].isin(['3. Self-reported positive antigen test whilst 3 times vaccinated','4. Self-reported positive antigen test whilst 3 times vaccinated (+ first positive anti-N after antigen positive date)']))
                   , 'PostVaccInfection_3Vaccine_Date'] = combined_flat_slice_grouped['PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date']


# Use month of infection to get likely variant
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Date'] < '2021-05-01')
                   , 'PostVaccInfection_1Vaccine_LikelyVariant'] = '1. Before May 2021: Alpha' 
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Date'] >= '2021-05-01')
                                & (combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Date'] < '2021-12-01')
                   , 'PostVaccInfection_1Vaccine_LikelyVariant'] = '2. May-Dec 2021: Delta' 
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Date'] >= '2021-12-01')
                   , 'PostVaccInfection_1Vaccine_LikelyVariant'] = '3. After Dec 2021: Omicron' 

combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Date'] < '2021-05-01')
                   , 'PostVaccInfection_2Vaccine_LikelyVariant'] = '1. Before May 2021: Alpha' 
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Date'] >= '2021-05-01')
                                & (combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Date'] < '2021-12-01')
                   , 'PostVaccInfection_2Vaccine_LikelyVariant'] = '2. May-Dec 2021: Delta' 
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Date'] >= '2021-12-01')
                   , 'PostVaccInfection_2Vaccine_LikelyVariant'] = '3. After Dec 2021: Omicron' 

combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Date'] < '2021-05-01')
                   , 'PostVaccInfection_3Vaccine_LikelyVariant'] = '1. Before May 2021: Alpha' 
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Date'] >= '2021-05-01')
                                & (combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Date'] < '2021-12-01')
                   , 'PostVaccInfection_3Vaccine_LikelyVariant'] = '2. May-Dec 2021: Delta' 
combined_flat_slice_grouped.loc[(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Date'] >= '2021-12-01')
                   , 'PostVaccInfection_3Vaccine_LikelyVariant'] = '3. After Dec 2021: Omicron' 
                
# Add month of infection in YYYY-MM format
combined_flat_slice_grouped['PostVaccInfection_1Vaccine_MonthYear'] = pd.to_datetime(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Date'], errors='coerce')
combined_flat_slice_grouped['PostVaccInfection_1Vaccine_MonthYear'] = combined_flat_slice_grouped['PostVaccInfection_1Vaccine_MonthYear'].dt.strftime('%Y-%m')

combined_flat_slice_grouped['PostVaccInfection_2Vaccine_MonthYear'] = pd.to_datetime(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Date'], errors='coerce')
combined_flat_slice_grouped['PostVaccInfection_2Vaccine_MonthYear'] = combined_flat_slice_grouped['PostVaccInfection_2Vaccine_MonthYear'].dt.strftime('%Y-%m')

combined_flat_slice_grouped['PostVaccInfection_3Vaccine_MonthYear'] = pd.to_datetime(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Date'], errors='coerce')
combined_flat_slice_grouped['PostVaccInfection_3Vaccine_MonthYear'] = combined_flat_slice_grouped['PostVaccInfection_3Vaccine_MonthYear'].dt.strftime('%Y-%m')


# Calculate number of weeks between vaccination and post-vaccination infection
combined_flat_slice_grouped['DaysBetween_Vacc1andPostVaccInfection'] = (pd.to_datetime(combined_flat_slice_grouped['PostVaccInfection_1Vaccine_Date'], errors = 'coerce') - pd.to_datetime(combined_flat_slice_grouped['ItemDate_Vaccine_1_EarliestResponse'], errors = 'coerce')).dt.days

combined_flat_slice_grouped['DaysBetween_Vacc2andPostVaccInfection'] = (pd.to_datetime(combined_flat_slice_grouped['PostVaccInfection_2Vaccine_Date'], errors = 'coerce') - pd.to_datetime(combined_flat_slice_grouped['ItemDate_Vaccine_2_EarliestResponse'], errors = 'coerce')).dt.days

combined_flat_slice_grouped['DaysBetween_Vacc3andPostVaccInfection'] = (pd.to_datetime(combined_flat_slice_grouped['PostVaccInfection_3Vaccine_Date'], errors = 'coerce') - pd.to_datetime(combined_flat_slice_grouped['ItemDate_Vaccine_3_EarliestResponse'], errors = 'coerce')).dt.days

combined_flat_slice_grouped['WeeksBetween_Vacc1andPostVaccInfection'] = (combined_flat_slice_grouped['DaysBetween_Vacc1andPostVaccInfection']/7).apply(np.floor) # round down
combined_flat_slice_grouped['WeeksBetween_Vacc2andPostVaccInfection'] = (combined_flat_slice_grouped['DaysBetween_Vacc2andPostVaccInfection']/7).apply(np.floor) # round down
combined_flat_slice_grouped['WeeksBetween_Vacc3andPostVaccInfection'] = (combined_flat_slice_grouped['DaysBetween_Vacc3andPostVaccInfection']/7).apply(np.floor) # round down

combined_flat_slice_grouped = combined_flat_slice_grouped.reset_index(drop = True)

# Use merge to add flags and dates to all rows of given individuals
selected_cols = ['StudyNumber',
                 'PostVaccInfection_UnknownVaccine_Flag','PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_1Vaccine_Symptom_1_Date', 'PostVaccInfection_1Vaccine_Symptom_2_Date', 'PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date', 'PostVaccInfection_1Vaccine_Date', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_2Vaccine_Symptom_1_Date', 'PostVaccInfection_2Vaccine_Symptom_2_Date', 'PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date', 'PostVaccInfection_2Vaccine_Date', 'PostVaccInfection_3Vaccine_Flag', 'PostVaccInfection_3Vaccine_Symptom_1_Date', 'PostVaccInfection_3Vaccine_Symptom_2_Date', 'PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date', 'PostVaccInfection_3Vaccine_Date',
                 'WeeksBetween_Vacc1andPostVaccInfection', 'WeeksBetween_Vacc2andPostVaccInfection', 'WeeksBetween_Vacc3andPostVaccInfection',
                 'DaysBetween_Vacc1andPostVaccInfection', 'DaysBetween_Vacc2andPostVaccInfection', 'DaysBetween_Vacc3andPostVaccInfection',
                 'PostVaccInfection_1Vaccine_LikelyVariant',
                 'PostVaccInfection_2Vaccine_LikelyVariant',
                 'PostVaccInfection_3Vaccine_LikelyVariant',
                 'PostVaccInfection_1Vaccine_MonthYear',
                 'PostVaccInfection_2Vaccine_MonthYear',
                 'PostVaccInfection_3Vaccine_MonthYear',
                 'PostVaccInfection_Latest_Date',
                 ]
combined_flat = pd.merge(combined_flat,combined_flat_slice_grouped[selected_cols], how = 'left', on = 'StudyNumber')

test_grouped = combined_flat[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag']].groupby('StudyNumber').max()


# -----------------------------------------------------------------------------
# Add text to specify whether post-vaccination infection was before Thriva #1, between Thriva #1 and #2, or after Thriva #2

# Work with a copy of dataset 
combined_flat_slice = combined_flat.copy()

#### Infection whilst once vaccinated
# Before Thriva #1
combined_flat.loc[((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_1_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_2_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_1Vaccine_Flag'] = combined_flat['PostVaccInfection_1Vaccine_Flag'] + ' (before Thriva #1)'


# After Thriva #1
combined_flat.loc[((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_1_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_2_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_1Vaccine_Flag'] = combined_flat['PostVaccInfection_1Vaccine_Flag'] + ' (after Thriva #1)'

# Before Thriva #2
combined_flat.loc[((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_1_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_2_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_1Vaccine_Flag'] = combined_flat['PostVaccInfection_1Vaccine_Flag'] + ' (before Thriva #2)'


# After Thriva #2
combined_flat.loc[((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_1_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_1Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_1Vaccine_Symptom_2_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_1Vaccine_Flag'] = combined_flat['PostVaccInfection_1Vaccine_Flag'] + ' (after Thriva #2)'


#### Infection whilst twice vaccinated
# Before Thriva #1
combined_flat.loc[((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_1_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_2_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_2Vaccine_Flag'] = combined_flat['PostVaccInfection_2Vaccine_Flag'] + ' (before Thriva #1)'

# After Thriva #1
combined_flat.loc[((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_1_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_2_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_2Vaccine_Flag'] = combined_flat['PostVaccInfection_2Vaccine_Flag'] + ' (after Thriva #1)'

# Before Thriva #2
combined_flat.loc[((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_1_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_2_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_2Vaccine_Flag'] = combined_flat['PostVaccInfection_2Vaccine_Flag'] + ' (before Thriva #2)'

# After Thriva #2
combined_flat.loc[((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_1_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_2Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_2Vaccine_Symptom_2_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_2Vaccine_Flag'] = combined_flat['PostVaccInfection_2Vaccine_Flag'] + ' (after Thriva #2)'



#### Infection whilst 3 times vaccinated
# Before Thriva #1
combined_flat.loc[((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_1_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_2_Date'] < combined_flat['ItemDate_Thriva1']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_3Vaccine_Flag'] = combined_flat['PostVaccInfection_3Vaccine_Flag'] + ' (before Thriva #1)'

# After Thriva #1
combined_flat.loc[((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_1_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_2_Date'] >= combined_flat['ItemDate_Thriva1']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_3Vaccine_Flag'] = combined_flat['PostVaccInfection_3Vaccine_Flag'] + ' (after Thriva #1)'

# Before Thriva #2
combined_flat.loc[((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_1_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_2_Date'] < combined_flat['ItemDate_Thriva2']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_3Vaccine_Flag'] = combined_flat['PostVaccInfection_3Vaccine_Flag'] + ' (before Thriva #2)'

# After Thriva #2
combined_flat.loc[((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('3.'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('4.')))
                        & (combined_flat['PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 3 & 4 - compare thriva 1 to earliest antigen test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.1')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_1_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 1.1 & 2.1 - compare thriva 1 to symptom 1 test date
                        |
                        ((combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('1.2'))
                        | (combined_flat['PostVaccInfection_3Vaccine_Flag'].str.contains('2.2')))
                        & (combined_flat['PostVaccInfection_3Vaccine_Symptom_2_Date'] >= combined_flat['ItemDate_Thriva2']) # For flag categories 1.2 & 2.2 - compare thriva 1 to symptom 2 test date
                        
                        , 'PostVaccInfection_3Vaccine_Flag'] = combined_flat['PostVaccInfection_3Vaccine_Flag'] + ' (after Thriva #2)'


# Group by study number to test flags working as expected
combined_flat_grouped_vacc1 = combined_flat[['StudyNumber','ItemDate_Thriva1','ItemDate_Thriva2','PostVaccInfection_1Vaccine_Flag','PostVaccInfection_1Vaccine_SelfReportAntigenTest_Date','PostVaccInfection_1Vaccine_Symptom_1_Date','PostVaccInfection_1Vaccine_Symptom_2_Date']].groupby(['StudyNumber']).max()

combined_flat_grouped_vacc2 = combined_flat[['StudyNumber','ItemDate_Thriva1','ItemDate_Thriva2','PostVaccInfection_2Vaccine_Flag','PostVaccInfection_2Vaccine_SelfReportAntigenTest_Date','PostVaccInfection_2Vaccine_Symptom_1_Date','PostVaccInfection_2Vaccine_Symptom_2_Date']].groupby(['StudyNumber']).max()

combined_flat_grouped_vacc3 = combined_flat[['StudyNumber','ItemDate_Thriva1','ItemDate_Thriva2','PostVaccInfection_3Vaccine_Flag','PostVaccInfection_3Vaccine_SelfReportAntigenTest_Date','PostVaccInfection_3Vaccine_Symptom_1_Date','PostVaccInfection_3Vaccine_Symptom_2_Date']].groupby(['StudyNumber']).max()



#%% Further processing of covariates
# -----------------------------------------------------------------------------
# Process and group deprivation data to suit use as continuous and categorical variable
# Set data type of ambiguous columns, to avoid separation of equivalent values
combined_flat['imd_quintile'] = combined_flat['imd_quintile'].astype(str)
combined_flat['imd_decile'] = combined_flat['imd_decile'].fillna('NoDataAvailable').astype(str)


# Reverse numbering of IMD deprivation decile, so that increasing number represents increasing deprivation rather than decreasing - more intuitive to interpret in later regression analysis
codebook = {}
codebook['imd_decile_reverse'] =  {'1.0': '10',
                                     '2.0': '9',
                                     '3.0': '8',
                                     '4.0': '7',
                                     '5.0': '6',
                                     '6.0': '5',
                                     '7.0': '4',
                                     '8.0': '3',
                                     '9.0': '2',
                                     '10.0': '1',
                                     'NoDataAvailable': 'NoDataAvailable'}
combined_flat['imd_decile_reverse'] = combined_flat['imd_decile'].map(codebook['imd_decile_reverse'])
codebook['imd_quintile_reverse'] =  {'1': '5',
                                     '2': '4',
                                     '3': '3',
                                     '4': '2',
                                     '5': '1',
                                     'NoDataAvailable': 'NoDataAvailable'}
combined_flat['imd_quintile_reverse'] = combined_flat['imd_quintile'].map(codebook['imd_quintile_reverse'])



# -----------------------------------------------------------------------------
# Groups for those who provided symptom start date
combined_flat.loc[(combined_flat['DaysSince_Symptom_1_Start_EarliestResponse'] < 0)
                  , 'TimeSince_Symptom_1_Start_EarliestResponse'] = '1. Before symptom start'
combined_flat.loc[(combined_flat['DaysSince_Symptom_1_Start_EarliestResponse'] >= 0)
                  & (combined_flat['DaysSince_Symptom_1_Start_EarliestResponse'] < 168)
                  , 'TimeSince_Symptom_1_Start_EarliestResponse'] = '2. 0-24 Weeks after symptom start'
combined_flat.loc[(combined_flat['DaysSince_Symptom_1_Start_EarliestResponse'] >= 168)
                  , 'TimeSince_Symptom_1_Start_EarliestResponse'] = '3. 24+ Weeks after symptom start'


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
# Create additional grouped max columns for CoPEs up to CoPE 4 only (CoPE 5 excluded), to use for Thriva #1 analyses

# GAD-2
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'PHQ4_Anxiety')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Value']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Value':'PHQ4_Anxiety_Value_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['PHQ4_Anxiety_Value_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)

# PHQ-2
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'PHQ4_Depression')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Value']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Value':'PHQ4_Depression_Value_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['PHQ4_Depression_Value_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)

# PHQ-4
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'PHQ4_AnxietyandDepression')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Value']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Value':'PHQ4_AnxietyandDepression_Value_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['PHQ4_AnxietyandDepression_Value_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)

# HADS anxiety
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'HADS_Anxiety')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Value']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Value':'HADS_Anxiety_Value_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['HADS_Anxiety_Value_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)

# HADS depression
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'HADS_Depression')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Value']
max_status = combined_flat_slice[columns].groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Value':'HADS_Depression_Value_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['HADS_Depression_Value_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)

test = combined_flat[['StudyNumber','StudyName','DataItem','Value','HADS_Depression_Value_MaxToCoPE4','HADS_Depression_Value_Max']]


# Had Covid Ever, self reported. 
# Group based on max - most affirmative
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'HadCovid_Ever')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Result']
max_status = combined_flat_slice[columns].fillna('0.0 Unknown - individual did not complete CoPE').groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Result':'HadCovid_Ever_SelfReport_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['HadCovid_Ever_SelfReport_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)
combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4'] = combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4'].fillna('0.0 Unknown - individual did not complete CoPE')

test = combined_flat[['StudyNumber','StudyName','DataItem','Value','HadCovid_Ever_SelfReport','HadCovid_Ever_SelfReport_MaxToCoPE4']]


# Symptom duration
# Group based on max - longest duration
# select only symptom duration results 
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'SymptomDuration')
                                    & (combined_flat['StudyName'].isin(['CoPE_1', 'CoPE_2', 'CoPE_2_paper', 'CoPE_3', 'CoPE_4 (full)','CoPE_4 (abridged)']))]
columns = ['StudyNumber', 'DataItem','Result']
max_status = combined_flat_slice[columns].fillna('0.0 Unknown - individual did not complete CoPE').groupby(['StudyNumber']).max()
max_status = max_status.rename(columns = {'Result':'SymptomDuration_MaxToCoPE4'})
# Use merge to add grouped column
combined_flat = pd.merge(combined_flat,max_status['SymptomDuration_MaxToCoPE4'], how = 'left', left_on = 'StudyNumber', right_index=True)
combined_flat['SymptomDuration_MaxToCoPE4'] = combined_flat['SymptomDuration_MaxToCoPE4'].fillna('0.0 Unknown - individual did not complete CoPE')

# If symptom start date provided, but answered 'no covid', replace 'no covid' with 'unknown didn't answer' in symptom duration to eliminate inconsistency
combined_flat.loc[(~(combined_flat['ItemDate_Symptom_1_Start_EarliestResponse'].isin([np.nan])))
                  & (combined_flat['SymptomDuration_MaxToCoPE4'] == '0.2 N/A - no covid')
                  , 'SymptomDuration_MaxToCoPE4'] = '0.1 Unknown - Answer not provided in CoPE'




# -----------------------------------------------------------------------------
# Group HADS Anxiety and Depression assessment results

# Group full column, including all CoPEs
# 1. Group above/below threshold
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_Max'] < 8),'HADS_Anxiety_grouped_cat2'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_Max'] >= 8),'HADS_Anxiety_grouped_cat2'] = '8+, above threshold'
combined_flat['HADS_Anxiety_grouped_cat2'] = combined_flat['HADS_Anxiety_grouped_cat2'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['HADS_Depression_Value_Max'] < 8),'HADS_Depression_grouped_cat2'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Depression_Value_Max'] >= 8),'HADS_Depression_grouped_cat2'] = '8+, above threshold'
combined_flat['HADS_Depression_grouped_cat2'] = combined_flat['HADS_Depression_grouped_cat2'].fillna('NoDataAvailable')

# 2. Group moderate +
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_Max'] < 8),'HADS_Anxiety_grouped_cat3'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_Max'] >= 8),'HADS_Anxiety_grouped_cat3'] = '8-10, mild'
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_Max'] >= 11),'HADS_Anxiety_grouped_cat3'] = '11+, moderate, severe'
combined_flat['HADS_Anxiety_grouped_cat3'] = combined_flat['HADS_Anxiety_grouped_cat3'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['HADS_Depression_Value_Max'] < 8),'HADS_Depression_grouped_cat3'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Depression_Value_Max'] >= 8),'HADS_Depression_grouped_cat3'] = '8-10, mild'
combined_flat.loc[(combined_flat['HADS_Depression_Value_Max'] >= 11),'HADS_Depression_grouped_cat3'] = '11+, moderate, severe'
combined_flat['HADS_Depression_grouped_cat3'] = combined_flat['HADS_Depression_grouped_cat3'].fillna('NoDataAvailable')

# Group column giving max up to and including CoPE 4 only, for analysis of Thriva #1
# 1. Group above/below threshold
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_MaxToCoPE4'] < 8),'HADS_Anxiety_grouped_MaxToCoPE4_cat2'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_MaxToCoPE4'] >= 8),'HADS_Anxiety_grouped_MaxToCoPE4_cat2'] = '8+, above threshold'
combined_flat['HADS_Anxiety_grouped_MaxToCoPE4_cat2'] = combined_flat['HADS_Anxiety_grouped_MaxToCoPE4_cat2'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['HADS_Depression_Value_MaxToCoPE4'] < 8),'HADS_Depression_grouped_MaxToCoPE4_cat2'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Depression_Value_MaxToCoPE4'] >= 8),'HADS_Depression_grouped_MaxToCoPE4_cat2'] = '8+, above threshold'
combined_flat['HADS_Depression_grouped_MaxToCoPE4_cat2'] = combined_flat['HADS_Depression_grouped_MaxToCoPE4_cat2'].fillna('NoDataAvailable')

# 2. Group moderate +
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_MaxToCoPE4'] < 8),'HADS_Anxiety_grouped_MaxToCoPE4_cat3'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_MaxToCoPE4'] >= 8),'HADS_Anxiety_grouped_MaxToCoPE4_cat3'] = '8-10, mild'
combined_flat.loc[(combined_flat['HADS_Anxiety_Value_MaxToCoPE4'] >= 11),'HADS_Anxiety_grouped_MaxToCoPE4_cat3'] = '11+, moderate, severe'
combined_flat['HADS_Anxiety_grouped_MaxToCoPE4_cat3'] = combined_flat['HADS_Anxiety_grouped_MaxToCoPE4_cat3'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['HADS_Depression_Value_MaxToCoPE4'] < 8),'HADS_Depression_grouped_MaxToCoPE4_cat3'] = '0-7, below threshold'
combined_flat.loc[(combined_flat['HADS_Depression_Value_MaxToCoPE4'] >= 8),'HADS_Depression_grouped_MaxToCoPE4_cat3'] = '8-10, mild'
combined_flat.loc[(combined_flat['HADS_Depression_Value_MaxToCoPE4'] >= 11),'HADS_Depression_grouped_MaxToCoPE4_cat3'] = '11+, moderate, severe'
combined_flat['HADS_Depression_grouped_MaxToCoPE4_cat3'] = combined_flat['HADS_Depression_grouped_MaxToCoPE4_cat3'].fillna('NoDataAvailable')




# -----------------------------------------------------------------------------
# Group GAD-2, PHQ-2, PHQ4 combined Anxiety and Depression assessment results

# Group full column, including all CoPEs
# 1. Binary - above/below threshold
combined_flat.loc[(combined_flat['PHQ4_Anxiety_Value_Max'] < 3),'PHQ4_Anxiety_grouped_cat2'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_Anxiety_Value_Max'] >= 3),'PHQ4_Anxiety_grouped_cat2'] = '3+, above threshold'
combined_flat['PHQ4_Anxiety_grouped_cat2'] = combined_flat['PHQ4_Anxiety_grouped_cat2'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['PHQ4_Depression_Value_Max'] < 3),'PHQ4_Depression_grouped_cat2'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_Depression_Value_Max'] >= 3),'PHQ4_Depression_grouped_cat2'] = '3+, above threshold'
combined_flat['PHQ4_Depression_grouped_cat2'] = combined_flat['PHQ4_Depression_grouped_cat2'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_Max'] < 3),'PHQ4_AnxietyandDepression_grouped_cat2'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_Max'] >= 3),'PHQ4_AnxietyandDepression_grouped_cat2'] = '3+, above threshold'
combined_flat['PHQ4_AnxietyandDepression_grouped_cat2'] = combined_flat['PHQ4_AnxietyandDepression_grouped_cat2'].fillna('NoDataAvailable')

# 2. Group moderate +
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_Max'] < 3),'PHQ4_AnxietyandDepression_grouped_cat3'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_Max'] >= 3),'PHQ4_AnxietyandDepression_grouped_cat3'] = '3-5, mild'
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_Max'] >= 6),'PHQ4_AnxietyandDepression_grouped_cat3'] = '6+, moderate, severe'
combined_flat['PHQ4_AnxietyandDepression_grouped_cat3'] = combined_flat['PHQ4_AnxietyandDepression_grouped_cat3'].fillna('NoDataAvailable')

# Group column giving max up to and including CoPE 4 only, for analysis of Thriva #1
# 1. Binary - above/below threshold
combined_flat.loc[(combined_flat['PHQ4_Anxiety_Value_MaxToCoPE4'] < 3),'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_Anxiety_Value_MaxToCoPE4'] >= 3),'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2'] = '3+, above threshold'
combined_flat['PHQ4_Anxiety_grouped_MaxToCoPE4_cat2'] = combined_flat['PHQ4_Anxiety_grouped_MaxToCoPE4_cat2'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['PHQ4_Depression_Value_MaxToCoPE4'] < 3),'PHQ4_Depression_grouped_MaxToCoPE4_cat2'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_Depression_Value_MaxToCoPE4'] >= 3),'PHQ4_Depression_grouped_MaxToCoPE4_cat2'] = '3+, above threshold'
combined_flat['PHQ4_Depression_grouped_MaxToCoPE4_cat2'] = combined_flat['PHQ4_Depression_grouped_MaxToCoPE4_cat2'].fillna('NoDataAvailable')

combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_MaxToCoPE4'] < 3),'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_MaxToCoPE4'] >= 3),'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2'] = '3+, above threshold'
combined_flat['PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2'] = combined_flat['PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2'].fillna('NoDataAvailable')

# 2. Group moderate +
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_MaxToCoPE4'] < 3),'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3'] = '0-2, below threshold'
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_MaxToCoPE4'] >= 3),'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3'] = '3-5, mild'
combined_flat.loc[(combined_flat['PHQ4_AnxietyandDepression_Value_MaxToCoPE4'] >= 6),'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3'] = '6+, moderate, severe'
combined_flat['PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3'] = combined_flat['PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3'].fillna('NoDataAvailable')


# -----------------------------------------------------------------------------
# Create wider grouping of self-reported had covid ever 
# Group full column, including all CoPEs
# Group self-reported positive test
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport'].str.contains('2.0 Unsure'))
                  , 'HadCovid_Ever_SelfReport_grouped'] = '2.0 Unsure'
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport'].str.contains('2.1 SuspectedCovid'))
                  , 'HadCovid_Ever_SelfReport_grouped'] = '2.1 SuspectedCovid'
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport'].str.contains('3. PositiveCovid'))
                  , 'HadCovid_Ever_SelfReport_grouped'] = '3. PositiveCovid'

combined_flat['HadCovid_Ever_SelfReport_grouped'] = combined_flat['HadCovid_Ever_SelfReport_grouped'].fillna(combined_flat['HadCovid_Ever_SelfReport'])

# Binary grouping as positive confirmed has too few < 250 results to present
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport_grouped'].isin(['2.1 SuspectedCovid', '3. PositiveCovid']))
                  , 'HadCovid_Ever_SelfReport_binary'] = '2-3. SuspectedOrPositiveCovid'

combined_flat['HadCovid_Ever_SelfReport_binary'] = combined_flat['HadCovid_Ever_SelfReport_binary'].fillna(combined_flat['HadCovid_Ever_SelfReport_grouped'])

# Rename original variable to distinguish naming
combined_flat = combined_flat.rename(columns = {'HadCovid_Ever_SelfReport':'HadCovid_Ever_SelfReport_original'}) 


# Group column giving max up to and including CoPE 4 only, for analysis of Thriva #1
# Group self-reported positive test
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4'].str.contains('2.0 Unsure'))
                  , 'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'] = '2.0 Unsure'
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4'].str.contains('2.1 SuspectedCovid'))
                  , 'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'] = '2.1 SuspectedCovid'
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4'].str.contains('3. PositiveCovid'))
                  , 'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'] = '3. PositiveCovid'

combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'] = combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'].fillna(combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4'])

# Binary grouping as positive confirmed has too few < 250 results to present
combined_flat.loc[(combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'].isin(['2.1 SuspectedCovid', '3. PositiveCovid']))
                  , 'HadCovid_Ever_SelfReport_MaxToCoPE4_binary'] = '2-3. SuspectedOrPositiveCovid'

combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4_binary'] = combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4_binary'].fillna(combined_flat['HadCovid_Ever_SelfReport_MaxToCoPE4_grouped'])

# Rename original variable to distinguish naming
combined_flat = combined_flat.rename(columns = {'HadCovid_Ever_SelfReport_MaxToCoPE4':'HadCovid_Ever_SelfReport_MaxToCoPE4_original'}) 


# -----------------------------------------------------------------------------
# Grouping symptom duration and affected functioning duration into larger categories

# Group full column, including all CoPEs
# Symptom duration
combined_flat.loc[(combined_flat['SymptomDuration_Max'].isin(['2: 0-2 weeks','3: 2-4 weeks'])),'SymptomDuration_Max_grouped'] = '2: 0-4 weeks'
combined_flat.loc[(combined_flat['SymptomDuration_Max'].isin(['4: 4-12 weeks'])),'SymptomDuration_Max_grouped'] = '4: 4-12 weeks'
combined_flat.loc[(combined_flat['SymptomDuration_Max'].isin(['5: 12+ weeks','5.1: 12+ weeks: 3-6 months', '5.2: 12+ weeks: 6-12 months', '5.3: 12+ weeks: 12+ months'])),'SymptomDuration_Max_grouped'] = '5: 12+ weeks'
combined_flat['SymptomDuration_Max_grouped'] = combined_flat['SymptomDuration_Max_grouped'].fillna(combined_flat['SymptomDuration_Max'])

# Affected functioning duration
combined_flat.loc[(combined_flat['AffectedFunctioningDuration_Max'].isin(['3: 1-3 days','4: 4-6 days','5: 7-13 days','6: 2-4 weeks'])),'AffectedFunctioningDuration_Max_grouped'] = '3: 0-4 weeks'
combined_flat.loc[(combined_flat['AffectedFunctioningDuration_Max'].isin(['7: 4-12 weeks'])),'AffectedFunctioningDuration_Max_grouped'] = '7: 4-12 weeks'
combined_flat.loc[(combined_flat['AffectedFunctioningDuration_Max'].isin(['8: 12+ weeks', '8.1: 12+ weeks: 3-6 months', '8.2: 12+ weeks: 6-12 months', '8.3: 12+ weeks: 12+ months'])),'AffectedFunctioningDuration_Max_grouped'] = '8: 12+ weeks'
combined_flat['AffectedFunctioningDuration_Max_grouped'] = combined_flat['AffectedFunctioningDuration_Max_grouped'].fillna(combined_flat['AffectedFunctioningDuration_Max'])

# Group column giving max up to and including CoPE 4 only, for analysis of Thriva #1
# Symptom duration
combined_flat.loc[(combined_flat['SymptomDuration_MaxToCoPE4'].isin(['2: 0-2 weeks','3: 2-4 weeks'])),'SymptomDuration_MaxToCoPE4_grouped'] = '2: 0-4 weeks'
combined_flat.loc[(combined_flat['SymptomDuration_MaxToCoPE4'].isin(['4: 4-12 weeks'])),'SymptomDuration_MaxToCoPE4_grouped'] = '4: 4-12 weeks'
combined_flat.loc[(combined_flat['SymptomDuration_MaxToCoPE4'].isin(['5: 12+ weeks'])),'SymptomDuration_MaxToCoPE4_grouped'] = '5: 12+ weeks'
combined_flat['SymptomDuration_MaxToCoPE4_grouped'] = combined_flat['SymptomDuration_MaxToCoPE4_grouped'].fillna(combined_flat['SymptomDuration_MaxToCoPE4'])



# -----------------------------------------------------------------------------
# Grouping HospitalisedFlag into clearer categories
combined_flat.loc[(combined_flat['HospitalisedFlag'].isin(['0.2 N/A - question not relevant','no'])),'HospitalisedFlag_grouped'] = 'no'
combined_flat['HospitalisedFlag_grouped'] = combined_flat['HospitalisedFlag_grouped'].fillna(combined_flat['HospitalisedFlag'])


# -----------------------------------------------------------------------------
# Grouping PrePandemicHealth_Earliest Poor and Fair due to low numbers in 'Poor'
combined_flat.loc[(combined_flat['PrePandemicHealth_Earliest'].isin(['1. Poor','2. Fair'])),'PrePandemicHealth_Earliest_grouped'] = '1,2. Poor, Fair'
combined_flat['PrePandemicHealth_Earliest_grouped'] = combined_flat['PrePandemicHealth_Earliest_grouped'].fillna(combined_flat['PrePandemicHealth_Earliest'])

# Grouping to be binary, 1-2 or 3-5
combined_flat.loc[(combined_flat['PrePandemicHealth_Earliest'].isin(['1. Poor', '2. Fair'])),'PrePandemicHealth_Earliest_binary'] = '1,2. Poor, Fair'
combined_flat.loc[(combined_flat['PrePandemicHealth_Earliest'].isin(['3. Good', '4. Very Good', '5. Excellent'])),'PrePandemicHealth_Earliest_binary'] = '3-5. Good, Very Good, Excellent'
combined_flat['PrePandemicHealth_Earliest_binary'] = combined_flat['PrePandemicHealth_Earliest_binary'].fillna(combined_flat['PrePandemicHealth_Earliest'])

# Creating PrePandemicHealth_Earliest ordinal to include as continuous variable
codebook['health_continuous'] =  {'1. Poor': '5',
                                     '2. Fair': '4',
                                     '3. Good': '3',
                                     '4. Very Good': '2',
                                     '5. Excellent': '1'
                                     }
combined_flat['PrePandemicHealth_Earliest_ordinal'] = combined_flat['PrePandemicHealth_Earliest'].map(codebook['health_continuous'])
combined_flat['PrePandemicHealth_Earliest_ordinal'] = combined_flat['PrePandemicHealth_Earliest_ordinal'].fillna(combined_flat['PrePandemicHealth_Earliest'])

# Rename original variable to distinguish naming
combined_flat = combined_flat.rename(columns = {'PrePandemicHealth_Earliest':'PrePandemicHealth_Earliest_original'}) 

# -----------------------------------------------------------------------------
# Frailty index categories as ordinal series
for n in range (1,5,1):
    combined_flat.loc[(combined_flat['FrailtyIndexOngoingCat'].str.contains(str(n)))
                    ,'FrailtyIndexOngoingCat_Ordinal'] = n
    
# -----------------------------------------------------------------------------
# Grouping Frailty Index Frail and Very Frail due to low numbers in 'Very Frail'
combined_flat.loc[(combined_flat['FrailtyIndexOngoingCat'].isin(['3. Frail','4. Very frail'])),'FrailtyIndexOngoingCat_grouped'] = '3,4. Frail, Very Frail'
combined_flat['FrailtyIndexOngoingCat_grouped'] = combined_flat['FrailtyIndexOngoingCat_grouped'].fillna(combined_flat['FrailtyIndexOngoingCat'])

# Rename original variable to distinguish naming
combined_flat = combined_flat.rename(columns = {'FrailtyIndexOngoingCat':'FrailtyIndexOngoingCat_original'}) 


# -----------------------------------------------------------------------------
# Create binary variable for BMI - obese or non-obese
combined_flat.loc[(combined_flat['BMI_cat5'].isin(['1: 0-18.5','2: 18.5-25','3: 25-30'])),'BMI_cat2'] = 'Non obese, < 30 kg/m2'
combined_flat.loc[(combined_flat['BMI_cat5'].isin(['4: 30-35', '5: 35+'])),'BMI_cat2'] = 'Obese, >= 30 kg/m2'
combined_flat['BMI_cat2'] = combined_flat['BMI_cat2'].fillna(combined_flat['BMI_cat5'])


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


# Create binary variable for characterisation of cohorts
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] == 0)
                  ,'SubDomain_MultimorbidityCount_Selected_Binary'] = '0'
combined_flat.loc[(combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] >= 1)
                  ,'SubDomain_MultimorbidityCount_Selected_Binary'] = '1+'
combined_flat['SubDomain_MultimorbidityCount_Selected_Binary'] = combined_flat['SubDomain_MultimorbidityCount_Selected_Binary'].fillna(nan_fill_text)


combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'] = combined_flat['SubDomain_MultimorbidityCount_Selected_Ungrouped'].fillna(nan_fill_text)

test = combined_flat[col_list + ['SubDomain_MultimorbidityCount_Selected_Ungrouped','SubDomain_MultimorbidityCount_Selected_Grouped']]


# -----------------------------------------------------------------------------
# Create custom grouping of age, 0-50 then decades up to 80+ 
# Group 18-50s
combined_flat.loc[(combined_flat['age_10yr_bands'].isin(['2: 18-30','3: 30-40','4: 40-50']))
                  ,'age_custom'] = '2: 18-50'
# Fill rest of field with decades from 10 year banding
combined_flat['age_custom'] = combined_flat['age_custom'].fillna(combined_flat['age_10yr_bands'])

# -----------------------------------------------------------------------------
# Create custom binary grouping of age, 0-50 or 50+ 
# Group 18-50s
combined_flat.loc[(combined_flat['age_3_bands'].isin(['2: 50-70', '3: 70+']))
                  ,'age_binary'] = '2: 50+'
# Fill rest of field with decades from 10 year banding
combined_flat['age_binary'] = combined_flat['age_binary'].fillna(combined_flat['age_3_bands'])

# -----------------------------------------------------------------------------
# Generate columns to show Thriva 1 and 2 anti-N results for estimating natural infection prevalence
# Thriva 1
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'Antibody_N')
                                    & (combined_flat['StudyName'].isin(['Thriva']))]
columns = ['StudyNumber', 'DataItem','Result']
combined_flat_slice = combined_flat_slice[columns]
combined_flat_slice = combined_flat_slice.rename(columns = {'Result':'Result_Thriva_N'})

# Use merge to add sliced columns
combined_flat = pd.merge(combined_flat,combined_flat_slice[['StudyNumber','Result_Thriva_N']], how = 'left', on = 'StudyNumber')

# Thriva 2
combined_flat_slice = combined_flat[(combined_flat['DataItem'] == 'Antibody_N')
                                    & (combined_flat['StudyName'].isin(['Thriva #2']))]
columns = ['StudyNumber', 'DataItem','Result']
combined_flat_slice = combined_flat_slice[columns]
combined_flat_slice = combined_flat_slice.rename(columns = {'Result':'Result_Thriva2_N'})

# Use merge to add sliced columns
combined_flat = pd.merge(combined_flat,combined_flat_slice[['StudyNumber','Result_Thriva2_N']], how = 'left', on = 'StudyNumber')


#%% Create 'dummy variables' from categorical variables (binary variables for each category) to use for input/control variables
col_list = combined_flat.columns.to_list() # save column names

# List of all possible categorical input variables
var_input_all = ['age_3_bands', 'age_10yr_bands', 'age_custom', 'age_binary', 'ethnicity_ons_cat_combined_whitenonwhite', 'sex', 'edu_bin_combined',
                 
                 'imd_decile', 'imd_quintile', 'PrePandemicEmploymentStatus_Earliest', 'RUC_grouped', 
                 
                 'Vaccine_1_name_grouped', 'Vaccine_2_name_grouped', 'Vaccine_3_name_grouped', 
                 
                 'Result_Thriva_N', 'Result_Thriva2_N',
                 'NaturalInfection_WideCDC_Interpretation_MaxToDate',
                 
                 'HadCovid_Ever_SelfReport_original', 'HadCovid_Ever_SelfReport_grouped', 'HadCovid_Ever_SelfReport_binary',
                 'HadCovid_Ever_SelfReport_MaxToCoPE4_original', 'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped', 'HadCovid_Ever_SelfReport_MaxToCoPE4_binary', 'PostVaccInfection_2Vaccine_Flag',
                 
                 
                 'SymptomDuration_Max_grouped', 'SymptomDuration_MaxToCoPE4_grouped',
                 
                 'AffectedFunctioningDuration_Max_grouped', 'HospitalisedFlag_grouped',
                 
                 'PrePandemicHealth_Earliest_original', 'PrePandemicHealth_Earliest_grouped', 'PrePandemicHealth_Earliest_binary',
                 
                 'HADS_Anxiety_grouped_cat2', 'HADS_Anxiety_grouped_cat3', 
                 'HADS_Depression_grouped_cat2', 'HADS_Depression_grouped_cat3',  
                 'HADS_Anxiety_grouped_MaxToCoPE4_cat2', 'HADS_Depression_grouped_MaxToCoPE4_cat2',
                 'HADS_Anxiety_grouped_MaxToCoPE4_cat3', 'HADS_Depression_grouped_MaxToCoPE4_cat3',
                 
                 'PHQ4_Anxiety_grouped_cat2', 'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2',
                 'PHQ4_Depression_grouped_cat2', 'PHQ4_Depression_grouped_MaxToCoPE4_cat2',
                 
                 'PHQ4_AnxietyandDepression_grouped_cat3', 'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3', 'PHQ4_AnxietyandDepression_grouped_cat2', 'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2', 
                 
                 'BMI_cat2', 'BMI_cat5', 
                 
                 'SubDomain_DiabetesType1', 'Domain_DiabetesType2', 'SubDomain_Hypertension', 'SubDomain_HighCholesterol', 'SubDomain_Asthma', 'SubDomain_AnxietyOrDepression', 'SubDomain_DiabetesAny', 'SubDomain_HeartDiseaseOrFailure', 
                 'SubDomain_MultimorbidityCount_Selected_Ungrouped', 'SubDomain_MultimorbidityCount_Selected_Grouped', 
                 
                 'ShieldingFlag', 'FrailtyIndexOngoingCat_original', 'FrailtyIndexOngoingCat_grouped', 
                 
                 
                 'Domain_CardiacDisease', 'Domain_CardiacRiskFactors', 'Domain_CancerAny', 'Domain_NeurologicalDisease', 'Domain_SubjectiveMemoryImpairment', 'Domain_ChronicLungDisease', 'Domain_Arthritis', 'Domain_Osteoporosis', 'SubDomain_HeartFailure', 'SubDomain_AtrialFibrillation', 'SubDomain_CoronaryHeartDisease', 'SubDomain_CancerLeukemia', 'SubDomain_CancerLung', 'SubDomain_CancerLymphoma', 'SubDomain_AnxietyStressDisorder', 'SubDomain_Depression', 'SubDomain_Stroke', 'SubDomain_Alzheimers', 'SubDomain_RheumatoidArthritis', 'SubDomain_Epilepsy', 'SubDomain_LiverCirrhosis', 
                 
                 'SubDomain_MultimorbidityCount_Grouped', 
                 
                 'MedicationFlag_NHSShielding_Immunosuppressant', 'MedicationFlag_SevereAsthma', 'MedicationFlag_SevereCOPD', 'MedicationFlag_ShieldingListAny', 'MedicationFlag_ED_Immunosuppressant',
                 'ValueGrouped_50s','ValueGrouped_100s', 'ValueGrouped_Binary',
                 'Value_decile_thriva1vacc1', 'Value_quintile_thriva1vacc1',
                 ]


# Create dummy variables
dummy_var_list_full = []
for var in var_input_all:
    combined_flat[var] = combined_flat[var].fillna('NaN') # fill NaN with 'No data' so missing data can be distinguished from 0 results
    cat_list='var'+'_'+var # variable name
    cat_list = pd.get_dummies(combined_flat[var], prefix=var) # create binary variable of category value
    combined_flat=combined_flat.join(cat_list) # join new column to dataframe
    dummy_var_list_full = dummy_var_list_full + cat_list.columns.to_list()

col_list_with_dummy = combined_flat.columns.to_list() # save column names



#%% Select which dummy variables will act as the 'reference category' and remove them from dataset
dummy_ref_var_list = ['age_3_bands_1: 0-50', 
                      'age_10yr_bands_5: 50-60',
                      'age_custom_2: 18-50',
                      'age_binary_1: 0-50',
                      'ethnicity_ons_cat_combined_whitenonwhite_White', 
                      'sex_F', 
                      'edu_bin_combined_nvq4/nvq5/degree or equivalent', # edu_bin_combined_All else
                      'imd_quintile_5',
                      'imd_decile_10.0',
                      'RUC_grouped_Rural',
                      
                      'PrePandemicEmploymentStatus_Earliest_Employed',
                      
                      'Vaccine_1_name_grouped_Pfizer BioNTech', 
                      'Vaccine_2_name_grouped_Pfizer BioNTech',
                      'Vaccine_3_name_grouped_Pfizer BioNTech',

                      'Result_Thriva_N_negative',
                      'Result_Thriva2_N_negative',
                      'NaturalInfection_WideCDC_Interpretation_MaxToDate_0. No evidence of natural infection', 
                      
                      'HadCovid_Ever_SelfReport_original_1. NoCovid',
                      'HadCovid_Ever_SelfReport_grouped_1. NoCovid',
                      'HadCovid_Ever_SelfReport_binary_1. NoCovid',
                      
                      'HadCovid_Ever_SelfReport_MaxToCoPE4_original_1. NoCovid',
                      'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped_1. NoCovid',
                      'HadCovid_Ever_SelfReport_MaxToCoPE4_binary_1. NoCovid',
                      
                      'SymptomDuration_Max_grouped_0.2 N/A - no covid',
                      'SymptomDuration_MaxToCoPE4_grouped_0.2 N/A - no covid',
                      'AffectedFunctioningDuration_Max_grouped_0.2 N/A - no covid',
                      'HospitalisedFlag_grouped_0.2 N/A - no covid',

                      'PrePandemicHealth_Earliest_original_5. Excellent',
                      'PrePandemicHealth_Earliest_grouped_5. Excellent', 
                      'PrePandemicHealth_Earliest_binary_3-5. Good, Very Good, Excellent',
   
                      'HADS_Anxiety_grouped_cat2_0-7, below threshold',
                      'HADS_Anxiety_grouped_cat3_0-7, below threshold',
                      'HADS_Depression_grouped_cat2_0-7, below threshold',
                      'HADS_Depression_grouped_cat3_0-7, below threshold',
                      'PHQ4_AnxietyandDepression_grouped_cat3_0-2, below threshold',
                      'PHQ4_AnxietyandDepression_grouped_cat2_0-2, below threshold',
                      
                      'PHQ4_Anxiety_grouped_cat2_0-2, below threshold', 'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2_0-2, below threshold',
                 'PHQ4_Depression_grouped_cat2_0-2, below threshold', 'PHQ4_Depression_grouped_MaxToCoPE4_cat2_0-2, below threshold',
                      
                      'HADS_Anxiety_grouped_MaxToCoPE4_cat2_0-7, below threshold',
                      'HADS_Anxiety_grouped_MaxToCoPE4_cat3_0-7, below threshold',
                      'HADS_Depression_grouped_MaxToCoPE4_cat2_0-7, below threshold',
                      'HADS_Depression_grouped_MaxToCoPE4_cat3_0-7, below threshold',
                      'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3_0-2, below threshold',
                      'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2_0-2, below threshold',
                      
                      'BMI_cat2_Non obese, < 30 kg/m2',
                      'BMI_cat5_2: 18.5-25',
                      
                      'SubDomain_DiabetesType1_No', 
                      'Domain_DiabetesType2_No',
                      'SubDomain_Hypertension_No',
                      'SubDomain_HighCholesterol_No',
                      'SubDomain_Asthma_No', 
                      'ShieldingFlag_no',
                      'FrailtyIndexOngoingCat_original_1. Healthy',
                      'FrailtyIndexOngoingCat_grouped_1. Healthy',
                      'Domain_CardiacDisease_No',
                      'Domain_CardiacRiskFactors_No',
                      'Domain_CancerAny_No', 
                      'Domain_NeurologicalDisease_No',
                      'Domain_SubjectiveMemoryImpairment_No',
                      'Domain_ChronicLungDisease_No',
                      'Domain_Arthritis_No',
                      'Domain_Osteoporosis_No',
                      'SubDomain_HeartFailure_No',
                      'SubDomain_AtrialFibrillation_No', 
                      'SubDomain_CoronaryHeartDisease_No',
                      'SubDomain_CancerLeukemia_No',
                      'SubDomain_CancerLung_No', 
                      'SubDomain_CancerLymphoma_No',
                      'SubDomain_AnxietyStressDisorder_No',
                      'SubDomain_Depression_No', 
                      'SubDomain_Stroke_No', 
                      'SubDomain_Alzheimers_No',
                      'SubDomain_RheumatoidArthritis_No', 
                      'SubDomain_Epilepsy_No', 
                      'SubDomain_LiverCirrhosis_No',
                      
                      'SubDomain_AnxietyOrDepression_No',
                      'SubDomain_DiabetesAny_No',
                      'SubDomain_HeartDiseaseOrFailure_No',
                      'SubDomain_MultimorbidityCount_Selected_Grouped_0',
                      'SubDomain_MultimorbidityCount_Grouped_0',
                      'MedicationFlag_NHSShielding_Immunosuppressant_No', 
                      'MedicationFlag_SevereAsthma_No',
                      'MedicationFlag_SevereCOPD_No',
                      'MedicationFlag_ShieldingListAny_No',
                      'MedicationFlag_ED_Immunosuppressant_No',
                      'ValueGrouped_50s_7. 250+ U/mL',
                      'ValueGrouped_100s_4. 250+ U/mL',
                      'ValueGrouped_Binary_2. 250+ U/mL',
                      'Value_decile_thriva1vacc1_9.0',
                      'Value_quintile_thriva1vacc1_5.0',
                      ]

# Drop reference variables from dummy variable list to create full list of dummy variables to use 
dummy_var_list_to_test = dummy_var_list_full.copy()
# [dummy_var_list_to_test.remove(x) for x in dummy_ref_var_list]
for x in dummy_ref_var_list:
    dummy_var_list_to_test.remove(x)
    
    


#%% Create list of 'NoDataAvailable' or 'NaN' dummy variables, to later use to filter out missing data
dummy_NaN_list = ['age_3_bands_NoDataAvailable',
                  'age_10yr_bands_NoDataAvailable',
                  'age_custom_NoDataAvailable',
                  'age_binary_NoDataAvailable',
                  
                  'ethnicity_ons_cat_combined_whitenonwhite_NoDataAvailable', 
                  'sex_NoDataAvailable', 
                  'edu_bin_combined_NoDataAvailable', 
                  'imd_quintile_NoDataAvailable', 
                  'imd_decile_NoDataAvailable', 
                  'RUC_grouped_NaN',
                  
                  # Employment
                  'PrePandemicEmploymentStatus_Earliest_0.0 Unknown - individual did not complete CoPE',
                  
                  'Vaccine_1_name_grouped_0.0 Unknown - individual did not complete CoPE', 
                  'Vaccine_2_name_grouped_0.0 Unknown - individual did not complete CoPE',
                  'Vaccine_3_name_grouped_0.0 Unknown - individual did not complete CoPE',
                  
                  'Result_Thriva_N_void',
                  'Result_Thriva2_N_void',
                  'NaturalInfection_WideCDC_Interpretation_MaxToDate_0. No antibody results to interpret', 'NaturalInfection_WideCDC_Interpretation_MaxToDate_1. Possible evidence of natural infection (N negative, S positive, Vaccination status unknown)', 
                  
                  'HadCovid_Ever_SelfReport_original_0.1 Unknown - Answer not provided in CoPE', 
                  'HadCovid_Ever_SelfReport_grouped_0.1 Unknown - Answer not provided in CoPE', 
                  'HadCovid_Ever_SelfReport_binary_0.1 Unknown - Answer not provided in CoPE',
                  'HadCovid_Ever_SelfReport_original_0.0 Unknown - individual did not complete CoPE',
                  'HadCovid_Ever_SelfReport_grouped_0.0 Unknown - individual did not complete CoPE',
                  'HadCovid_Ever_SelfReport_binary_0.0 Unknown - individual did not complete CoPE',
                  
                  'HadCovid_Ever_SelfReport_MaxToCoPE4_original_0.1 Unknown - Answer not provided in CoPE', 
                  'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped_0.1 Unknown - Answer not provided in CoPE', 
                  'HadCovid_Ever_SelfReport_MaxToCoPE4_binary_0.1 Unknown - Answer not provided in CoPE',
                  'HadCovid_Ever_SelfReport_MaxToCoPE4_original_0.0 Unknown - individual did not complete CoPE',
                  'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped_0.0 Unknown - individual did not complete CoPE',
                  'HadCovid_Ever_SelfReport_MaxToCoPE4_binary_0.0 Unknown - individual did not complete CoPE',
                  
                  
                  'SymptomDuration_Max_grouped_0.0 Unknown - individual did not complete CoPE', 'SymptomDuration_Max_grouped_0.1 Unknown - Answer not provided in CoPE', 
                  'SymptomDuration_MaxToCoPE4_grouped_0.0 Unknown - individual did not complete CoPE', 'SymptomDuration_MaxToCoPE4_grouped_0.1 Unknown - Answer not provided in CoPE', 
                  
                  
                  
                  'AffectedFunctioningDuration_Max_grouped_0.0 Unknown - individual did not complete CoPE', 'AffectedFunctioningDuration_Max_grouped_0.1 Unknown - Answer not provided in CoPE', 
                  'HospitalisedFlag_grouped_0.0 Unknown - individual did not complete CoPE', 'HospitalisedFlag_grouped_0.1 Unknown - Answer not provided in CoPE', 

                  'PrePandemicHealth_Earliest_original_0.0 Unknown - individual did not complete CoPE',#'PrePandemicHealth_Earliest_original_0.1 Unknown - Answer not provided in CoPE', 
                  'PrePandemicHealth_Earliest_grouped_0.0 Unknown - individual did not complete CoPE', #'PrePandemicHealth_Earliest_grouped_0.1 Unknown - Answer not provided in CoPE', 
                  'PrePandemicHealth_Earliest_binary_0.0 Unknown - individual did not complete CoPE', #'PrePandemicHealth_Earliest_grouped_0.1 Unknown - Answer not provided in CoPE', 

                  'HADS_Anxiety_grouped_cat2_0.0 Unknown - individual did not complete CoPE', 'HADS_Anxiety_grouped_cat2_1. Assessment incomplete', 'HADS_Anxiety_grouped_cat2_NoDataAvailable'
                  'HADS_Anxiety_grouped_cat3_0.0 Unknown - individual did not complete CoPE', 'HADS_Anxiety_grouped_cat3_1. Assessment incomplete', 'HADS_Anxiety_grouped_cat3_NoDataAvailable', 

                  'HADS_Depression_grouped_cat2_0.0 Unknown - individual did not complete CoPE', 'HADS_Depression_grouped_cat2_1. Assessment incomplete', 'HADS_Depression_grouped_cat2_NoDataAvailable', 
                  'HADS_Depression_grouped_cat3_0.0 Unknown - individual did not complete CoPE', 'HADS_Depression_grouped_cat3_1. Assessment incomplete', 'HADS_Depression_grouped_cat3_NoDataAvailable', 
                  #PHQ

                  'PHQ4_AnxietyandDepression_grouped_cat3_0.0 Unknown - individual did not complete CoPE', 'PHQ4_AnxietyandDepression_grouped_cat3_1. Assessment incomplete', 'PHQ4_AnxietyandDepression_grouped_cat3_NoDataAvailable', 
                  'PHQ4_AnxietyandDepression_grouped_cat2_0.0 Unknown - individual did not complete CoPE', 'PHQ4_AnxietyandDepression_grouped_cat2_1. Assessment incomplete', 'PHQ4_AnxietyandDepression_grouped_cat2_NoDataAvailable', 
                  
                  'PHQ4_Depression_grouped_cat2_NoDataAvailable', 
                  'PHQ4_Anxiety_grouped_cat2_NoDataAvailable', 
                  
                  # Max to CoPE 4 for thriva 1
                  'HADS_Anxiety_grouped_MaxToCoPE4_cat2_0.0 Unknown - individual did not complete CoPE', 'HADS_Anxiety_grouped_MaxToCoPE4_cat2_1. Assessment incomplete', 'HADS_Anxiety_grouped_MaxToCoPE4_cat2_NoDataAvailable', 
                  'HADS_Anxiety_grouped_MaxToCoPE4_cat3_0.0 Unknown - individual did not complete CoPE', 'HADS_Anxiety_grouped_MaxToCoPE4_cat3_1. Assessment incomplete', 'HADS_Anxiety_grouped_MaxToCoPE4_cat3_NoDataAvailable', 

                  'HADS_Depression_grouped_MaxToCoPE4_cat2_0.0 Unknown - individual did not complete CoPE', 'HADS_Depression_grouped_MaxToCoPE4_cat2_1. Assessment incomplete', 'HADS_Depression_grouped_MaxToCoPE4_cat2_NoDataAvailable', 
                  'HADS_Depression_grouped_MaxToCoPE4_cat3_0.0 Unknown - individual did not complete CoPE', 'HADS_Depression_grouped_MaxToCoPE4_cat3_1. Assessment incomplete', 'HADS_Depression_grouped_MaxToCoPE4_cat3_NoDataAvailable', 
                  #PHQ

                  'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3_0.0 Unknown - individual did not complete CoPE', 'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3_1. Assessment incomplete', 'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3_NoDataAvailable', 
                  'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2_0.0 Unknown - individual did not complete CoPE', 'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2_1. Assessment incomplete', 'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2_NoDataAvailable', 
                  
                  'PHQ4_Depression_grouped_MaxToCoPE4_cat2_NoDataAvailable', 
                  'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2_NoDataAvailable', 
                  
                  'BMI_cat2_NoDataAvailable',
                  'BMI_cat5_NoDataAvailable', 
                  
                  'SubDomain_DiabetesType1_NoDataAvailable', 
                  'Domain_DiabetesType2_NoDataAvailable', 
                  'SubDomain_Hypertension_NoDataAvailable', 
                  'SubDomain_HighCholesterol_NoDataAvailable', 
                  'SubDomain_Asthma_NoDataAvailable', 
                  'ShieldingFlag_0.0 Unknown - individual did not complete CoPE', 'ShieldingFlag_0.1 Unknown - Answer not provided in CoPE', 
                  'FrailtyIndexOngoingCat_original_NoDataAvailable', 
                  'FrailtyIndexOngoingCat_grouped_NoDataAvailable', 
                  'Domain_CardiacDisease_NoDataAvailable', 
                  'Domain_CardiacRiskFactors_NoDataAvailable', 
                  'Domain_CancerAny_NoDataAvailable', 
                  'Domain_NeurologicalDisease_NoDataAvailable', 
                  'Domain_SubjectiveMemoryImpairment_NoDataAvailable', 
                  'Domain_ChronicLungDisease_NoDataAvailable', 
                  'Domain_Arthritis_NoDataAvailable', 
                  'Domain_Osteoporosis_NoDataAvailable', 
                  'SubDomain_HeartFailure_NoDataAvailable', 
                  'SubDomain_AtrialFibrillation_NoDataAvailable',
                  'SubDomain_CoronaryHeartDisease_NoDataAvailable', 
                  'SubDomain_CancerLeukemia_NoDataAvailable', 
                  'SubDomain_CancerLung_NoDataAvailable', 
                  'SubDomain_CancerLymphoma_NoDataAvailable',
                  'SubDomain_AnxietyStressDisorder_NoDataAvailable',
                  'SubDomain_Depression_NoDataAvailable',
                  'SubDomain_Stroke_NoDataAvailable', 
                  'SubDomain_Alzheimers_NoDataAvailable',
                  'SubDomain_RheumatoidArthritis_NoDataAvailable',
                  'SubDomain_Epilepsy_NoDataAvailable',
                  'SubDomain_LiverCirrhosis_NoDataAvailable',
                  
                  'SubDomain_AnxietyOrDepression_NoDataAvailable',
                  'SubDomain_DiabetesAny_NoDataAvailable',
                  'SubDomain_HeartDiseaseOrFailure_NoDataAvailable',
                  'SubDomain_MultimorbidityCount_Selected_Grouped_NoDataAvailable',
                 
                  'SubDomain_MultimorbidityCount_Grouped_NoDataAvailable',
                  
                  'MedicationFlag_NHSShielding_Immunosuppressant_NoDataAvailable', 
                  'MedicationFlag_SevereAsthma_NoDataAvailable',
                  'MedicationFlag_SevereCOPD_NoDataAvailable',
                  'MedicationFlag_ShieldingListAny_NoDataAvailable',
                  'MedicationFlag_ED_Immunosuppressant_NoDataAvailable',
                  'ValueGrouped_50s_0. void',
                  'ValueGrouped_50s_NaN',
                  ]

# Drop NaN variables from dummy variable list to create list of dummy variables to use as input variables in models
for x in dummy_NaN_list:
    # print(x)
    if x in dummy_var_list_to_test:
        dummy_var_list_to_test.remove(x)



#%% Define analysis sample datasets (apply inclusion & exclusion criteria to filter datasets)
# -----------------------------------------------------------------------------
# Thriva 1, all tests
data_thriva1_all_tests = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           ].copy().reset_index()

# -----------------------------------------------------------------------------
# Thriva 1, all individuals who go on to have at least one vaccination recorded
data_thriva1_all_vaccinated = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['age'] >= 18)
                           & (combined_flat['Vaccine_Status_Final'].isin(['2.1 Vaccinated once', '2.2 Vaccinated twice', '2.3 Vaccinated 3 times']))
                           ].copy().reset_index()

# -----------------------------------------------------------------------------
# Thriva 1, 0 vaccinations
data_thriva1_0vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '0. Not Vaccinated')
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()

# -----------------------------------------------------------------------------
# Thriva 1, 1 vaccination
data_thriva1_1vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                           & (combined_flat['WeeksSinceVacc_1'] >= 4)
                           & (combined_flat['WeeksSinceVacc_1'] <= 11)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()

# -----------------------------------------------------------------------------
# Thriva 1, 2 vaccinations
data_thriva1_2vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.2 Vaccinated twice')
                           & (combined_flat['WeeksSinceVacc_2'] >= 2)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()
# exclude those with less than 21 days between 1st and 2nd vaccination - likely one date is erroneous 
min_days_between_vacc = 21
# Option 1 - exclude those with date difference less than limit - those with no 1st vaccination date are still included
data_thriva1_2vacc = data_thriva1_2vacc[~(((data_thriva1_2vacc['DaysSinceVacc_1'] - data_thriva1_2vacc['DaysSinceVacc_2']) < min_days_between_vacc))]


# -----------------------------------------------------------------------------
# Thriva 2, all individuals
data_thriva2_all = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()

data_thriva2_test = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['age'] >= 18)
                            & (combined_flat['Result'] != 'void')
                            & (combined_flat['Result_Vaccine_Status_CoPE5'].isin(['3 vaccines received', '2 vaccines received', 'Not vaccinated (and does not intend to be)', '1 vaccine received', 'Not vaccinated (does intend to be)']))
                            & (combined_flat['Vaccine_Status_Current'].isin([
                                                                               '0. Not Vaccinated',
                                                                                # '2.1 Vaccinated once', 
                                                                                '2.2 Vaccinated twice', 
                                                                                # '2.3 Vaccinated 3 times'
                                                                              ]))
                           ].copy().reset_index()


# -----------------------------------------------------------------------------
# Thriva 2, 0 vaccinations
data_thriva2_0vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '0. Not Vaccinated')
                           & ((combined_flat['Result_Vaccine_Status_CoPE5'].isin(['Not vaccinated (and does not intend to be)']))
                              |
                              ((combined_flat['Result_Vaccine_Status_CoPE5'].isin(['Not vaccinated (does intend to be)']))
                              & (combined_flat['DaysBetween_Thriva2_CoPE5'] <= 7))
                              )
                           & (combined_flat['age'] >= 18)
                           & (combined_flat['DaysBetween_Thriva2_CoPE5'] <= 7) # sampled for thriva 2 no more than 7 days after completing cope5, in case of vaccination shortly after cope5
                           ].copy().reset_index()

# -----------------------------------------------------------------------------
# Thriva 2, 1 vaccination
data_thriva2_1vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.1 Vaccinated once')
                           & (combined_flat['Result_Vaccine_Status_CoPE5'] == '1 vaccine received')
                           & (combined_flat['WeeksSinceVacc_1'] >= 4)
                           & (combined_flat['age'] >= 18)
                           & (combined_flat['DaysBetween_Thriva2_CoPE5'] <= 7) # sampled for thriva 2 no more than 7 days after completing cope5, in case of vaccination shortly after cope5
                           ].copy().reset_index()

# -----------------------------------------------------------------------------
# Thriva 2, 2 vaccinations
data_thriva2_2vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'].isin(['2.2 Vaccinated twice']))
                           & (combined_flat['Result_Vaccine_Status_CoPE5'].isin(['2 vaccines received','3 vaccines received']))
                            & (combined_flat['WeeksSinceVacc_2'] >= 2)
                           & (combined_flat['age'] >= 18)
                            & (combined_flat['DaysBetween_Thriva2_CoPE5'] <= 7) # sampled for thriva 2 no more than 7 days after completing cope5, in case of vaccination shortly after cope5
                           ].copy().reset_index()

test = data_thriva2_2vacc[['ItemDate','ItemDate_Vaccine_Status_CoPE5','DaysBetween_Thriva2_CoPE5','Value']]


# -----------------------------------------------------------------------------
# Thriva 2, 3 vaccinations
data_thriva2_3vacc = combined_flat[(combined_flat['StudyName'] == 'Thriva #2')
                           & (combined_flat['DataItem'] == 'Antibody_S')
                           & (combined_flat['Result'] != 'void')
                           & (combined_flat['Vaccine_Status_Current'] == '2.3 Vaccinated 3 times')
                           & (combined_flat['WeeksSinceVacc_3'] >= 2)
                           & (combined_flat['age'] >= 18)
                           ].copy().reset_index()


test_thriva1 = data_thriva1_all_tests[['Result','Result_Thriva_N']]
test_thriva2 = data_thriva2_all[['Result','Result_Thriva2_N']]





#%% Add outcome flags - low antibody levels & post-vaccination infection
# Set anti-S values to use as thresholds for outcome variable in logistic regression
antibodylevel_thriva1_1vacc_5pct = data_thriva1_1vacc['Value'].quantile(0.05)
antibodylevel_thriva1_1vacc_10pct = data_thriva1_1vacc['Value'].quantile(0.10)
antibodylevel_thriva1_1vacc_20pct = data_thriva1_1vacc['Value'].quantile(0.2)

antibodylevel_thriva1_2vacc_5pct = data_thriva1_2vacc['Value'].quantile(0.05)

antibodylevel_thriva2_2vacc_5pct = data_thriva2_2vacc['Value'].quantile(0.05)
antibodylevel_thriva2_2vacc_10pct = data_thriva2_2vacc['Value'].quantile(0.1)
antibodylevel_thriva2_2vacc_20pct = data_thriva2_2vacc['Value'].quantile(0.2)

antibodylevel_thriva2_3vacc_5pct = data_thriva2_3vacc['Value'].quantile(0.05)
antibodylevel_thriva2_3vacc_10pct = data_thriva2_3vacc['Value'].quantile(0.1)
antibodylevel_thriva2_3vacc_1pct = data_thriva2_3vacc['Value'].quantile(0.01)
antibodylevel_thriva2_3vacc_20pct = data_thriva2_3vacc['Value'].quantile(0.2)


thresh_Low_1Vacc = 10 # Roughly corresponds to bottom 10%, for 4-10 weeks after 1st vaccination
thresh_Low_2Vacc = 250 # Roughly corresponds to bottom 8%, for 2+ weeks after 2nd vaccination

# -----------------------------------------------------------------------------
# Thriva 1 - Response after 1 vaccination
# 5th percentile
data_thriva1_1vacc.loc[(data_thriva1_1vacc['Value'] <= antibodylevel_thriva1_1vacc_5pct)
                       , 'OutcomeFlag_thriva1_1vacc_5pct'] = 1
data_thriva1_1vacc['OutcomeFlag_thriva1_1vacc_5pct'] = data_thriva1_1vacc['OutcomeFlag_thriva1_1vacc_5pct'].fillna(0)

# 10th percentile
data_thriva1_1vacc.loc[(data_thriva1_1vacc['Value'] <= antibodylevel_thriva1_1vacc_10pct)
                       , 'OutcomeFlag_thriva1_1vacc_10pct'] = 1
data_thriva1_1vacc['OutcomeFlag_thriva1_1vacc_10pct'] = data_thriva1_1vacc['OutcomeFlag_thriva1_1vacc_10pct'].fillna(0)

# 20th percentile
data_thriva1_1vacc.loc[(data_thriva1_1vacc['Value'] <= antibodylevel_thriva1_1vacc_20pct)
                       , 'OutcomeFlag_thriva1_1vacc_20pct'] = 1
data_thriva1_1vacc['OutcomeFlag_thriva1_1vacc_20pct'] = data_thriva1_1vacc['OutcomeFlag_thriva1_1vacc_20pct'].fillna(0)


# -----------------------------------------------------------------------------
# Thriva 1 - Response after 2 vaccinations
# < 250 (8%)
data_thriva1_2vacc.loc[(data_thriva1_2vacc['Value'] < thresh_Low_2Vacc)
                       , 'OutcomeFlag_Low_2Vacc_Thriva1'] = 1
data_thriva1_2vacc['OutcomeFlag_Low_2Vacc_Thriva1'] = data_thriva1_2vacc['OutcomeFlag_Low_2Vacc_Thriva1'].fillna(0)

# 5th percentile
data_thriva1_2vacc.loc[(data_thriva1_2vacc['Value'] <= antibodylevel_thriva1_2vacc_5pct)
                       , 'OutcomeFlag_thriva1_2vacc_5pct'] = 1
data_thriva1_2vacc['OutcomeFlag_thriva1_2vacc_5pct'] = data_thriva1_2vacc['OutcomeFlag_thriva1_2vacc_5pct'].fillna(0)



# -----------------------------------------------------------------------------
# Thriva 2 - Response after 2 vaccination
# 5th percentile
data_thriva2_2vacc.loc[(data_thriva2_2vacc['Value'] <= antibodylevel_thriva2_2vacc_5pct)
                       , 'OutcomeFlag_thriva2_2vacc_5pct'] = 1
data_thriva2_2vacc['OutcomeFlag_thriva2_2vacc_5pct'] = data_thriva2_2vacc['OutcomeFlag_thriva2_2vacc_5pct'].fillna(0)
# 10th percentile
data_thriva2_2vacc.loc[(data_thriva2_2vacc['Value'] <= antibodylevel_thriva2_2vacc_10pct)
                       , 'OutcomeFlag_thriva2_2vacc_10pct'] = 1
data_thriva2_2vacc['OutcomeFlag_thriva2_2vacc_10pct'] = data_thriva2_2vacc['OutcomeFlag_thriva2_2vacc_10pct'].fillna(0)
# < 250 (20%)
data_thriva2_2vacc.loc[(data_thriva2_2vacc['Value'] <= thresh_Low_2Vacc)
                       , 'OutcomeFlag_Low_2Vacc_Thriva2'] = 1
data_thriva2_2vacc['OutcomeFlag_Low_2Vacc_Thriva2'] = data_thriva2_2vacc['OutcomeFlag_Low_2Vacc_Thriva2'].fillna(0)

# 20th percentile
data_thriva2_2vacc.loc[(data_thriva2_2vacc['Value'] <= antibodylevel_thriva2_2vacc_20pct)
                       , 'OutcomeFlag_thriva2_2vacc_20pct'] = 1
data_thriva2_2vacc['OutcomeFlag_thriva2_2vacc_20pct'] = data_thriva2_2vacc['OutcomeFlag_thriva2_2vacc_20pct'].fillna(0)


# -----------------------------------------------------------------------------
# Thriva 2 - Response after 3 vaccination
# 5th percentile
data_thriva2_3vacc.loc[(data_thriva2_3vacc['Value'] <= antibodylevel_thriva2_3vacc_5pct)
                       , 'OutcomeFlag_thriva2_3vacc_5pct'] = 1
data_thriva2_3vacc['OutcomeFlag_thriva2_3vacc_5pct'] = data_thriva2_3vacc['OutcomeFlag_thriva2_3vacc_5pct'].fillna(0)
# 10th percentile
data_thriva2_3vacc.loc[(data_thriva2_3vacc['Value'] <= antibodylevel_thriva2_3vacc_10pct)
                       , 'OutcomeFlag_thriva2_3vacc_10pct'] = 1
data_thriva2_3vacc['OutcomeFlag_thriva2_3vacc_10pct'] = data_thriva2_3vacc['OutcomeFlag_thriva2_3vacc_10pct'].fillna(0)

# 20th percentile
data_thriva2_3vacc.loc[(data_thriva2_3vacc['Value'] <= antibodylevel_thriva2_3vacc_20pct)
                       , 'OutcomeFlag_thriva2_3vacc_20pct'] = 1
data_thriva2_3vacc['OutcomeFlag_thriva2_3vacc_20pct'] = data_thriva2_3vacc['OutcomeFlag_thriva2_3vacc_20pct'].fillna(0)




#%% Generate outcome flags for post-vaccination infection
# -----------------------------------------------------------------------------
# Thriva 1, Once vaccinated - Identifying individuals with post-vaccination infection after Thriva #1
# Identify cases
data_thriva1_1vacc.loc[((data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('after Thriva #1')))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = 1

# Fill blanks with 0
data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAfterThriva1'] = data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAfterThriva1'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = np.nan
# Exclude individuals who had a post-vaccination infection prior to Thriva #1 - they shouldn't be considered controls
data_thriva1_1vacc.loc[((data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('before Thriva #1')))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = np.nan

test = data_thriva1_1vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAfterThriva1']].groupby(['StudyNumber']).max()


# -----------------------------------------------------------------------------
# Thriva 1, Once vaccinated - Identifying individuals with post-vaccination infection at any time (so including before Thriva #1 as well as after)
# Identify cases
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        | (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        | (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        | (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime'] = 1

# Fill blanks with 0
data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime'] = data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime'] = np.nan

test = data_thriva1_1vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime']].groupby(['StudyNumber']).max()


### Unknown vaccination status at time of infection
# Identify cases
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = 1

# Fill blanks with 0
data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = np.nan

test = data_thriva1_1vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine']].groupby(['StudyNumber']).max()

### Once vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = 1

# Fill blanks with 0
data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = np.nan

test = data_thriva1_1vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine']].groupby(['StudyNumber']).max()

### Twice vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = 1

# Fill blanks with 0
data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = np.nan

test = data_thriva1_1vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine']].groupby(['StudyNumber']).max()

### 3 times vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = 1

# Fill blanks with 0
data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = data_thriva1_1vacc['OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_1vacc.loc[(data_thriva1_1vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_1vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = np.nan

test = data_thriva1_1vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine']].groupby(['StudyNumber']).max()


# -----------------------------------------------------------------------------
# Thriva 1, Twice vaccinated - Identifying individuals with post-vaccination infection after Thriva #1
# Identify cases
data_thriva1_2vacc.loc[((data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('after Thriva #1')))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = 1

# Fill blanks with 0
data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAfterThriva1'] = data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAfterThriva1'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = np.nan
# Exclude individuals who had a post-vaccination infection prior to Thriva #1 - they shouldn't be considered controls
data_thriva1_2vacc.loc[((data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('before Thriva #1')))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = np.nan

test = data_thriva1_2vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAfterThriva1']].groupby(['StudyNumber']).max()


# -----------------------------------------------------------------------------
# Thriva 1, Twice vaccinated - Identifying individuals with post-vaccination infection at any time (so including before Thriva #1 as well as after)
# Identify cases
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        | (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        | (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        | (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime'] = 1

# Fill blanks with 0
data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime'] = data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime'] = np.nan

test = data_thriva1_2vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime']].groupby(['StudyNumber']).max()


### Unknown vaccination status at time of infection
# Identify cases
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = 1

# Fill blanks with 0
data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = np.nan

test = data_thriva1_2vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine']].groupby(['StudyNumber']).max()

### Once vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = 1

# Fill blanks with 0
data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = np.nan

test = data_thriva1_2vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine']].groupby(['StudyNumber']).max()

### Twice vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = 1

# Fill blanks with 0
data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = np.nan

test = data_thriva1_2vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine']].groupby(['StudyNumber']).max()

### 3 times vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = 1

# Fill blanks with 0
data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = data_thriva1_2vacc['OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_2vacc.loc[(data_thriva1_2vacc['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_2vacc['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = np.nan

test = data_thriva1_2vacc[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine']].groupby(['StudyNumber']).max()



# -----------------------------------------------------------------------------
# Thriva 1, All who go on to be vaccinated - Identifying individuals with post-vaccination infection after Thriva #1
# Identify cases
data_thriva1_all_vaccinated.loc[((data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('after Thriva #1')))
                        |
                        ((data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('after Thriva #1')))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = 1

# Fill blanks with 0
data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAfterThriva1'] = data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAfterThriva1'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = np.nan
# Exclude individuals who had a post-vaccination infection prior to Thriva #1 - they shouldn't be considered controls
data_thriva1_all_vaccinated.loc[((data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('before Thriva #1')))
                        |
                        ((data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('before Thriva #1')))
                       , 'OutcomeFlag_PostVaccInfectionAfterThriva1'] = np.nan

test = data_thriva1_all_vaccinated[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAfterThriva1']].groupby(['StudyNumber']).max()

# -----------------------------------------------------------------------------
# Thriva 1, All who go on to be vaccinated - Identifying individuals with post-vaccination infection at any time (so including before Thriva #1 as well as after)

### Unknown vaccination status at time of infection
# Identify cases
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = 1

# Fill blanks with 0
data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine'] = np.nan

test = data_thriva1_all_vaccinated[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine']].groupby(['StudyNumber']).max()

### Once vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = 1

# Fill blanks with 0
data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine'] = np.nan

test = data_thriva1_all_vaccinated[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine']].groupby(['StudyNumber']).max()

### Twice vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = 1

# Fill blanks with 0
data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine'] = np.nan

test = data_thriva1_all_vaccinated[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine']].groupby(['StudyNumber']).max()

### 3 times vaccinated vaccination status at time of infection
# Identify cases
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = 1

# Fill blanks with 0
data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine'] = np.nan

test = data_thriva1_all_vaccinated[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine']].groupby(['StudyNumber']).max()

### Any vaccination status at time of infection
# Identify cases
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('1.0'))
                        | (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        | (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                        | (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('1.1|1.2|2.1|2.2|3.|4.'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime'] = 1

# Fill blanks with 0
data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime'] = data_thriva1_all_vaccinated['OutcomeFlag_PostVaccInfectionAnyTime'].fillna(0)

# Set individuals where all flags show insufficient data to NaN
data_thriva1_all_vaccinated.loc[(data_thriva1_all_vaccinated['PostVaccInfection_UnknownVaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_1Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_2Vaccine_Flag'].str.contains('0.0'))
                        & (data_thriva1_all_vaccinated['PostVaccInfection_3Vaccine_Flag'].str.contains('0.0'))
                       , 'OutcomeFlag_PostVaccInfectionAnyTime'] = np.nan

test = data_thriva1_all_vaccinated[['StudyNumber','PostVaccInfection_UnknownVaccine_Flag', 'PostVaccInfection_1Vaccine_Flag', 'PostVaccInfection_2Vaccine_Flag', 'PostVaccInfection_3Vaccine_Flag', 'OutcomeFlag_PostVaccInfectionAnyTime']].groupby(['StudyNumber']).max()



#%% Combine Thriva 1 , 0+1+2 vaccinations datasets
# -----------------------------------------------------------------------------
# Thriva 1, 1+2 vaccinations combined, for testing post-vaccination infection odds
data_thriva1_1and2vacc = data_thriva1_1vacc.append(data_thriva1_2vacc)




#%% Mapping shorthand variable names to 'tidy' names for plotting
codebook['TidyVariableNames'] = {
                             
                             # Sociodemographics
                             'age_3_bands_2: 50-70': 'Age: 50-69',
                             'age_3_bands_3: 70+': 'Age: 70+',
                             'age_custom_5: 50-60': 'Age: 50-59',
                             'age_custom_6: 60-70': 'Age: 60-69',
                             'age_custom_7: 70-80': 'Age: 70-79',
                             'age_custom_8: 80+': 'Age: 80+',
                             'age_binary_2: 50+': 'Age: 50+',
                             'age': 'Age: +1 year',
                             'sex_M':'Sex: Male',
                             'ethnicity_ons_cat_combined_whitenonwhite_Other than White':'Ethnicity: Other than white',
                             
                             'edu_bin_combined_nvq4/nvq5/degree or equivalent': 'Level of education: NVQ level 4 or higher',
                             'edu_bin_combined_All else': 'Level of education: NVQ level 3 or lower',
                             'imd_decile_reverse' : 'IMD: -1 decile (increasing deprivation)',
                             'imd_quintile_reverse': 'IMD: -1 quintile (increasing deprivation)',
                             'imd_quintile_1': 'Deprivation (IMD): Quintile 1 (most deprived 20%)',
                             'imd_quintile_2': 'Deprivation (IMD): Quintile 2',
                             'imd_quintile_3': 'Deprivation (IMD): Quintile 3',
                             'imd_quintile_4': 'Deprivation (IMD): Quintile 4',
                             
                             'RUC_grouped_Urban': 'RUC: Urban',
                             
                             'PrePandemicEmploymentStatus_Earliest_In education at school/college/university, or in an apprenticeship': 'Employment status: In education',
                             'PrePandemicEmploymentStatus_Earliest_In unpaid/ voluntary work': 'Employment status: Unpaid/voluntary work',
                             'PrePandemicEmploymentStatus_Earliest_Looking after home or family': 'Employment status: Looking after home or family (unpaid care)',
                             'PrePandemicEmploymentStatus_Earliest_Other': 'Employment status: Other',
                             'PrePandemicEmploymentStatus_Earliest_Permanently (or long-term) sick or disabled': 'Employment status: Permanently (or long-term) sick or disabled',
                             'PrePandemicEmploymentStatus_Earliest_Retired': 'Employment status: Retired',
                             'PrePandemicEmploymentStatus_Earliest_Self-employed': 'Employment status: Self-employed',
                             'PrePandemicEmploymentStatus_Earliest_Semi-retired/part-time employment': 'Employment status: Semi-retired/part-time employment',
                             'PrePandemicEmploymentStatus_Earliest_Maternity leave': 'Employment status: Maternity leave',
                             'PrePandemicEmploymentStatus_Earliest_Unemployed': 'Employment status: Unemployed',
                             
                             # Infection related 
                             'Result_Thriva_N_positive':'Anti-N antibody status, Q2: Positive',
                             'Result_Thriva2_N_positive':'Anti-N antibody status, Q4: Positive',
                             'NaturalInfection_WideCDC_Interpretation_MaxToDate_2. Evidence of natural infection':'COVID-19 infection status (serology-based): Evidence of natural infection',
                             
                             'HadCovid_Ever_SelfReport_grouped_2.0 Unsure': 'COVID-19 infection status (self-reported): Unsure',
                             'HadCovid_Ever_SelfReport_grouped_2.1 SuspectedCovid': 'COVID-19 infection status (self-reported): Suspected case',
                             'HadCovid_Ever_SelfReport_grouped_3. PositiveCovid': 'COVID-19 infection status (self-reported): Confirmed case',
                             'HadCovid_Ever_SelfReport_binary_2-3. SuspectedOrPositiveCovid': 'COVID-19 infection status (self-reported): Suspected or confirmed case',
                             
                             'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped_2.0 Unsure': 'COVID-19 infection status (self-reported): Unsure',
                             'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped_2.1 SuspectedCovid': 'COVID-19 infection status (self-reported): Suspected case',
                             'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped_3. PositiveCovid': 'COVID-19 infection status (self-reported): Confirmed case',
                             'HadCovid_Ever_SelfReport_MaxToCoPE4_binary_2-3. SuspectedOrPositiveCovid': 'COVID-19 infection status (self-reported): Suspected or confirmed case',
                             
                             'SymptomDuration_Max_grouped_1. No symptoms': 'Symptom duration (self-reported): No symptoms',
                             'SymptomDuration_Max_grouped_2: 0-4 weeks': 'Symptom duration (self-reported): 0-4 weeks',
                             'SymptomDuration_Max_grouped_4: 4+ weeks': 'Symptom duration (self-reported): 4+ weeks',
                             
                             'SymptomDuration_MaxToCoPE4_grouped_1. No symptoms': 'Symptom duration (self-reported): No symptoms',
                             'SymptomDuration_MaxToCoPE4_grouped_2: 0-4 weeks': 'Symptom duration (self-reported): 0-4 weeks',
                             'SymptomDuration_MaxToCoPE4_grouped_4: 4+ weeks': 'Symptom duration (self-reported): 4+ weeks',
                             
                             # General health, frailty, multimorbidity
                             'BMI_cat5_1: 0-18.5': 'BMI: 0-18.5 kg/m^2',
                             'BMI_cat5_3: 25-30': 'BMI: 25-30 kg/m^2',
                             'BMI_cat5_4: 30-35': 'BMI: 30-35 kg/m^2',
                             'BMI_cat5_5: 35+': 'BMI: 35+ kg/m^2',
                             'BMI_cat2_Obese, >= 30 kg/m2': 'BMI: Obese, >= 30 kg/m^2',
                             'PrePandemicHealth_Earliest_binary_1,2. Poor, Fair': 'Self-rated health: Poor, Fair',
                             'PrePandemicHealth_Earliest_original_1. Poor':'Self-rated health: 1. Poor',
                             'PrePandemicHealth_Earliest_original_2. Fair':'Self-rated health: 2. Fair',
                             'PrePandemicHealth_Earliest_original_3. Good':'Self-rated health: 3. Good',
                             'PrePandemicHealth_Earliest_original_4. Very Good':'Self-rated health: 4. Very Good',
                             'ShieldingFlag_yes': 'Advised on "shielding list": Yes',
                             'FrailtyIndexOngoingCat_original_2. Pre-frail': 'Frailty Index: 2. Pre-frail',
                             'FrailtyIndexOngoingCat_original_3. Frail': 'Frailty Index: 3. Frail',
                             'FrailtyIndexOngoingCat_original_4. Very frail': 'Frailty Index: 4. Very frail',
                             'FrailtyIndexOngoingCat_Ordinal': 'Frailty Index: +1 Category',       
                             
                             'BMI': 'BMI: +1 kg/m^2',
                             'PrePandemicHealth_Earliest_ordinal': 'Self-rated health: -1 decreasing health',
                            
                             'MedicationFlag_ShieldingListAny_Yes': 'Prescribed "NHS shielding list" medication: Yes',
                            
                             'SubDomain_MultimorbidityCount_Ungrouped': 'Number of comorbidities (ALL): +1',
                             'SubDomain_MultimorbidityCount_Grouped_1-3': 'Number of comorbidities (ALL): 1-3',
                             'SubDomain_MultimorbidityCount_Grouped_4+': 'Number of comorbidities (ALL): 4+',
                            
                             
                             'SubDomain_MultimorbidityCount_Selected_Ungrouped': '# Selected comorbidities: +1',
                             'SubDomain_MultimorbidityCount_Selected_Grouped_1': '# Selected comorbidities: 1',
                             'SubDomain_MultimorbidityCount_Selected_Grouped_2': '# Selected comorbidities: 2',
                             'SubDomain_MultimorbidityCount_Selected_Grouped_3': '# Selected comorbidities: 3',
                             'SubDomain_MultimorbidityCount_Selected_Grouped_4': '# Selected comorbidities: 4',
                             'SubDomain_MultimorbidityCount_Selected_Grouped_5': '# Selected comorbidities: 5',
                             'SubDomain_MultimorbidityCount_Selected_Grouped_4+': '# Selected comorbidities: 4+',
                             
                             
                             # Comborbidity domains
                             'Domain_CardiacDisease_Yes':'Comorbidity domain: Cardiac Disease',
                             'Domain_CardiacRiskFactors_Yes':'Comorbidity domain: Cardiac Risk Factors',
                             'Domain_CancerAny_Yes':'Comorbidity: Cancer (any)',
                             'Domain_NeurologicalDisease_Yes':'Comorbidity domain: Neurological Disease',
                             'Domain_SubjectiveMemoryImpairment_Yes':'Comorbidity domain: Subjective Memory Impairment',
                             'Domain_Arthritis_Yes': 'Comorbidity domain: Arthritis (any)',
                             
                             # Individual comorbidities
                             'SubDomain_Hypertension_Yes': 'Comorbidity: Hypertension',
                             'SubDomain_HighCholesterol_Yes': 'Comorbidity: High Cholesterol', 
                             'Domain_ChronicLungDisease_Yes': 'Comorbidity: Chronic Lung Disease', 
                             'SubDomain_AtrialFibrillation_Yes': 'Comorbidity: Atrial Fibrillation',
                             'SubDomain_CoronaryHeartDisease_Yes': 'Comorbidity: Coronary Heart Disease',
                             'Domain_DiabetesType2_Yes': 'Comorbidity: Diabetes (Type II)',
                             
                             'SubDomain_Asthma_Yes': 'Comorbidity: Asthma',
                             'SubDomain_AnxietyStressDisorder_Yes': 'Comorbidity: Anxiety or Stress Disorder',
                             'SubDomain_Depression_Yes': 'Comorbidity: Depression',
                             'SubDomain_Stroke_Yes': 'Comorbidity: Stroke',
                             'SubDomain_Alzheimers_Yes': "Comorbidity: Alzheimers",
                             'Domain_Osteoporosis_Yes': 'Comorbidity: Osteoporosis',
                             'SubDomain_RheumatoidArthritis_Yes': 'Comorbidity: Rheumatoid Arthritis',
                             'SubDomain_Epilepsy_Yes': 'Comorbidity: Epilepsy',
                             
                             'SubDomain_AnxietyOrDepression_Yes': "Comorbidity: Anxiety or Depression",
                             'SubDomain_DiabetesAny_Yes': "Comorbidity: Diabetes (any)",
                             'SubDomain_HeartDiseaseOrFailure_Yes': "Comorbidity: Heart disease (CHD or Heart failure)",
                             
                             # Mental health assessments 
                             # HADS
                             'HADS_Anxiety_grouped_cat2_8+, above threshold': 'Anxiety (HADS): Above threshold',
                             'HADS_Depression_grouped_cat2_8+, above threshold': 'Depression (HADS): Above threshold',
                             
                             'HADS_Anxiety_grouped_cat3_11+, moderate, severe': 'Anxiety (HADS): Above threshold, 11+',
                             'HADS_Anxiety_grouped_cat3_8-10, mild': 'Anxiety (HADS): Above threshold, 8-10',
                             'HADS_Depression_grouped_cat3_11+, moderate, severe': 'Depression (HADS): Above threshold, 11+',
                             'HADS_Depression_grouped_cat3_8-10, mild': 'Depression (HADS): Above threshold, 8-10',
                             
            
                             'HADS_Anxiety_grouped_MaxToCoPE4_cat2_8+, above threshold': 'Anxiety (HADS): Above threshold',
                             'HADS_Depression_grouped_MaxToCoPE4_cat2_8+, above threshold': 'Depression (HADS): Above threshold',                                           
                             'HADS_Anxiety_grouped_MaxToCoPE4_cat3_11+, moderate, severe': 'Anxiety (HADS): Above threshold, 11+',
                             'HADS_Anxiety_grouped_MaxToCoPE4_cat3_8-10, mild': 'Anxiety (HADS): Above threshold, 8-10',
                             'HADS_Depression_grouped_MaxToCoPE4_cat3_11+, moderate, severe': 'Depression (HADS): Above threshold, 11+',
                             'HADS_Depression_grouped_MaxToCoPE4_cat3_8-10, mild': 'Depression (HADS): Above threshold, 8-10',
                             
                             'HADS_Anxiety_Value_Max': 'Anxiety (HADS) score: +1 score',
                             'HADS_Depression_Value_Max': 'Depression (HADS): +1 score',
                             
                             'HADS_Anxiety_Value_MaxToCoPE4': 'Anxiety (HADS) score: +1 score',
                             'HADS_Depression_Value_MaxToCoPE4': 'Depression (HADS): +1 score',
                             
                             # PHQ-4                             
                             'PHQ4_Anxiety_grouped_cat2_3+, above threshold': 'Anxiety (GAD-2): Above threshold',
                             'PHQ4_Depression_grouped_cat2_3+, above threshold': 'Depression (PHQ-2): Above threshold',
                             'PHQ4_AnxietyandDepression_grouped_cat2_3+, above threshold': 'Anxiety & Depression (PHQ-4): Above threshold',
                             'PHQ4_AnxietyandDepression_grouped_cat3_6+, moderate, severe': 'Anxiety & Depression (PHQ-4): Above threshold, 6+',
                             'PHQ4_AnxietyandDepression_grouped_cat3_3-5, mild': 'Anxiety & Depression (PHQ-4): Above threshold, 3-5',
                             
                             'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2_3+, above threshold': 'Anxiety (GAD-2): Above threshold',
                             'PHQ4_Depression_grouped_MaxToCoPE4_cat2_3+, above threshold': 'Depression (PHQ-2): Above threshold',
                             'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2_3+, above threshold': 'Anxiety & Depression (PHQ-4): Above threshold',
                             'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3_6+, moderate, severe': 'Anxiety & Depression (PHQ-4): Above threshold, 6+',
                             'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3_3-5, mild': 'Anxiety & Depression (PHQ-4): Above threshold, 3-5',
                             
                             'PHQ4_Anxiety_Value_Max': 'Anxiety (GAD-2) score: +1 score',
                             'PHQ4_Depression_Value_Max': 'Depression (PHQ-2) score: +1 score',
                             'PHQ4_AnxietyandDepression_Value_Max': 'Anxiety & Depression (PHQ-4) score: +1 score',
                            
                             'PHQ4_Anxiety_Value_MaxToCoPE4': 'Anxiety (GAD-2) score: +1 score',
                             'PHQ4_Depression_Value_MaxToCoPE4': 'Depression (PHQ-2) score: +1 score',
                             'PHQ4_AnxietyandDepression_Value_MaxToCoPE4': 'Anxiety & Depression (PHQ-4) score: +1 score',
                             
                             
                             # Vaccination related
                             'WeeksSinceVacc_1': 'Weeks since vaccination 1: +1 week',
                             'Vaccine_1_name_grouped_Oxford AstraZeneca': 'Vaccine 1 received: Oxford/AZ',
                             'Vaccine_1_name_grouped_Other': 'Vaccine 1 received: Other',
                             
                             'WeeksSinceVacc_2': 'Weeks since vaccination 2: +1 week',
                             'Vaccine_2_name_grouped_Oxford AstraZeneca': 'Vaccine 2 received: Oxford/AZ',
                             'Vaccine_2_name_grouped_Other': 'Vaccine 2 received: Other',
                             
                             'WeeksSinceVacc_3': 'Weeks since vaccination 3: +1 week',
                             'Vaccine_3_name_grouped_Moderna': 'Vaccine 3 received: Moderna',
                             'Vaccine_3_name_grouped_Other': 'Vaccine 3 received: Other',
                             
                             'MedicationFlag_ED_Immunosuppressant_Yes': 'Prescribed immunosuppressant medication',
                             
                             # Antibody level related
                             'ValueGrouped_50s_1. 0.4-0.8 U/mL': 'Q2 Antibody level: 1. 0.4-0.8 BAU/mL',
                             'ValueGrouped_50s_2. 0.8-25 U/mL': 'Q2 Antibody level: 2. 0.8-25 BAU/mL',
                             'ValueGrouped_50s_2.2 25-50 U/mL': 'Q2 Antibody level: 3. 25-50 BAU/mL',
                             'ValueGrouped_50s_3. 50-100 U/mL': 'Q2 Antibody level: 4. 50-100 BAU/mL',
                             'ValueGrouped_50s_4. 100-150 U/mL': 'Q2 Antibody level: 5. 100-150 BAU/mL',
                             'ValueGrouped_50s_5. 150-200 U/mL': 'Q2 Antibody level: 6. 150-200 BAU/mL',
                             'ValueGrouped_50s_6. 200-250 U/mL': 'Q2 Antibody level: 7. 200-250 BAU/mL',
                             
                             'ValueGrouped_100s_1. 0.4-50 U/mL': 'Q2 Antibody level: 1. 0.4-50 BAU/mL',
                             'ValueGrouped_100s_2. 50-150 U/mL': 'Q2 Antibody level: 2. 50-150 BAU/mL',
                             'ValueGrouped_100s_3. 150-250 U/mL': 'Q2 Antibody level: 3. 150-250 BAU/mL',
                             
                             'Value_quintile_thriva1vacc1_1.0': 'Q2 Antibody level: Quintile 1 (lowest 20%)',
                             'Value_quintile_thriva1vacc1_2.0': 'Q2 Antibody level: Quintile 2',
                             'Value_quintile_thriva1vacc1_3.0': 'Q2 Antibody level: Quintile 3',
                             'Value_quintile_thriva1vacc1_4.0': 'Q2 Antibody level: Quintile 4',
                             
                             'Value_decile_thriva1vacc1_1.0': 'Q2 Antibody level: Decile 1 (lowest 10%)',
                             'Value_decile_thriva1vacc1_2.0': 'Q2 Antibody level: Decile 2',
                             'Value_decile_thriva1vacc1_3.0': 'Q2 Antibody level: Decile 3',
                             'Value_decile_thriva1vacc1_4.0': 'Q2 Antibody level: Decile 4',
                             'Value_decile_thriva1vacc1_5.0': 'Q2 Antibody level: Decile 5',
                             'Value_decile_thriva1vacc1_6.0': 'Q2 Antibody level: Decile 6',
                             'Value_decile_thriva1vacc1_7.0': 'Q2 Antibody level: Decile 7',
                             'Value_decile_thriva1vacc1_8.0': 'Q2 Antibody level: Decile 8',
                             
                             'ValueGrouped_Binary_1. 0.4-250 U/mL': 'Q2 Antibody level: 1. 0.4-250 BAU/mL',
                             '': '',
                             
                             }


codebook['table_characteristics_vars'] = {'Value':'Anti-S antibody level value (BAU/mL)', 
                                          'age':'Age (years)', 
                                          'sex':'Sex',
                                          'ethnicity_ons_cat_combined_whitenonwhite':'Ethnicity',
                                          'imd_decile':'Local area deprivation, IMD decile',
                                          'imd_binary':'Local area deprivation, IMD',
                                          'edu_bin_combined':'Level of education',
                                          'WeeksSinceVacc_1':'Weeks since vaccination 1',
                                          'Vaccine_1_name_grouped':'Vaccine 1 received',
                                          'NaturalInfection_WideCDC_Interpretation_MaxToDate':'COVID-19 infection status (serology-based) at time of antibody testing',
                                          'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped':'COVID-19 infection status (self-reported), Q2',
                                          'HadCovid_Ever_SelfReport_grouped':'COVID-19 infection status (self-reported), Q4',
                                          
                                          'SymptomDuration_MaxToCoPE4_grouped':'Symptom duration (self-reported), Q2',
                                          'SymptomDuration_Max_grouped':'Symptom duration (self-reported), Q4',
                                          'Result_Thriva_N':'Anti-N antibody status, Q2',
                                          'Result_Thriva2_N':'Anti-N antibody status, Q4',
                                          'Result':'Anti-S antibody status',
                                          'FrailtyIndexOngoingCat_original':'Frailty Index',
                                          'ShieldingFlag':'Advised on "shielding list"',
                                          'MedicationFlag_ED_Immunosuppressant':'Prescribed immunosuppressant medication',
                                          'PrePandemicHealth_Earliest_binary':'Self-rated health',
                                          'BMI':'BMI',
                                          'SubDomain_MultimorbidityCount_Selected_Binary':'Number of selected comorbidities',
                                          'SubDomain_AnxietyStressDisorder':'Comorbidity: Anxiety or Stress Disorder',
                                          'SubDomain_Depression':'Comorbidity: Depression',
                                          'HADS_Anxiety_Value_MaxToCoPE4':'Anxiety (HADS) score up to Q2',
                                          'HADS_Anxiety_grouped_MaxToCoPE4_cat3':'Anxiety (HADS) score up to Q2',
                                          'HADS_Depression_Value_MaxToCoPE4':'Depression (HADS) score up to Q2',
                                          'HADS_Depression_grouped_MaxToCoPE4_cat3':'Depression (HADS) score up to Q2',
                                          'HADS_Anxiety_Value_Max':'Anxiety (HADS) score up to Q4',
                                          'HADS_Anxiety_grouped_cat3':'Anxiety (HADS) score up to Q4',
                                          'HADS_Depression_Value_Max':'Depression (HADS) score up to Q4',
                                          'HADS_Depression_grouped_cat3':'Depression (HADS) score up to Q4',
                                          'WeeksSinceVacc_2':'Weeks since vaccination 2',
                                          'Vaccine_2_name_grouped':'Vaccine 2 received',
                                          'WeeksSinceVacc_3':'Weeks since vaccination 3',
                                          'Vaccine_3_name_grouped':'Vaccine 3 received',
                                          
                                          'age_3_bands':'Age group',
                                          
                                          
                                          'OutcomeFlag_PostVaccInfectionAfterThriva1':'Post-vaccination infection between Q2 and Q4 testing',
                                          'OutcomeFlag_PostVaccInfectionAnyTime':'Post-vaccination infection at any time',
                                          'OutcomeFlag_PostVaccInfectionAnyTime_UnknownVaccine':'Post-vaccination infection, exact date of infection unknown',
                                          'OutcomeFlag_PostVaccInfectionAnyTime_1Vaccine':'Post-vaccination infection while once vaccinated',
                                          'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine':'Post-vaccination infection while twice vaccinated',
                                          'OutcomeFlag_PostVaccInfectionAnyTime_3Vaccine':'Post-vaccination infection while 3 times vaccinated',
                                          'WeeksBetween_Vacc1andPostVaccInfection':'Post-vaccination infection while once vaccinated: Weeks since 1st vaccination',
                                          'WeeksBetween_Vacc2andPostVaccInfection':'Post-vaccination infection while twice vaccinated: Weeks since 2nd vaccination',
                                          'WeeksBetween_Vacc3andPostVaccInfection':'Post-vaccination infection while 3 times vaccinated: Weeks since 3rd vaccination',
                                          'DaysBetween_Vacc1andPostVaccInfection':'Post-vaccination infection while once vaccinated: Days since 1st vaccination',
                                          'DaysBetween_Vacc2andPostVaccInfection':'Post-vaccination infection while twice vaccinated: Days since 2nd vaccination',
                                          'DaysBetween_Vacc3andPostVaccInfection':'Post-vaccination infection while 3 times vaccinated: Days since 3rd vaccination',
                                          'PostVaccInfection_1Vaccine_LikelyVariant':'Post-vaccination infection while once vaccinated: Date & likely variant',
                                          'PostVaccInfection_2Vaccine_LikelyVariant':'Post-vaccination infection while twice vaccinated: Date & likely variant',
                                          'PostVaccInfection_3Vaccine_LikelyVariant':'Post-vaccination infection while 3 times vaccinated: Date & likely variant',
                                          
                                          'age_10yr_bands':'Age group',
                                          'imd_quintile':'Local area deprivation, IMD quintile',
                                          
                                           'RUC_grouped':'RUC',
                                           'StudyName':'Overall',
                                           'ACTUAL_ZYGOSITY': 'Zygosity'
                                           # '':'',
                                           # '':'',
                                          }



codebook['table_characteristics_cats'] = {'':np.nan, 
                                          'M':'Male',
                                          'Other than White':'Other than White',
                                          'MostDeprived 40% (Decile 1-4)':'Most deprived 40% (decile 1-4)',
                                          'All else':'NVQ level 3 or lower',
                                          'Oxford AstraZeneca':'Oxford/AZ', 
                                          'Pfizer BioNTech':'Pfizer BioNTech',
                                          'Other':'Other',
                                          '2. Evidence of natural infection':'Evidence of natural infection',
                                          '2.0 Unsure':'Unsure',
                                          '2.1 SuspectedCovid':'Suspected case',
                                          '3. PositiveCovid':'Confirmed case',
                                          '1. No symptoms':'No symptoms',
                                          '2: 0-4 weeks':'0-4 weeks',
                                          '4: 4-12 weeks':'4-12 weeks',
                                          '5: 12+ weeks':'12+ weeks',
                                          'positive':'Positive',
                                          '3. Frail':'Frail',
                                          '4. Very frail':'Very frail',
                                          'yes':'Yes',
                                          'Yes':'Yes',
                                          '1,2. Poor, Fair':'Poor, Fair',
                                          '1+':'1+',
                                          '8-10, mild':'8-10, mild',
                                          '11+, moderate, severe':'11+, moderate, severe',
                                          'Moderna':'Moderna',
                                          
                                          '1. Poor': '1. Poor',
                                          '2. Fair': '2. Fair',
                                          '3. Good': '3. Good',
                                          '4. Very Good': '4. Very Good',
                                          '5. Excellent': '5. Excellent',
                                          
                                          'no':'No',
                                          '1. NoCovid':'No infection',
                                          '0. No evidence of natural infection':'No evidence of natural infection',
                                          'No':'No',
                                          'White':'White',
                                          '1. Healthy':'Healthy',
                                          '2. Pre-frail':'Pre-frail',
                                          '3,4. Frail, Very Frail':'Frail, Very Frail',
                                          'nvq4/nvq5/degree or equivalent':'NVQ level 4 or higher',
                                          'LeastDeprived 60% (Decile 5-10)':'Least deprived 60% (decile 5-10)',
                                          '0':'0',
                                          '3-5. Good, Very Good, Excellent':'Good, Very Good, Excellent',
                                          'F':'Female',
                                          '1: 0-50':'18-49',
                                          '2: 50-70':'50-69',
                                          '3: 70+':'70+',
                                          '0.0':'0',
                                          '1.0':'1',
                                          '2.0':'2',
                                          
                                          '1. Before May 2021: Alpha':'1. Before May 2021: Alpha',
                                          '2. May-Dec 2021: Delta':'2. May-Dec 2021: Delta',
                                          '3. After Dec 2021: Omicron':'3. After Dec 2021: Omicron',
                                          
                                          1: 'Yes',
                                          0: 'No',
                                          
                                          'Rural':'Rural',
                                          'Urban':'Urban',
                                          
                                          'MZ': 'Monozygotic'
                                          #  '':'',
                                          #  '':'',
                                          }


#%% Set exposure/test variables to include in model
# -----------------------------------------------------------------------------

# All variables for table
var_test_list_categorical_thriva2 = ['controls_only', 'imd_quintile', 'edu_bin_combined', 'PrePandemicEmploymentStatus_Earliest', 'RUC_grouped', 'ethnicity_ons_cat_combined_whitenonwhite', # Demographics
                              # Vaccination related
                              'Vaccine_2_name_grouped',
                              # Infection related
                              'Result_Thriva2_N', 'NaturalInfection_WideCDC_Interpretation_MaxToDate', 'HadCovid_Ever_SelfReport_original', 'HadCovid_Ever_SelfReport_grouped', 'HadCovid_Ever_SelfReport_binary', 'SymptomDuration_Max_grouped',
                              # General health factors
                              'ShieldingFlag', 'PrePandemicHealth_Earliest_original', 'PrePandemicHealth_Earliest_binary', 'BMI_cat5', 'BMI_cat2', 'FrailtyIndexOngoingCat_original', 'MedicationFlag_ED_Immunosuppressant',
                              # Comorbidities
                              'SubDomain_AnxietyOrDepression', 'SubDomain_DiabetesAny', 'SubDomain_HeartDiseaseOrFailure', 'Domain_CancerAny', 'SubDomain_Hypertension', 'SubDomain_AnxietyStressDisorder', 'SubDomain_Depression', 'SubDomain_MultimorbidityCount_Selected_Grouped',
                              # Mental health assessments
                              'PHQ4_Anxiety_grouped_cat2', 'PHQ4_Depression_grouped_cat2',
                              'PHQ4_AnxietyandDepression_grouped_cat3','PHQ4_AnxietyandDepression_grouped_cat2', 'HADS_Anxiety_grouped_cat2', 'HADS_Anxiety_grouped_cat3', 'HADS_Depression_grouped_cat2', 'HADS_Depression_grouped_cat3',
                              # Comorbidities extended
                              'Domain_CardiacDisease', 'Domain_CardiacRiskFactors', 'Domain_CancerAny', 'Domain_NeurologicalDisease', 'Domain_SubjectiveMemoryImpairment', 'Domain_Arthritis',
                              'SubDomain_HighCholesterol', 'Domain_ChronicLungDisease', 'SubDomain_AtrialFibrillation', 'SubDomain_Asthma', 'SubDomain_Stroke', 'SubDomain_Alzheimers', 'Domain_Osteoporosis', 'SubDomain_RheumatoidArthritis', 'SubDomain_Epilepsy',
                              ]

var_test_list_continuous_thriva2 = ['imd_decile_reverse', 'imd_quintile_reverse', # Demographics
                            # General health factors
                            'PrePandemicHealth_Earliest_ordinal', 'BMI', 'FrailtyIndexOngoingCat_Ordinal',
                            # Comorbidities
                            'SubDomain_MultimorbidityCount_Selected_Ungrouped', 'SubDomain_MultimorbidityCount_Ungrouped',
                            # Mental health assessments
                            'PHQ4_Anxiety_Value_Max', 'PHQ4_Depression_Value_Max', 'PHQ4_AnxietyandDepression_Value_Max', 'HADS_Anxiety_Value_Max', 'HADS_Depression_Value_Max']

# Extras for control set with age and sex only
var_test_list_continuous_thriva2_agesex_vacc2 = var_test_list_continuous_thriva2 + ['WeeksSinceVacc_2']
var_test_list_continuous_thriva2_agesex_vacc3 = var_test_list_continuous_thriva2 + ['WeeksSinceVacc_3']

var_test_list_categorical_thriva2_agesex_vacc2 = var_test_list_categorical_thriva2
var_test_list_categorical_thriva2_agesex_vacc3 = var_test_list_categorical_thriva2 + ['Vaccine_3_name_grouped']


var_test_list_categorical_thriva1 = ['controls_only', 'imd_decile', 'imd_quintile', 'edu_bin_combined', 'PrePandemicEmploymentStatus_Earliest', 'RUC_grouped', 'ethnicity_ons_cat_combined_whitenonwhite', # Demographics
                              
                              # Infection related
                              'Result_Thriva_N', 'NaturalInfection_WideCDC_Interpretation_MaxToDate', 'HadCovid_Ever_SelfReport_MaxToCoPE4_original', 'HadCovid_Ever_SelfReport_MaxToCoPE4_grouped', 'HadCovid_Ever_SelfReport_MaxToCoPE4_binary', 'SymptomDuration_MaxToCoPE4_grouped',
                              # General health factors
                              'ShieldingFlag', 'PrePandemicHealth_Earliest_original', 'PrePandemicHealth_Earliest_binary', 'BMI_cat5', 'BMI_cat2', 'FrailtyIndexOngoingCat_original', 'MedicationFlag_ED_Immunosuppressant',
                              # Comorbidities
                              'SubDomain_AnxietyOrDepression', 'SubDomain_DiabetesAny', 'SubDomain_HeartDiseaseOrFailure', 'Domain_CancerAny', 'SubDomain_Hypertension', 'SubDomain_AnxietyStressDisorder', 'SubDomain_Depression', 'SubDomain_MultimorbidityCount_Selected_Grouped',
                              # Mental health assessments
                              'PHQ4_Anxiety_grouped_MaxToCoPE4_cat2', 'PHQ4_Depression_grouped_MaxToCoPE4_cat2',
                              'PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat3','PHQ4_AnxietyandDepression_grouped_MaxToCoPE4_cat2', 'HADS_Anxiety_grouped_MaxToCoPE4_cat2', 'HADS_Anxiety_grouped_MaxToCoPE4_cat3', 'HADS_Depression_grouped_MaxToCoPE4_cat2', 'HADS_Depression_grouped_MaxToCoPE4_cat3',
                              # Comorbidities extended
                              'Domain_CardiacDisease', 'Domain_CardiacRiskFactors', 'Domain_CancerAny', 'Domain_NeurologicalDisease', 'Domain_SubjectiveMemoryImpairment', 'Domain_Arthritis',
                              'SubDomain_HighCholesterol', 'Domain_ChronicLungDisease', 'SubDomain_AtrialFibrillation', 'SubDomain_Asthma', 'SubDomain_Stroke', 'SubDomain_Alzheimers', 'Domain_Osteoporosis', 'SubDomain_RheumatoidArthritis', 'SubDomain_Epilepsy',
                              ]

var_test_list_continuous_thriva1 = ['imd_decile_reverse', 'imd_quintile_reverse', # Demographics
                            # General health factors
                            'PrePandemicHealth_Earliest_ordinal', 'BMI', 'FrailtyIndexOngoingCat_Ordinal',
                            # Comorbidities
                            'SubDomain_MultimorbidityCount_Selected_Ungrouped', 'SubDomain_MultimorbidityCount_Ungrouped',
                            # Mental health assessments
                            'PHQ4_Anxiety_Value_MaxToCoPE4', 'PHQ4_Depression_Value_MaxToCoPE4', 'PHQ4_AnxietyandDepression_Value_MaxToCoPE4', 'HADS_Anxiety_Value_MaxToCoPE4', 'HADS_Depression_Value_MaxToCoPE4']

# Extras for control set with age and sex only
var_test_list_continuous_thriva1_agesex_vacc1 = var_test_list_continuous_thriva1 + ['WeeksSinceVacc_1']
var_test_list_continuous_thriva1_agesex_vacc2 = var_test_list_continuous_thriva1 + ['WeeksSinceVacc_2']

var_test_list_categorical_thriva1_agesex_vacc1 = var_test_list_categorical_thriva1 + ['Vaccine_1_name_grouped']
var_test_list_categorical_thriva1_agesex_vacc2 = var_test_list_categorical_thriva1 + ['Vaccine_2_name_grouped']


# Create combined list of test variables
var_test_list_thriva1 = [var_test_list_categorical_thriva1, var_test_list_continuous_thriva1]
var_test_list_thriva2 = [var_test_list_categorical_thriva2, var_test_list_continuous_thriva2]

var_test_list_thriva1_agesex_vacc1 = [var_test_list_categorical_thriva1_agesex_vacc1, var_test_list_continuous_thriva1_agesex_vacc1]
var_test_list_thriva1_agesex_vacc2 = [var_test_list_categorical_thriva1_agesex_vacc2, var_test_list_continuous_thriva1_agesex_vacc2]

var_test_list_thriva2_agesex_vacc2 = [var_test_list_categorical_thriva2_agesex_vacc2, var_test_list_continuous_thriva2_agesex_vacc2]
var_test_list_thriva2_agesex_vacc3 = [var_test_list_categorical_thriva2_agesex_vacc3, var_test_list_continuous_thriva2_agesex_vacc3]

# Variable list when testing age as categorical rather than continuous
var_test_list_age_only = [['age_3_bands', 'age_custom'], []]


#%% Generate variable lists for post-vaccination infection models

# -----------------------------------------------------------------------------
# Models which look at association with antibody levels values only
# For 1 vacc at time of testing
var_test_list_categorical_thriva1_postvaccinfection_antibodylevels = ['ValueGrouped_50s', 'ValueGrouped_100s','Value_decile_thriva1vacc1', 'Value_quintile_thriva1vacc1']
var_test_list_continuous_thriva1_postvaccinfection_antibodylevels = ['Value', 'Value_DividedBy100']

var_test_list_thriva1_postvaccinfection_antibodylevels = [var_test_list_categorical_thriva1_postvaccinfection_antibodylevels, var_test_list_continuous_thriva1_postvaccinfection_antibodylevels]

# For 2 vacc at time of testing
var_test_list_categorical_thriva1_postvaccinfection_antibodylevels_2vacc = ['ValueGrouped_100s','ValueGrouped_Binary']
var_test_list_thriva1_postvaccinfection_antibodylevels_2vacc = [var_test_list_categorical_thriva1_postvaccinfection_antibodylevels_2vacc, var_test_list_continuous_thriva1_postvaccinfection_antibodylevels]

# -----------------------------------------------------------------------------
# Models which look at association with all other factors, same as for associations with low antibody levels
# Full list, same as for associations with low antibody levels
var_test_list_categorical_thriva1_postvaccinfection_all = var_test_list_categorical_thriva1 + ['Vaccine_1_name_grouped', 'age_3_bands', 'age_custom', 'sex']
var_test_list_continuous_thriva1_postvaccinfection_all = var_test_list_continuous_thriva1 + ['age']

var_test_list_thriva1_postvaccinfection_all = [var_test_list_categorical_thriva1_postvaccinfection_all, var_test_list_continuous_thriva1_postvaccinfection_all]

# Full list, same as for associations with low antibody levels, but without age as this is used as control
var_test_list_categorical_thriva1_postvaccinfection_noage = var_test_list_categorical_thriva1 + ['Vaccine_1_name_grouped', 'sex']
var_test_list_thriva1_postvaccinfection_all_noage = [var_test_list_categorical_thriva1_postvaccinfection_noage, var_test_list_continuous_thriva1]

# Full list, same as for associations with low antibody levels, but without age and serology based natural infection status as this is used as control
var_test_list_categorical_thriva1_postvaccinfection_noage_noserology = var_test_list_categorical_thriva1_postvaccinfection_noage.copy()
var_test_list_categorical_thriva1_postvaccinfection_noage_noserology.remove('NaturalInfection_WideCDC_Interpretation_MaxToDate')

var_test_list_thriva1_postvaccinfection_all_noage_noserology = [var_test_list_categorical_thriva1_postvaccinfection_noage_noserology, var_test_list_continuous_thriva1]


# Full list, same as for associations with low antibody levels, but without age, sex or serology based natural infection status as these are used as control
var_test_list_categorical_thriva1_postvaccinfection_noage_noserology_nosex = var_test_list_categorical_thriva1_postvaccinfection_noage_noserology.copy()
var_test_list_categorical_thriva1_postvaccinfection_noage_noserology.remove('sex')

var_test_list_thriva1_postvaccinfection_all_noage_noserology_nosex = [var_test_list_categorical_thriva1_postvaccinfection_noage_noserology_nosex, var_test_list_continuous_thriva1]



#%% Analysis A - logistic regression for outcome of Anti-S value < 250 U/mL, 2+ weeks after 2nd vaccination
# -----------------------------------------------------------------------------
# Add individual test variables one at a time to set of control variables

if do_analysis_logreg == 1:
    

    # lists of control variables to use for each set of data and outcome. nested lists for categorical and continuous variables separately
    
    
    # -------------------------------------------------------------------------
    ### Control variable sets for low antibody level models
    
    # age, sex only - as a least adjusted to check 
    control_var_list_agesex = [['sex'], ['age'], 'Adjusted for: Age, Sex']
    
    # sex, weeks since vaccination, vaccine received - use to test age as categorical
    control_var_list_1vacc_0 = [['sex', 'Vaccine_1_name_grouped'], ['WeeksSinceVacc_1'], 'Adjusted for: Sex, Vaccine received, Weeks since vaccination']
    control_var_list_2vacc_0 = [['sex', 'Vaccine_2_name_grouped'], ['WeeksSinceVacc_2'], 'Adjusted for: Sex, Vaccine received, Weeks since vaccination']
    control_var_list_3vacc_0 = [['sex', 'Vaccine_3_name_grouped'], ['WeeksSinceVacc_3'], 'Adjusted for: Sex, Vaccine received, Weeks since vaccination']
    
    # sex, weeks since vaccination, vaccine received, natural infection status - use to test age as categorical
    control_var_list_1vacc_0b = [['sex', 'Vaccine_1_name_grouped', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['WeeksSinceVacc_1'], 'Adjusted for: Sex, Vaccine received, Weeks since vaccination, Natural infection status (serology-based)']
    control_var_list_2vacc_0b = [['sex', 'Vaccine_2_name_grouped', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['WeeksSinceVacc_2'], 'Adjusted for: Sex, Vaccine received, Weeks since vaccination, Natural infection status (serology-based)']
    control_var_list_3vacc_0b = [['sex', 'Vaccine_3_name_grouped', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['WeeksSinceVacc_3'], 'Adjusted for: Sex, Vaccine received, Weeks since vaccination, Natural infection status (serology-based)']
    
    # Age, sex, weeks since vaccination, vaccine received
    control_var_list_1vacc_1 = [['sex', 'Vaccine_1_name_grouped'], ['age', 'WeeksSinceVacc_1'], 'Adjusted for: Age, Sex, Vaccine received, Weeks since vaccination']
    control_var_list_2vacc_1 = [['sex', 'Vaccine_2_name_grouped'], ['age', 'WeeksSinceVacc_2'], 'Adjusted for: Age, Sex, Vaccine received, Weeks since vaccination']
    control_var_list_3vacc_1 = [['sex', 'Vaccine_3_name_grouped'], ['age', 'WeeksSinceVacc_3'], 'Adjusted for: Age, Sex, Vaccine received, Weeks since vaccination']
    
    # Age, sex, weeks since vaccination, vaccine received, natural infection status
    control_var_list_1vacc_2 = [['sex', 'Vaccine_1_name_grouped', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['age', 'WeeksSinceVacc_1'], 'Adjusted for: Age, Sex, Vaccine received, Weeks since vaccination, Natural infection status (serology-based)']
    control_var_list_2vacc_2 = [['sex', 'Vaccine_2_name_grouped', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['age', 'WeeksSinceVacc_2'], 'Adjusted for: Age, Sex, Vaccine received, Weeks since vaccination, Natural infection status (serology-based)']
    control_var_list_3vacc_2 = [['sex', 'Vaccine_3_name_grouped', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['age', 'WeeksSinceVacc_3'], 'Adjusted for: Age, Sex, Vaccine received, Weeks since vaccination, Natural infection status (serology-based)']
    
    
    # Age, sex, weeks since vaccination
    control_var_list_agesexweekssincevacc_1vacc = [['sex'], ['age', 'WeeksSinceVacc_1'], 'Adjusted for: Age, Sex, Weeks since vaccination']
    control_var_list_agesexweekssincevacc_2vacc = [['sex'], ['age', 'WeeksSinceVacc_2'], 'Adjusted for: Age, Sex, Weeks since vaccination']
    
    # -------------------------------------------------------------------------
    ### Control variable sets for post-vaccination infection models
    # No controls
    control_var_list_nocontrol = [[],[], 'No control variables']
    
    # age only - as a least adjusted to check 
    control_var_list_age = [[], ['age'], 'Adjusted for: Age']
    
    # age and sex
    control_var_list_age_sex = [['sex'], ['age'], 'Adjusted for: Age, Sex']
    
    # age and serology based natural infection status 
    control_var_list_age_serology = [['NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['age'], 'Adjusted for: Age, Natural infection status (serology-based)']
    
    # age, sex and serology based natural infection status 
    control_var_list_age_serology_sex = [['sex', 'NaturalInfection_WideCDC_Interpretation_MaxToDate'], ['age'], 'Adjusted for: Age, Sex, Natural infection status (serology-based)']
    
    # age, sex and weeks since vaccination
    # control_var_list_agesexweekssincevacc_1vacc
    
    # age, sex and weeks since vaccination, employment status 
    control_var_list_agesexweekssincevaccemployment_1vacc = [['sex', 'PrePandemicEmploymentStatus_Earliest'], ['age', 'WeeksSinceVacc_1'], 'Adjusted for: Age, Sex, Employment status, Weeks since vaccination']
    
    # Weeks since vacc at thriva 1 only
    control_var_list_weekssincevacconly_1vacc = [[], ['WeeksSinceVacc_1'], 'Adjusted for: Weeks since vaccination']
    control_var_list_weekssincevacconly_2vacc = [[], ['WeeksSinceVacc_2'], 'Adjusted for: Weeks since vaccination']
        
    # Age, Weeks since vacc at thriva 1 only
    control_var_list_age_weekssincevacconly_1vacc = [[], ['age','WeeksSinceVacc_1'], 'Adjusted for: Age, Weeks since vaccination']
    control_var_list_age_weekssincevacconly_2vacc = [[], ['age','WeeksSinceVacc_2'], 'Adjusted for: Age, Weeks since vaccination']
        
    
    # Antibody level value only
    control_var_list_value = [['ValueGrouped_100s'], [], 'Adjusted for: antibody level']
    
    # Weeks since vacc & value
    control_var_list_weekssincevacc_value_1vacc = [['ValueGrouped_100s'], ['WeeksSinceVacc_1'], 'Adjusted for: Weeks since vaccination, antibody level']
    control_var_list_weekssincevacc_value_2vacc = [['ValueGrouped_100s'], ['WeeksSinceVacc_2'], 'Adjusted for: Weeks since vaccination, antibody level']
    
    
    # List of dataset + outcomes to test with logistic regression
    # Testing age as categorical variables, controlling for sex, weeks since vaccination, vaccine received
    dataset_plus_outcome_list_0 = [[data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_5pct', 1, var_test_list_age_only, '5%', control_var_list_1vacc_0, 'Thriva'], #0
                                 [data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_10pct', 1, var_test_list_age_only, '10%', control_var_list_1vacc_0, 'Thriva'], #1
                                 [data_thriva1_2vacc, 'OutcomeFlag_thriva1_2vacc_5pct', 2, var_test_list_age_only, '5%', control_var_list_2vacc_0, 'Thriva'], #2
                                 [data_thriva1_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva1', 2, var_test_list_age_only, '< 250 (8%)', control_var_list_2vacc_0, 'Thriva'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_5pct', 2, var_test_list_age_only, '5%', control_var_list_2vacc_0, 'Thriva #2'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_10pct', 2, var_test_list_age_only, '10%', control_var_list_2vacc_0, 'Thriva #2'], #4
                                 [data_thriva2_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva2', 2, var_test_list_age_only, '< 250 (20%)', control_var_list_2vacc_0, 'Thriva #2'], #5
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_5pct', 3, var_test_list_age_only, '5%', control_var_list_3vacc_0, 'Thriva #2'],#6
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_10pct', 3, var_test_list_age_only, '10%', control_var_list_3vacc_0, 'Thriva #2']#7
                                 ]
    # Testing age as categorical variables, controlling for sex, weeks since vaccination, vaccine received
    dataset_plus_outcome_list_0b = [[data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_5pct', 1, var_test_list_age_only, '5%', control_var_list_1vacc_0b, 'Thriva'], #0
                                 [data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_10pct', 1, var_test_list_age_only, '10%', control_var_list_1vacc_0b, 'Thriva'], #1
                                 [data_thriva1_2vacc, 'OutcomeFlag_thriva1_2vacc_5pct', 2, var_test_list_age_only, '5%', control_var_list_2vacc_0b, 'Thriva'], #2
                                 [data_thriva1_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva1', 2, var_test_list_age_only, '< 250 (8%)', control_var_list_2vacc_0b, 'Thriva'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_5pct', 2, var_test_list_age_only, '5%', control_var_list_2vacc_0b, 'Thriva #2'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_10pct', 2, var_test_list_age_only, '10%', control_var_list_2vacc_0b, 'Thriva #2'], #4
                                 [data_thriva2_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva2', 2, var_test_list_age_only, '< 250 (20%)', control_var_list_2vacc_0b, 'Thriva #2'], #5
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_5pct', 3, var_test_list_age_only, '5%', control_var_list_3vacc_0b, 'Thriva #2'],#6
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_10pct', 3, var_test_list_age_only, '10%', control_var_list_3vacc_0b, 'Thriva #2']#7
                                 ]
    
    # Testing main list of variables, controlling for age, sex, weeks since vaccination, vaccine received
    dataset_plus_outcome_list_1 = [[data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_5pct', 1, var_test_list_thriva1, '5%', control_var_list_1vacc_1, 'Thriva'], #0
                                 [data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_10pct', 1, var_test_list_thriva1, '10%', control_var_list_1vacc_1, 'Thriva'], #1
                                 [data_thriva1_2vacc, 'OutcomeFlag_thriva1_2vacc_5pct', 2, var_test_list_thriva1, '5%', control_var_list_2vacc_1, 'Thriva'], #2
                                 [data_thriva1_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva1', 2, var_test_list_thriva1, '< 250 (8%)', control_var_list_2vacc_1, 'Thriva'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_5pct', 2, var_test_list_thriva2, '5%', control_var_list_2vacc_1, 'Thriva #2'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_10pct', 2, var_test_list_thriva2, '10%', control_var_list_2vacc_1, 'Thriva #2'], #4
                                 [data_thriva2_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva2', 2, var_test_list_thriva2, '< 250 (20%)', control_var_list_2vacc_1, 'Thriva #2'], #5
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_5pct', 3, var_test_list_thriva2, '5%', control_var_list_3vacc_1, 'Thriva #2'],#6
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_10pct', 3, var_test_list_thriva2, '10%', control_var_list_3vacc_1, 'Thriva #2']#7
                                 ]
    
    # Testing main list of variables, controlling for age, sex, weeks since vaccination, vaccine received, natural infection status
    dataset_plus_outcome_list_2 = [[data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_5pct', 1, var_test_list_thriva1, '5%', control_var_list_1vacc_2, 'Thriva'], #0
                                 [data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_10pct', 1, var_test_list_thriva1, '10%', control_var_list_1vacc_2, 'Thriva'], #1
                                 [data_thriva1_2vacc, 'OutcomeFlag_thriva1_2vacc_5pct', 2, var_test_list_thriva1, '5%', control_var_list_2vacc_2, 'Thriva'], #2
                                 [data_thriva1_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva1', 2, var_test_list_thriva1, '< 250 (8%)', control_var_list_2vacc_2, 'Thriva'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_5pct', 2, var_test_list_thriva2, '5%', control_var_list_2vacc_2, 'Thriva #2'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_10pct', 2, var_test_list_thriva2, '10%', control_var_list_2vacc_2, 'Thriva #2'], #4
                                 [data_thriva2_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva2', 2, var_test_list_thriva2, '< 250 (20%)', control_var_list_2vacc_2, 'Thriva #2'], #5
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_5pct', 3, var_test_list_thriva2, '5%', control_var_list_3vacc_2, 'Thriva #2'],#6
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_10pct', 3, var_test_list_thriva2, '10%', control_var_list_3vacc_2, 'Thriva #2']#7
                                 ]
    
    # Testing main list of variables, controlling for age and sex only
    dataset_plus_outcome_list_3 = [[data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_5pct', 1, var_test_list_thriva1_agesex_vacc1, '5%', control_var_list_agesex, 'Thriva'], #0
                                 [data_thriva1_1vacc, 'OutcomeFlag_thriva1_1vacc_10pct', 1, var_test_list_thriva1_agesex_vacc1, '10%', control_var_list_agesex, 'Thriva'], #1
                                 [data_thriva1_2vacc, 'OutcomeFlag_thriva1_2vacc_5pct', 2, var_test_list_thriva1_agesex_vacc2, '5%', control_var_list_agesex, 'Thriva'], #2
                                 [data_thriva1_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva1', 2, var_test_list_thriva1_agesex_vacc2, '< 250 (8%)', control_var_list_agesex, 'Thriva'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_5pct', 2, var_test_list_thriva2_agesex_vacc2, '5%', control_var_list_agesex, 'Thriva #2'], #3
                                 [data_thriva2_2vacc, 'OutcomeFlag_thriva2_2vacc_10pct', 2, var_test_list_thriva2_agesex_vacc2, '10%', control_var_list_agesex, 'Thriva #2'], #4
                                 [data_thriva2_2vacc, 'OutcomeFlag_Low_2Vacc_Thriva2', 2, var_test_list_thriva2_agesex_vacc2, '< 250 (20%)', control_var_list_agesex, 'Thriva #2'], #5
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_5pct', 3, var_test_list_thriva2_agesex_vacc3, '5%', control_var_list_agesex, 'Thriva #2'],#6
                                 [data_thriva2_3vacc, 'OutcomeFlag_thriva2_3vacc_10pct', 3, var_test_list_thriva2_agesex_vacc3, '10%', control_var_list_agesex, 'Thriva #2']#7
                                 ]
    
    
    # -------------------------------------------------------------------------
    ##### Post-vaccination infection models
    # Testing association with antibody levels at Thriva 1 only
    # Individuals once vacc at thriva 1
    datasetplusoutcomelist_PVI_afterthriva1_antibodylevels = [[data_thriva1_1vacc, # 1 vacc at Thriva 1, infection after any vacc
                                                               'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1', control_var_list_nocontrol, 'Thriva'], # 1 vacc at thriva 1, infection at any time, no controls
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (any vacc status)', control_var_list_age, 'Thriva'], # 1 vacc at thriva 1, infection at any time, controls: age
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (any vacc status)', control_var_list_weekssincevacconly_1vacc, 'Thriva'], # 1 vacc at thriva 1, infection at any time, controls: weeks since vacc 1 when tested
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (any vacc status)', control_var_list_age_weekssincevacconly_1vacc, 'Thriva'], # 1 vacc at thriva 1, infection at any time, controls: age, weeks since vacc 1 when tested
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (any vacc status)', control_var_list_agesexweekssincevacc_1vacc, 'Thriva'], # 1 vacc at thriva 1, infection at any time, controls: age, sex, weeks since vacc 1 when tested
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (any vacc status)', control_var_list_agesexweekssincevaccemployment_1vacc, 'Thriva'], # 1 vacc at thriva 1, infection at any time, controls: age, sex, employment status, weeks since vacc 1 when tested
                                                             
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (any vacc status)', control_var_list_1vacc_2, 'Thriva'], # 1 vacc at thriva 1, infection at any time, controls: age, sex, weeks since vacc 1 when tested, natural infection status
                                                              
                                                              # 1 vacc at Thriva 1, infection at vacc 2 only
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (while twice vaccinated)', control_var_list_nocontrol, 'Thriva'], # 1 vacc at thriva 1, infection while twice vacc, no controls
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (while twice vaccinated)', control_var_list_age, 'Thriva'], # 1 vacc at thriva 1, infection while twice vacc, controls: age
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (while twice vaccinated)', control_var_list_weekssincevacconly_1vacc, 'Thriva'], # 1 vacc at thriva 1, infection while twice vacc, controls: weeks since vacc 1 when tested
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (while twice vaccinated)', control_var_list_age_weekssincevacconly_1vacc, 'Thriva'], # 1 vacc at thriva 1, infection while twice vacc, controls: age, weeks since vacc 1 when tested
                                                              [data_thriva1_1vacc, 'OutcomeFlag_PostVaccInfectionAnyTime_2Vaccine', 1, var_test_list_thriva1_postvaccinfection_antibodylevels, 'Post-Vaccination Infection after Thriva 1 (while twice vaccinated)', control_var_list_1vacc_2, 'Thriva'], # 1 vacc at thriva 1, infection while twice vacc, controls: age, sex, weeks since vacc 1 when tested, natural infection status
                                                              
                                                              # 2 vacc at Thriva 1, infection at vacc 2 or 3 only
                                                              [data_thriva1_2vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 2, var_test_list_thriva1_postvaccinfection_antibodylevels_2vacc, 'Post-Vaccination Infection after Thriva 1 (while twice or 3 times vaccinated)', control_var_list_nocontrol, 'Thriva'], # 2 vacc at thriva 1, infection while 2/3 vacc, no controls
                                                              [data_thriva1_2vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 2, var_test_list_thriva1_postvaccinfection_antibodylevels_2vacc, 'Post-Vaccination Infection after Thriva 1 (while twice or 3 times vaccinated)', control_var_list_age, 'Thriva'], # 2 vacc at thriva 1, infection while 2/3 vacc, controls: age
                                                              [data_thriva1_2vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 2, var_test_list_thriva1_postvaccinfection_antibodylevels_2vacc, 'Post-Vaccination Infection after Thriva 1 (while twice or 3 times vaccinated)', control_var_list_age_weekssincevacconly_2vacc, 'Thriva'], # 2 vacc at thriva 1, infection while 2/3 vacc, controls: age, weeks since vacc 2 when tested
                                                              [data_thriva1_2vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 2, var_test_list_thriva1_postvaccinfection_antibodylevels_2vacc, 'Post-Vaccination Infection after Thriva 1 (while twice or 3 times vaccinated)', control_var_list_agesexweekssincevacc_2vacc, 'Thriva'], # 2 vacc at thriva 1, infection at any time, controls: age, sex, weeks since vacc 2 when tested
                                                              [data_thriva1_2vacc, 'OutcomeFlag_PostVaccInfectionAfterThriva1', 2, var_test_list_thriva1_postvaccinfection_antibodylevels_2vacc, 'Post-Vaccination Infection after Thriva 1 (while twice or 3 times vaccinated)', control_var_list_2vacc_2, 'Thriva'], # 2 vacc at thriva 1, infection while 2/3 vacc, controls: age, sex, weeks since vacc 1 when tested, natural infection status
                                                              ]
    
    # Testing association with factors other than antibody level
    # Individuals from Thriva 1 who are at least once vaccinated at any point in time
    datasetplusoutcomelist_PVI_anytime = [[data_thriva1_all_vaccinated, # Thriva 1, vaccinated at least once then or later, infection after any vacc, before or after thriva 1
                                           'OutcomeFlag_PostVaccInfectionAnyTime', 123, var_test_list_thriva1_postvaccinfection_all, 'Post-Vaccination Infection any time (any vacc status at Thriva 1)', control_var_list_nocontrol, 'Thriva'], # Any vacc status at thriva 1, infection at any time, no controls
                                          
                                          [data_thriva1_all_vaccinated, 'OutcomeFlag_PostVaccInfectionAnyTime', 123, var_test_list_thriva1_postvaccinfection_all_noage, 'Post-Vaccination Infection any time (any vacc status at Thriva 1)', control_var_list_age, 'Thriva'], # Any vacc status at thriva 1, infection at any time, controls: age
                                          
                                          [data_thriva1_all_vaccinated, 'OutcomeFlag_PostVaccInfectionAnyTime', 123, var_test_list_thriva1_postvaccinfection_all_noage_noserology, 'Post-Vaccination Infection any time (any vacc status at Thriva 1)', control_var_list_age_sex, 'Thriva'], # Any vacc status at thriva 1, infection at any time, controls: age, sex
                                          [data_thriva1_all_vaccinated, 'OutcomeFlag_PostVaccInfectionAnyTime', 123, var_test_list_thriva1_postvaccinfection_all_noage_noserology_nosex, 'Post-Vaccination Infection any time (any vacc status at Thriva 1)', control_var_list_age_serology_sex, 'Thriva'], # Any vacc status at thriva 1, infection at any time, controls: age, serology status, sex
                                          ]
    
    # -------------------------------------------------------------------------
    # Full list of datasets and parameters to do in modelling
    dataset_plus_outcome_list_lowantibodylevels = dataset_plus_outcome_list_1 + dataset_plus_outcome_list_2 +  dataset_plus_outcome_list_3 + dataset_plus_outcome_list_0 + dataset_plus_outcome_list_0b 

    if do_postvaccinf != 1:
        dataset_plus_outcome_list = dataset_plus_outcome_list_lowantibodylevels
    if do_postvaccinf == 1:       
        dataset_plus_outcome_list = datasetplusoutcomelist_PVI_afterthriva1_antibodylevels + datasetplusoutcomelist_PVI_anytime
        
    # -------------------------------------------------------------------------
    ### Run models
    regression_results_all_filtered_list = []
    # Loop through every combination of dataset, outcome variable, control variables to use in model, and run model
    for n in range(0,len(dataset_plus_outcome_list),1): #  range(0,1,1): #
        print(n)
        #n = 2
        
        data_full = dataset_plus_outcome_list[n][0]
        outcome_var = dataset_plus_outcome_list[n][1] 
        vacc = dataset_plus_outcome_list[n][2] 
        outcome_string = dataset_plus_outcome_list[n][4]
        
        var_test_list_categorical = dataset_plus_outcome_list[n][3][0] 
        var_test_list_continuous = dataset_plus_outcome_list[n][3][1] 
        var_test_list = var_test_list_categorical + var_test_list_continuous
        
        var_control_dummy = dataset_plus_outcome_list[n][5][0] 
        var_control_continuous = dataset_plus_outcome_list[n][5][1] 
        model_name = dataset_plus_outcome_list[n][5][2]
        study_name = dataset_plus_outcome_list[n][6]
        
        # Create empty lists to fill with results of models
        regression_results_summarytable_list = []
        regression_results_summarytable_filtered_list = []
        regression_metrics_list = [] 
        
        # Cycle through each variable in list of test variables
        for var in var_test_list: 
            # For each test variable, do model for each control variable variant and 1 & 2 vaccinations
            var_original = var # take copy of unmodified test variable, so that vaccine number can be added correctly for relevant variables
        
            # Create copy that will be filtered to remove missing data
            data_slice = data_full.copy()
            
            # -------------------------------------------------------------
            #### Remove missing data from outcome variable
            data_slice = data_slice[(data_slice[outcome_var] != 'NoDataAvailable')
                                    # & (data_slice[col_nan] != 'Other')
                                    & (data_slice[outcome_var] != '0.0 Unknown - individual did not complete CoPE')
                                    & (data_slice[outcome_var] != '1. Assessment incomplete')
                                    & (data_slice[outcome_var] != '0.1 Unknown - Answer not provided in CoPE')
                                    & ~(data_slice[outcome_var].isnull())
                                    ]
            data_slice = data_slice.reset_index(drop = True)
            
            # -------------------------------------------------------------
            #### Setup control variables
                   
            # Generate lists of control variables to include in model and NaN variables to filter out NaN rows
            # categorical variables - pull out dummy variables
            input_var_dummy_control = gen_dummy_var_list(var_control_dummy, dummy_var_list_to_test)
            input_var_dummy_control_nan = gen_dummy_var_list(var_control_dummy, dummy_NaN_list)
            
            # Filter out NaN rows present in control variables
            # data_slice = data_full.copy()
            
            # Loop through categorical dummy NaN variables
            for col_nan in input_var_dummy_control_nan:
                data_slice = data_slice[(data_slice[col_nan] != 1)]
                data_slice = data_slice.reset_index(drop = True)
            # Loop through continuous variables, removing rows which contain missing or excluded data 
            for col_nan in var_control_continuous:
                data_slice = data_slice[(data_slice[col_nan] != 'NoDataAvailable')
                                        # & (data_slice[col_nan] != 'Other')
                                        & (data_slice[col_nan] != '0.0 Unknown - individual did not complete CoPE')
                                        & (data_slice[col_nan] != '1. Assessment incomplete')
                                        & (data_slice[col_nan] != '0.1 Unknown - Answer not provided in CoPE')
                                        & ~(data_slice[col_nan].isnull())
                                        ]
                data_slice = data_slice.reset_index(drop = True)
                
            # automatically drop any dummy variables from list of control test dummys where sum of column in dataset is below a threshold value of 1 - i.e. where are no observations that are = 1 and so variable shouldn't be included in model
            for col in input_var_dummy_control: 
                if data_slice[col].sum() <= 1:
                    input_var_dummy_control.remove(col)
        
        
            if ((var == 'controls_only') | (var in var_control_dummy)): # to get controls on their own without any test variables
                data_slice_var = data_slice.copy()      
                # generate List of control and test variables to include in model
                input_var_control_test = input_var_dummy_control + var_control_continuous
            
            
            else:
                # -------------------------------------------------------------
                #### Setup test variables
                # Categorical test variables
                if var in var_test_list_categorical:
                    # Generate lists of test variables to include in model and NaN variables to filter out NaN rows
                    input_var_dummy_test = gen_dummy_var_list([var], dummy_var_list_to_test)
                    input_var_dummy_test_nan = gen_dummy_var_list([var], dummy_NaN_list)
                    # Filter out NaN rows present in test variable
                    data_slice_var = data_slice.copy()
                    for col_nan in input_var_dummy_test_nan:
                        if col_nan in dummy_var_list_to_test:
                            data_slice_var = data_slice_var[(data_slice_var[col_nan] != 1)]
                            data_slice_var = data_slice_var.reset_index(drop = True)    
                    # automatically drop any dummy variables from list of control test dummys where sum of column in dataset is below a threshold value of 1 - i.e. where are no observations that are = 1 and so variable shouldn't be included in model
                    for col in input_var_dummy_test: 
                        if data_slice_var[col].astype(float).sum() <= 1:
                            input_var_dummy_test.remove(col)
                 
                    test_var_list = input_var_dummy_test
                    
                # Continuous test variables
                if var_original in var_test_list_continuous: 
                    data_slice_var = data_slice.copy()
                    # Filter out missing or excluded data
                    data_slice_var = data_slice_var[(data_slice_var[var] != 'NoDataAvailable')
                                            # & (data_slice_var[var] != 'Other')
                                            & (data_slice_var[var] != '0.0 Unknown - individual did not complete CoPE')
                                            & (data_slice_var[var] != '1. Assessment incomplete')
                                            & (data_slice_var[var] != '0.1 Unknown - Answer not provided in CoPE')
                                            & (data_slice_var[var] != 'NaN')
                                            & ~(data_slice_var[var].isnull())
                                            ]
                    data_slice_var = data_slice_var.reset_index(drop = True)
                    # Cast as float to make sure data type is correct for continuous data
                    data_slice_var[var] = data_slice_var[var].astype(float)
                    test_var_list = [var]
                
                # generate List of control and test variables to include in model
                input_var_control_test = input_var_dummy_control + var_control_continuous + test_var_list
                
            # generate x dataset for selected control and test dummy variables only
            x_data = data_slice_var[input_var_control_test].reset_index(drop=True) # create input variable tables for models 
            # generate y datasets from selected number of vaccinations and outcome of interest
            y_data = data_slice_var[outcome_var].reset_index(drop=True) # set output variable
            
                
            # Do logistic regression (stats models) of control + test variables
            sm_summary, sm_fit_metrics_summary = sm_logreg_summary(x_data, y_data, CI_alpha = 0.05, plot_roc = 0, do_robust_se = 'HC3', cluster_df = data_slice_var, cluster_col = 'FamilyNumber')
                        
            # Add columns to summary table to help later filtering
            sm_summary['NumVacc'] = vacc # Add control variable variant and number of vaccinations
            sm_summary['ControlVars'] = model_name # Add control variable variant and number of vaccinations
            sm_summary['OutcomeThresh'] = outcome_string # Add outcome threshold
            sm_summary['TestVar'] = var # add test variable name
            sm_summary['Variable_tidy'] = sm_summary['Variable'].map(codebook['TidyVariableNames'])
            sm_summary['StudyName'] = study_name
            sm_summary['SampleSize'] = len(y_data)
            
            # Create column which combines OR, CI and p value in one cell
            round_dp = 2
            sm_summary['OR_tidy'] = sm_summary['Odds ratio'].round(round_dp).astype(str) + ' (' + sm_summary['OR C.I. (lower)'].round(round_dp).astype(str) + ', ' + sm_summary['OR C.I. (upper)'].round(round_dp).astype(str) + '),' 
            sm_summary.loc[(sm_summary['P-value'] >= 0.05)
                           ,'OR_tidy'] = sm_summary['OR_tidy'] + ' p = ' + sm_summary['P-value'].round(2).astype(str)
            sm_summary.loc[(sm_summary['P-value'] < 0.05) &
                           (sm_summary['P-value'] >= 0.01)
                           ,'OR_tidy'] = sm_summary['OR_tidy'] + ' p = ' + sm_summary['P-value'].round(2).astype(str) + '*'
            sm_summary.loc[(sm_summary['P-value'] < 0.01) &
                           (sm_summary['P-value'] >= 0.001)
                           ,'OR_tidy'] = sm_summary['OR_tidy'] + ' p = ' + sm_summary['P-value'].round(3).astype(str) + '**'
            sm_summary.loc[(sm_summary['P-value'] < 0.001) &
                           (sm_summary['P-value'] >= 0.0001)
                           ,'OR_tidy'] = sm_summary['OR_tidy'] + ' p = ' + sm_summary['P-value'].round(4).astype(str) + '***'
            sm_summary.loc[(sm_summary['P-value'] < 0.0001)
                           ,'OR_tidy'] = sm_summary['OR_tidy'] + ' p < 0.0001***'
            
            # filter for test variable result only
            if (var == 'controls_only'):
                sm_summary_filtered = sm_summary[(sm_summary['Variable'] != 'const')]
            else:
                sm_summary_filtered = sm_summary[(sm_summary['Variable'].str.contains(sm_summary['TestVar'].max()))]
        
            # Append result to master list
            regression_results_summarytable_list.append(sm_summary)
            regression_metrics_list.append(sm_fit_metrics_summary)
            regression_results_summarytable_filtered_list.append(sm_summary_filtered)
        
        
        # Append tables together
        regression_summary_master = regression_results_summarytable_list[0]
        for i in range(1,(len(regression_results_summarytable_list))):
            regression_summary_master = regression_summary_master.append(regression_results_summarytable_list[i])
        
        # table filtered for test variables only
        regression_summary_filtered_master = regression_results_summarytable_filtered_list[0]
        for i in range(1,(len(regression_results_summarytable_filtered_list))):
            regression_summary_filtered_master = regression_summary_filtered_master.append(regression_results_summarytable_filtered_list[i])
        
        regression_results_all_filtered_list.append(regression_summary_filtered_master)
            
    # Append results tables from various runs together
    regression_results_all_filtered = regression_results_all_filtered_list[0]
    for i in range(1,(len(regression_results_all_filtered_list))):
        regression_results_all_filtered = regression_results_all_filtered.append(regression_results_all_filtered_list[i])
    
    # -----------------------------------------------------------------------------
    # Export results
    if export_csv == 1:
        if do_postvaccinf != 1:
            regression_results_all_filtered.to_csv('TwinsCovid_LogisticRegression_ResultsAll.csv')
        elif do_postvaccinf == 1:
            regression_results_all_filtered.to_csv('TwinsCovid_LogisticRegression_ResultsAll_PostVaccInf.csv')



#%% Cross tabulations of SARS-CoV-2 infection prevalence against socio-demographic factors, TwinsUK Q4
# -----------------------------------------------------------------------------
# Summarise cross-tabs by grouping by 2 variables
# Groupby counting column to get counts and proportions    
# crosstab_table = df_slice.groupby(groupby_field)[count_colname].count().reset_index()
list_row_colname = ['age_10yr_bands', 'sex',  'ethnicity_ons_cat_combined_whitenonwhite', 'imd_quintile', 'RUC_grouped', 'StudyName']

list_col_colname = ['NaturalInfection_WideCDC_Interpretation_MaxToDate',
                   'Result_Thriva2_N',
                    'HadCovid_Ever_SelfReport_grouped',
                   ]

list_col_category = [['2. Evidence of natural infection'],
                     ['positive'],
                      ['2.1 SuspectedCovid','3. PositiveCovid'],
                     ]

df = data_thriva2_all

crosstab_table_filter_list= []
count_colname = 'StudyNumber' # Value

for row_colname in list_row_colname:
    for n in range(0,len(list_col_colname),1):
        col_colname = list_col_colname[n]
        col_category = list_col_category[n]

        df_slice = df.copy()
        
        # Delete value where row and column fields have missing value categories, so row is not counted
        df_slice.loc[df_slice[row_colname].isin(missing_vals), count_colname] = np.nan
        df_slice.loc[df_slice[col_colname].isin(missing_vals), count_colname] = np.nan
        
        df_slice.loc[df_slice[row_colname].isin(missing_vals), row_colname] = np.nan
        df_slice.loc[df_slice[col_colname].isin(missing_vals), col_colname] = np.nan
        
        df_slice = df_slice.rename(columns = {row_colname: 'row_cat',
                                              col_colname: 'col_cat',})
        
        groupby_field = ['row_cat', 'col_cat']
        
        crosstab_table = df_slice.groupby(groupby_field)[count_colname].count().unstack(fill_value=0).stack().reset_index()
        crosstab_table = crosstab_table.rename(columns = {0:count_colname})
        
        row_count = df_slice.groupby(groupby_field[0])[count_colname].count().reset_index()
        row_count = row_count.rename(columns = {count_colname: 'row_count'})
        col_count = df_slice.groupby(groupby_field[1])[count_colname].count().reset_index()
        col_count = col_count.rename(columns = {count_colname: 'col_count'})
        # merge row and column totals to main grouped tables
        crosstab_table = pd.merge(left = crosstab_table, right = row_count, how = 'left', on = groupby_field[0])
        crosstab_table = pd.merge(left = crosstab_table, right = col_count, how = 'left', on = groupby_field[1])
        # calculate proportions
        crosstab_table['row_prop'] = crosstab_table[count_colname] / crosstab_table['row_count']
        crosstab_table['col_prop'] = crosstab_table[count_colname] / crosstab_table['col_count']
        
        crosstab_table['row_prop_pct'] = (crosstab_table['row_prop']*100)
        
        # pivot to get crosstab tables to output
        crosstab_table_count = crosstab_table.pivot(index = groupby_field[0], columns = groupby_field[1], values = count_colname).fillna(0)
        
        # test association between row and column variable counts with chi-squared
        chi2, p, dof, ex = sp.stats.chi2_contingency(crosstab_table_count, correction=False)
        
        # Add test details to crosstab table
        crosstab_table['row_variable'] = row_colname
        crosstab_table['col_variable'] = col_colname
        crosstab_table['chi_square_pvalue'] = p
        # add significance level
        crosstab_table.loc[(crosstab_table['chi_square_pvalue'] < 0.05), 'significance_level'] = '*, p < 0.05'
        crosstab_table.loc[(crosstab_table['chi_square_pvalue'] < 0.01), 'significance_level'] = '**, p < 0.01'
        crosstab_table.loc[(crosstab_table['chi_square_pvalue'] < 0.001), 'significance_level'] = '***, p < 0.001'
        crosstab_table.loc[(crosstab_table['chi_square_pvalue'] >= 0.05), 'significance_level'] = 'p >= 0.05'
        
        # Add tidy row proportion with (%)
        crosstab_table['row_prop_tidy'] = crosstab_table[count_colname].astype(str) + '/' + crosstab_table['row_count'].astype(str) + ' (' + (100*crosstab_table['row_prop']).round(1).astype(str) + '%)'
        
        # Add tidy row proportion with (%) and chi-square p-value
        crosstab_table['row_prop_tidy_plusp'] = crosstab_table['row_prop_tidy'] + ' p = ' + crosstab_table['chi_square_pvalue'].astype(str)
        
        # Filter for category of interest only
        crosstab_table_filter = crosstab_table[(crosstab_table['col_cat'].isin(col_category))]
        
        # Plot as bar chart
        plt.figure()
        sns.barplot(data = crosstab_table_filter, x = 'row_cat', y = 'row_prop_pct', hue = 'col_cat')
        
        crosstab_table_filter_list.append(crosstab_table_filter)


# Append results tables from various runs together
crosstab_table_filter_combined = crosstab_table_filter_list[0]
for i in range(1,(len(crosstab_table_filter_list))):
    crosstab_table_filter_combined = crosstab_table_filter_combined.append(crosstab_table_filter_list[i])
    
crosstab_table_filter_combined = crosstab_table_filter_combined.reset_index()
        
crosstab_table_filter_combined['row_tidy'] = crosstab_table_filter_combined['row_variable'].map(codebook['table_characteristics_vars']) + ': ' + crosstab_table_filter_combined['row_cat']
crosstab_table_filter_combined['col_tidy'] = crosstab_table_filter_combined['col_variable'].map(codebook['table_characteristics_vars']) + ': ' + crosstab_table_filter_combined['col_cat']

crosstab_table_filter_combined_tidy = crosstab_table_filter_combined[['row_tidy','col_tidy','row_prop_tidy_plusp']]

crosstab_table_filter_combined_pivot = crosstab_table_filter_combined.pivot_table(index = ['row_tidy'], columns = 'col_tidy', values = 'row_prop')



#%% Calculate difference in antibody levels within MZ and DZ twins, as well as within all non-related pairs

# All individuals
values_all = np.array(data_thriva2_3vacc['Value'])

diff_matrix_all = abs(values_all[:,None]-values_all)
diff_matrix_all = pd.DataFrame(diff_matrix_all, index = data_thriva2_3vacc['StudyNumber'], columns = data_thriva2_3vacc['StudyNumber'])

# Flatten matrix
diff_matrix_all_flat = pd.melt(diff_matrix_all, ignore_index=False)
diff_matrix_all_flat = diff_matrix_all_flat.rename(columns = {'StudyNumber':'StudyNumber_b'})
diff_matrix_all_flat = diff_matrix_all_flat.reset_index()
diff_matrix_all_flat = diff_matrix_all_flat.rename(columns = {'StudyNumber':'StudyNumber_a'})

# Add family number of person a and b
diff_matrix_all_flat = pd.merge(diff_matrix_all_flat, data_thriva2_3vacc[['StudyNumber','FamilyNumber']], how = 'left', left_on = 'StudyNumber_a', right_on = 'StudyNumber')
diff_matrix_all_flat = diff_matrix_all_flat.rename(columns = {'FamilyNumber':'FamilyNumber_a'})

diff_matrix_all_flat = pd.merge(diff_matrix_all_flat, data_thriva2_3vacc[['StudyNumber','FamilyNumber']], how = 'left', left_on = 'StudyNumber_b', right_on = 'StudyNumber')
diff_matrix_all_flat = diff_matrix_all_flat.rename(columns = {'FamilyNumber':'FamilyNumber_b'})

# Add zygosity of person a and b
diff_matrix_all_flat = pd.merge(diff_matrix_all_flat, data_thriva2_3vacc[['StudyNumber','ACTUAL_ZYGOSITY']], how = 'left', left_on = 'StudyNumber_a', right_on = 'StudyNumber')
diff_matrix_all_flat = diff_matrix_all_flat.rename(columns = {'ACTUAL_ZYGOSITY':'ACTUAL_ZYGOSITY_a'})

diff_matrix_all_flat = pd.merge(diff_matrix_all_flat, data_thriva2_3vacc[['StudyNumber','ACTUAL_ZYGOSITY']], how = 'left', left_on = 'StudyNumber_b', right_on = 'StudyNumber')
diff_matrix_all_flat = diff_matrix_all_flat.rename(columns = {'ACTUAL_ZYGOSITY':'ACTUAL_ZYGOSITY_b'})

# remove rows where studynumber a == studynumber b
diff_matrix_all_flat = diff_matrix_all_flat[(diff_matrix_all_flat['StudyNumber_a'] != diff_matrix_all_flat['StudyNumber_b'])]


# Twin pairs
# Select rows where family number a == family number b
diff_matrix_completepairs_flat = diff_matrix_all_flat[(diff_matrix_all_flat['FamilyNumber_a'] == diff_matrix_all_flat['FamilyNumber_b'])]

# MZ Twin pairs
diff_matrix_completepairs_MZ_flat = diff_matrix_all_flat[(diff_matrix_all_flat['FamilyNumber_a'] == diff_matrix_all_flat['FamilyNumber_b'])
                                                      & (diff_matrix_all_flat['ACTUAL_ZYGOSITY_a'] == 'MZ')]
# DZ Twin pairs
diff_matrix_completepairs_DZ_flat = diff_matrix_all_flat[(diff_matrix_all_flat['FamilyNumber_a'] == diff_matrix_all_flat['FamilyNumber_b'])
                                                      & (diff_matrix_all_flat['ACTUAL_ZYGOSITY_a'] == 'DZ')]

# Differences between twins of different families
# Select rows where family number a != family number b
diff_matrix_notrelated_flat = diff_matrix_all_flat[(diff_matrix_all_flat['FamilyNumber_a'] != diff_matrix_all_flat['FamilyNumber_b'])]



# group differences into 1,000 BAU/mL bins and plot histograms
diff_bins = range(0,28000,2000)

hist_all = np.histogram(diff_matrix_all_flat['value'], bins = diff_bins)
hist_completepairs = np.histogram(diff_matrix_completepairs_flat['value'], bins = diff_bins)
hist_completepairs_MZ = np.histogram(diff_matrix_completepairs_MZ_flat['value'], bins = diff_bins)
hist_completepairs_DZ = np.histogram(diff_matrix_completepairs_DZ_flat['value'], bins = diff_bins)
hist_notrelated = np.histogram(diff_matrix_notrelated_flat['value'], bins = diff_bins)
# hist_unpaired = np.histogram(diff_matrix_unpaired_flat['value'], bins = diff_bins)

# sns.histplot(diff_matrix_all_flat['value'], bins=diff_bins)
fig = plt.figure()
sns.histplot(diff_matrix_completepairs_flat['value'], bins=diff_bins, stat = 'probability', color = 'red')
sns.histplot(diff_matrix_notrelated_flat['value'], bins=diff_bins, stat = 'probability', color = 'yellow')
fig = plt.figure()
sns.histplot(diff_matrix_completepairs_MZ_flat['value'], bins=diff_bins, stat = 'probability', color = 'blue')
sns.histplot(diff_matrix_completepairs_DZ_flat['value'], bins=diff_bins, stat = 'probability', color = 'green')

fig, ax = plt.subplots()
sns.ecdfplot(diff_matrix_notrelated_flat['value'], color = 'blue', ax = ax, label = 'Non-related pairs')
sns.ecdfplot(diff_matrix_completepairs_MZ_flat['value'], color = 'red', ax = ax, label = 'Related MZ pairs')
sns.ecdfplot(diff_matrix_completepairs_DZ_flat['value'], color = 'green', ax = ax, label = 'Related DZ pairs')

plt.ylabel('Proportion of pairs with difference < x')
plt.xlabel('Antibody level difference, x (BAU/mL)')
ax.legend()

fig, ax = plt.subplots()
sns.ecdfplot(diff_matrix_notrelated_flat['value'], color = 'blue', ax = ax)
sns.ecdfplot(diff_matrix_completepairs_flat['value'], color = 'black', ax = ax)


mean_completepairs = diff_matrix_completepairs_flat['value'].describe()
mean_completepairs_MZ = diff_matrix_completepairs_MZ_flat['value'].describe()
mean_completepairs_DZ = diff_matrix_completepairs_DZ_flat['value'].describe()
mean_notrelated = diff_matrix_notrelated_flat['value'].describe()


#%% Other statistical tests
# -----------------------------------------------------------------------------
# Comparing distributions of antibody level differences of MZ, DZ and non-related pairs
mannwhitney_pvalue_MZ_DZ = mannwhitneyu(diff_matrix_completepairs_MZ_flat['value'], diff_matrix_completepairs_DZ_flat['value'])
mannwhitney_pvalue_MZ_nonrelated = mannwhitneyu(diff_matrix_completepairs_MZ_flat['value'], diff_matrix_notrelated_flat['value'])


# -----------------------------------------------------------------------------
# Mann-Kendall Trend Test to test presence of trend with time since 3rd vaccination at later weeks. Adapted from https://www.statology.org/mann-kendall-test-python/
# Calculate median by weeks since vacc
trendtest_3vacc_median = data_thriva2_3vacc.groupby('WeeksSinceVacc_3').agg({'Value':['count','median']}).reset_index()


# 2 to 8 weeks
test_data = trendtest_3vacc_median[(trendtest_3vacc_median['WeeksSinceVacc_3'] >= 2)
                                   & (trendtest_3vacc_median['WeeksSinceVacc_3'] <= 8)]['Value']['median']
trendtest_3vacc_2to8weeks = mk.original_test(test_data)

# 5+ weeks
test_data = trendtest_3vacc_median[(trendtest_3vacc_median['WeeksSinceVacc_3'] >= 5)
                                   & (trendtest_3vacc_median['WeeksSinceVacc_3'] <= 16)]['Value']['median']
trendtest_3vacc_5to16weeks = mk.original_test(test_data)

# 8+ weeks
test_data = trendtest_3vacc_median[(trendtest_3vacc_median['WeeksSinceVacc_3'] >= 8)
                                   & (trendtest_3vacc_median['WeeksSinceVacc_3'] <= 16)]['Value']['median']
trendtest_3vacc_8to16weeks = mk.original_test(test_data)

# 11+ weeks
test_data = trendtest_3vacc_median[(trendtest_3vacc_median['WeeksSinceVacc_3'] >= 11)
                                   & (trendtest_3vacc_median['WeeksSinceVacc_3'] <= 16)]['Value']['median']
trendtest_3vacc_11to16weeks = mk.original_test(test_data)





