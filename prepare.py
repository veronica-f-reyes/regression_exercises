#Prepare Telco data using various functions to clean up and section data off 


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import acquire
import prepare
import explore

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

###################### Prepare Telco Data No Dummies ######################
def prep_telco_raw(df):

    '''Prepares acquired Telco data for exploration'''

    # drop unneeded columns using .drop(columns=column_name)
    df = df.drop(columns=['Unnamed: 0', 'contract_type_id.1', 'internet_service_type_id.1','payment_type_id.1'])

    #drop duplicates if there are any
    df = df.drop_duplicates()

    # feature engineering: create new columns to filter by automatic payment and month-to-month churned customers
    df['automatic_pmt'] = np.where(df['payment_type'].str.contains("automatic", case=False), 1, 0)

    #Since customers with null values in the total_charges column have a tenure of 0, they are new customers who have not 
    #been charged yet.  Thus, set the null value in total_charges equal to 0.
    df = df.replace(np.nan, 0, regex=True)
    
    return df


###################### Prepare Telco Data Encoded with Dummies ######################

def prep_telco_encoded(df):

    '''Prepares acquired Telco data for exploration'''

    # drop unneeded columns using .drop(columns=column_name)
    df = df.drop(columns=['Unnamed: 0','customer_id', 'contract_type_id.1', 'internet_service_type_id.1','payment_type_id.1'])

    #drop duplicates if there are any
    df = df.drop_duplicates()

    #Rename values in payment type & contract type before creating dummies
    df.replace({'payment_type':{'Bank transfer (automatic)':'bank_transfer_auto','Credit card (automatic)':'credit_card_auto','Mailed check':'mailed_check','Electronic check':'e_check' }}, inplace=True)
    df.replace({'contract_type':{'Month-to-month':'m_to_m', 'One year':'one_yr', 'Two year':'two_yr'}}, inplace=True)

    # create dummies dataframe using .get_dummies(column_names,not dropping any of the dummy columns)
    features_multi = ['multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies',
                        'contract_type','internet_service_type','payment_type']

    dummy_df = pd.get_dummies(df, columns=features_multi, drop_first=False)
    
    # join original df with dummies df using .concat([original_df,dummy_df])
    df = pd.concat([df, dummy_df], axis=1)

    # feature engineering: create new columns to filter by automatic payment and month-to-month churned customers
    df['automatic_pmt'] = np.where(df['payment_type'].str.contains("auto", case=False), 1, 0)

    #Since customers with null values in the total_charges column have a tenure of 0, they are new customers who have not 
    #been charged yet.  Thus, set the null value in total_charges equal to 0.
    df.total_charges = df.total_charges.replace('', 0)

    # encode features that have string values 
    df.replace({'partner':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'dependents':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'phone_service':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'multiple_lines':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'paperless_billing':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'churn':{'Yes':1, 'No':0}}, inplace=True)
    df.replace({'gender':{'Male':1, 'Female':0}}, inplace=True)

    # Drop unnecessary columns after creating dummy variables to represent them
    df = df.drop(columns=['multiple_lines','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies',
                        'contract_type','internet_service_type','payment_type','internet_service_type_id', 'contract_type_id','payment_type_id'])
    
    
    return df

###################### Prepare Telco Data With Split ######################

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames; stratify on churn.
    return train, validate, test DataFrames.
    '''
    
    # splits df into train_validate and test using train_test_split() stratifying on churn to get an even mix 
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.churn)
    
    # splits train_validate into train and validate using train_test_split() stratifying on churn to get an even mix
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify=train_validate.churn)
    return train, validate, test

#Plots a normalized value count as a percent using catplotS
def category_percentages_by_another_category_col(df, category_a, category_b):
    """
    Produces a .catplot with a normalized value count
    """
    (df.groupby(category_b)[category_a].value_counts(normalize=True)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot, 'data'), x=category_a, y='percent', col=category_b, kind='bar', ))