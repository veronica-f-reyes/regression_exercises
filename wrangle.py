 
# Functions to obtain Zillow data from the Codeup Data Science Database: zillow
# This function prepares the data for use in exploration in modeling by dropping nulls
#It returns a pandas dataframe.
#--------------------------------

#This function calls acquire.py to obtain Zillow data from SQL database using a SQL query, 
#and prepares the data be removing null values .  

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

#Custom module: acquire.py
import acquire

#FUNCTION to acquire and prepare Zillow data for exploration and modeling
#------------------------------------------------------------------------
def wrangle_zillow():

    #Use acquire.py to pull data from the Zillow database using a SQL query, create a local csv pandas, and return a pandas DataFrame
    df = acquire.get_zillow_data()

    # Drop all rows with any Null values, assign to df, and verify.
    df = df.dropna()

    #Remove outliers
    df = remove_outliers(df, 2.0, ['calculatedfinishedsquarefeet',
       'taxvaluedollarcnt', 'taxamount','bedroomcnt', 'bathroomcnt'] )

    # Rename
    #df = df.rename(columns = { 'bedroomcnt': 'no_bedrooms',
        
    #})   

# Return the dataframe to the calling code
    return df


# FUNCTION to remove outliers per John Salas
# ------------------------------------------
#   The value for k is a constant that sets the threshold. 
#   Usually, you’ll see k start at 1.5, or 3 or less, depending on how many outliers you want to keep. 
#   The higher the k, the more outliers you keep. Recommend not going beneath 1.5, but this is worth using, 
#   especially with data w/ extreme high/low values

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


#FUNCTION to get min/max from columns in a dataframe
# --------------------------------------------------
# To call this function use:   df.apply(minMax)  (Outside will have to call as:  df.apply(wrangle.minMax))
# x is a list dataframe or 
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

#FUNCTION to get a dataframe with the original columns + scaled columns
# ---------------------------------------------------------------------
def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    
    # new column names
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    
    # Fit the scaler on the train
    scaler.fit(train[columns_to_scale])
    
    # transform train validate and test
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index),
    ], axis=1)
    
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index),
    ], axis=1)
    
    return train, validate, test

###################### Prepare Zillow Data With Split ######################

def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''

    # splits df into train_validate and test using train_test_split() 
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    
    # splits train_validate into train and validate using train_test_split() 
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test


################### Plotting Functions #####################################
#Plots a normalized value count as a percent using catplotS
def category_percentages_by_another_category_col(df, category_a, category_b):
    """
    Produces a .catplot with a normalized value count
    """
    (df.groupby(category_b)[category_a].value_counts(normalize=True)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot, 'data'), x=category_a, y='percent', col=category_b, kind='bar', ))


# FUNCTION to plot a scatterplot passing in two variables
# -------------------------------------------------------
def plot_scatter(a, b):

    ax1 = df.plot.scatter(x=a,y=b,c='Navy')        

    return ax1