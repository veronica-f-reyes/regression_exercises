 
# Functions to obtain Zillow data from the Codeup Data Science Database: zillow
#It returns a pandas dataframe.
#--------------------------------



#This function uses my user info from my env file to create a connection url to access the Codeup db.  

from typing import Container
import pandas as pd
import os
from env import host, user, password

#FUNCTION to connect to database for SQL query use
# -------------------------------------------------
def get_db_url(host, user, password, database):
        
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
    return url


#FUNCTION to get data from zillow database
# ----------------------------------------
def get_zillow_data():
    
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:

        database = 'zillow'

        #Create SQL query to select data from zillow database
        query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid = 261;


                
                '''

         # read the SQL query into a dataframe
        df = pd.read_sql(query, get_db_url(host,user, password, database))

         # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df



#CALL function to get and create zillow.csv locally
get_zillow_data()





