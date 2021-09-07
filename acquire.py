 
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






#This function uses my user info from my env file to create a connection url to access the Codeup db.  

from typing import Container
import pandas as pd
import os
from env import host, user, password

#Function to connect to database for SQL query use
def get_db_url(host, user, password, database):
        
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    
    return url


#Function to get data from telco_churn database
def get_telco_data():
    
    filename = "telco_churn.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:

        database = 'telco_churn'

        #Create SQL query to select data from telco_churn database
        query = '''
                SELECT * FROM customers
                JOIN contract_types ON customers.contract_type_id = contract_types.contract_type_id
                JOIN internet_service_types ON customers.internet_service_type_id = internet_service_types.internet_service_type_id
                JOIN payment_types ON customers.payment_type_id = payment_types.payment_type_id ;
                
                '''

         # read the SQL query into a dataframe
        df = pd.read_sql(query, get_db_url(host,user, password, database))

         # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

#Call function to get and create telco_churn.csv locally
get_telco_data()
