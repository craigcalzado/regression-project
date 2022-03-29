import pandas as pd
import numpy as np
import os

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from env import host, user, password

def get_zillow17_data(use_cache=True):
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Let me get that for you...")
        return pd.read_csv(filename)
    print("Sorry, nothing on file, let me create one for you...")
    data = 'zillow'
    url = f'mysql+pymysql://{user}:{password}@{host}/{data}'
    query = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
            FROM properties_2017
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)
            WHERE propertylandusedesc IN ('Single Family Residential')
            '''
    zillow17_data = pd.read_sql(query, url)
    zillow17_data.to_csv(filename)
    return zillow17_data

# fuction for lesson
def wrangle_grades():
    """
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    """
    # Acquire data from csv file.
    grades = pd.read_csv("student_grades.csv")
    # Replace white space values with NaN values.
    grades = grades.replace(r"^\s*$", np.nan, regex=True)
    # Drop all rows with NaN values.
    df = grades.dropna()
    # Convert all columns to int64 data types.
    df = df.astype("int")
    return df
# acquire data from csv file

def get_telco_data(use_cache=True):
    filename = "telco.csv"
    if os.path.isfile(filename) and use_cache:
        print("Let me get that for you...")
        return pd.read_csv(filename)
    else:
        print("Sorry, nothing on file, let me create one for you...")
        data = 'telco_churn'
        url = f'mysql+pymysql://{user}:{password}@{host}/{data}'
    query = '''
    SELECT * FROM customers 
    JOIN contract_types USING (contract_type_id) 
    JOIN payment_types USING (payment_type_id) 
    JOIN internet_service_types USING (internet_service_type_id)
    '''
    df = pd.read_sql(query, url)
    return df


def wrangle_telco():
    '''
    Function checks to see if telco.csv already exists, if so it returns the dataframe,
    if not it creates the dataframe and returns it.
    '''
    filename = "telco.csv"
    if os.path.isfile(filename):
        print("Let me get that for you...")
        return pd.read_csv(filename)
    else:
        df = get_telco_data()
    df = df.to_csv(filename)
    return df

def prep_telco(df):
    # create a new dataframe with only the columns we want
    df = df[['customer_id', 'tenure', 'total_charges', 'monthly_charges']]
     # replace blank values and special characters
    df = df.replace(r"^\s*$", np.nan, regex=True)
    # change total charges to float
    df['total_charges'] = df['total_charges'].astype(float)
    # fill missing values
    df['total_charges'] = df['total_charges'].fillna(df['total_charges'].mean())
    return df

def split_dataframe(df):
   train, test = train_test_split(df, test_size=0.2, random_state=789)
   train, validate = train_test_split(train, test_size=0.3, random_state=789)
   return train, validate, test 
   
   
   # all zillow project functions 
def get_zillow_project_data(use_cache=True):
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Let me get that for you...")
        # if Unnamed: 0 column exists, drop it
        df = pd.read_csv(filename)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
        else:
            pass
        return df
    print("Sorry, nothing on file, let me create one for you...")
    data = 'zillow'
    url = f'mysql+pymysql://{user}:{password}@{host}/{data}'
    query = '''
            SELECT parcelid, bathroomcnt, bedroomcnt, finishedsquarefeet12, fips, lotsizesquarefeet, yearbuilt, taxvaluedollarcnt, logerror, transactiondate, taxamount
            FROM properties_2017 
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)
            JOIN predictions_2017 USING (parcelid)
            WHERE propertylandusedesc IN ('Single Family Residential');
            '''
    zillow17_data = pd.read_sql(query, url)
    zillow17_data.to_csv(filename)
    return zillow17_data

def zillow_proj_prep(df):
    # drop columns we don't need parcelid
    df = df.drop(columns=['parcelid'])
    # drop all null values
    df.dropna(subset=['finishedsquarefeet12', 'taxvaluedollarcnt','lotsizesquarefeet'], inplace=True)
    # impute missing values for yearbuilt with mode
    df['yearbuilt'] = df['yearbuilt'].fillna(df['yearbuilt'].mode()[0])
    # drop colums we don't need
    df = df.drop(columns=['taxamount'])
    # rename columns 
    df.rename(columns={'bathroomcnt': 'bathrooms', 'bedroomcnt': 'bedrooms', 'finishedsquarefeet12': 'sqft', 'fips': 'county_fips', 'lotsizesquarefeet': 'lotsqft', 'taxvaluedollarcnt': 'value', 'yearbuilt': 'year'}, inplace=True)
    return df