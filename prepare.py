import pandas as pd
import os

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split


# prepare data
def prep_zillow17(df):
    imputer = SimpleImputer(strategy='median')
    for col in df.columns:
        if col not in ['fips', 'yearbuilt']:
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))
    df.rename(columns={'taxvaluedollarcnt': 'tax_value', 'taxamount': 'tax_amount', 'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'area', 'yearbuilt': 'year_built'}, inplace=True)
    df['fips'].fillna(df['fips'].median(), inplace=True)
    df['year_built'].fillna(df['year_built'].median(), inplace=True)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


# function to remove outliers
def remove_outliers(df, k, col):
    k = 1.5
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - k * iqr
    fence_high = q3 + k * iqr
    df = df.loc[(df[col] > fence_low) & (df[col] < fence_high)]
    return df

# outlier removal specific to the data
def remove_outliers_fips(df, k):
    for col in df.columns:
        if col not in ['county_fips', 'year', 'logerror', 'transactiondate']:
            df = remove_outliers(df, k, col)
    return df

# common split data function
def split_dataframe(df):
   train, test = train_test_split(df, test_size=0.2, random_state=789)
   train, validate = train_test_split(train, test_size=0.3, random_state=789)
   return train, validate, test 

   # Telco prepare
def prep_telco(df):
    df = df.drop(columns=['Unnamed: 0', 'internet_service_type_id', 'payment_type_id', 'contract_type_id', 'multiple_lines'])
    dummy_df = pd.get_dummies(df[['gender', 'payment_type', 'contract_type', 'internet_service_type']], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df.drop(columns=['gender', 'payment_type', 'contract_type', 'internet_service_type'])

