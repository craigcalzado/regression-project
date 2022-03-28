
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, RFE, f_regression
import sklearn.preprocessing as skl_pp

# function to scale data to 0-1 range(MinMaxScaler)
def scale_minmax(df):
    scaler = skl_pp.MinMaxScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

def scale_data(train, validate, test, y, return_scaler=False):
    '''
    Scales the 3 data splits.
    
    takes in the train, validate, and test data splits and returns their scaled counterparts.
    
    If return_scaler is true, the scaler object will be returned as well.

    X = columns to scale (not target)
    '''
    columns_to_scale = train.columns.tolist()
    
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    scaler = skl_pp.MinMaxScaler()
    scaler.fit(train[columns_to_scale])
    
    train_scaled[columns_to_scale] = scaler.transform(train[columns_to_scale])
    validate_scaled[columns_to_scale] = scaler.transform(validate[columns_to_scale])
    test_scaled[columns_to_scale] = scaler.transform(test[columns_to_scale])
    
    X_train_df = pd.DataFrame(train_scaled[columns_to_scale], columns=train.columns)
    y_train = train_scaled[y]
    X_validate_df = pd.DataFrame(validate_scaled[columns_to_scale], columns=validate.columns)
    y_validate = validate_scaled[y]
    X_test_df = pd.DataFrame(test_scaled[columns_to_scale], columns=test.columns)
    y_test = test_scaled[y]

    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return X_train_df, y_train, X_validate_df, y_validate, X_test_df, y_test



# function to input scaled data into a kbest
def select_kbest(X, y, k):
    kbest = SelectKBest(f_regression, k=k)
    kbest.fit(X, y)
    return X.columns[kbest.get_support()]

# function to input scaled data into a RFE
def rfe(X, y, k):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k)
    rfe.fit(X, y)
    feature_mask = rfe.support_
    rfe_feature = X.iloc[:,feature_mask].columns.tolist()
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    return rfe_feature, rfe_ranks_df.sort_values('Rank')