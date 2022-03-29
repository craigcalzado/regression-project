# imports
import wrangle
import f_engineer
import prepare
import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
import f_engineer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import  mean_squared_error, explained_variance_score
# remove warnings
import warnings
warnings.filterwarnings('ignore')
# create a function that identifies the tax rate for each county
def get_tax_rate(df):
    tax_df = df[['taxamount', 'taxvaluedollarcnt', 'fips']]

    # created a new column for tax rate
    tax_df['tax_rates'] = round((tax_df.taxamount / tax_df.taxvaluedollarcnt) * 100, 2) 

    los_angeles_median = tax_df[tax_df.fips == 6037].tax_rates.median() # get the median tax rate for Los Angeles
    orange_median = tax_df[tax_df.fips == 6059].tax_rates.median() # get the median tax rate for Orange
    ventura_median = tax_df[tax_df.fips == 6111].tax_rates.median() # get the median tax rate for Ventura

    print("Median tax rate for Los Angelos", los_angeles_median)
    print("Median tax rate for Onange", orange_median)
    print("Median tax rate for Ventura", ventura_median)
    return
# create a function for calculating the correlation of a feature
def sample_corr(df, feature):
    corr = df.corr()[feature].sort_values(ascending=False)
    corr = corr.drop(feature)
    plt.figure(figsize=(8,5))
    sns.barplot(corr.index, corr.values)
    plt.xticks(rotation=90)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Correlation', fontsize=15)
    plt.title('Correlation of Features vs. Value', fontsize=15)
    plt.grid(True)
    plt.ylim(0,1)
    plt.show()
# create a function for calculating the correlation of a feature
def pearson_corr(df, target):
    feature = train[['sqft', 'bathrooms', 'bedrooms', 'lotsqft']]
    """
    Calculate the pearson correlation between the target and feature
    """
    # create a loop that calculates the pearson correlation for each feature
    for i in feature.columns:
        # calculate the pearson correlation
        corr = pearsonr(df[target], df[i])
        # print the feature and the correlation
        print(i, corr) 
# create a function for calculating the correlation of a feature
def ttest_corr(df, target):
    feature = train[['sqft', 'bathrooms', 'bedrooms', 'lotsqft']]
    """
    Calculate the ttest correlation between the target and feature
    """
    # create a loop that calculates the ttest correlation for each feature
    for i in feature.columns:
        # calculate the ttest correlation
        corr = ttest_ind(df[target], df[i])
        # print the feature and the correlation
        print(i, corr)
# Create a function that creates your X and y dataframes
def create_X_y(train, validate, test):
    # Create subsets with only predictive features (x)
    X_train = train.drop(columns=['value', 'transactiondate', 'logerror', 'county_fips'])
    y_train = train.value
    X_validate = validate.drop(columns=['value', 'transactiondate', 'logerror', 'county_fips'])
    y_validate = validate.value
    X_test = test.drop(columns=['value', 'transactiondate', 'logerror', 'county_fips'])
    y_test = test.value
    return X_train.shape, y_train.shape, X_validate.shape, y_validate.shape, X_test.shape, y_test.shape
# create a function that scales your X train, validate, and test dataframes
def scale_X(train, validate, test):
    X_train_scaled = f_engineer.scale_minmax(X_train)
    X_validate_scaled = f_engineer.scale_minmax(X_validate)
    X_test_scaled = f_engineer.scale_minmax(X_test)
    return X_train_scaled, X_validate_scaled, X_test_scaled
# create kbest function for X_train and y_train
def kbest_X_y(X, y):
    #create a copy of y_train
    y_train = y.value.copy()
    kbest = SelectKBest(f_regression, k=3)
    kbest.fit(X, y)
    kbest_results = pd.DataFrame(dict(p=kbest.pvalues_, f=kbest.scores_), index=X.columns)
    index = X.columns[kbest.get_support()]
    return print(kbest_results, print(index))
def rmse_r2(y_train, y_validate, y_test):
    # create a median baseline
    rmse_train = mean_squared_error(y_train.value, y_train.price_pred_median) ** .5
    rmse_validate = mean_squared_error(y_validate.value, y_validate.price_pred_median) ** .5
    rmse_test = mean_squared_error(y_test.value, y_test.price_pred_median) ** .5
    # RMSE of price_pred_median (median baseline)
    r2_train = explained_variance_score(y_train.value, y_train.price_pred_median)
    r2_validate = explained_variance_score(y_validate.value, y_validate.price_pred_median)
    p = print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
    "\nValidate/Out-of-Sample: ", round(rmse_validate, 2), 
    "\n",
    "\n",
    "R^2 using Mean\nTrain/In-Sample: ", round(r2_train, 2),
    "\nValidate/Out-of-Sample: ", round(r2_validate, 2)
    )
    plot = sns.scatterplot(x=y_train['value'], y=y_train['price_pred_median'])
    return p, plot
# create a function of the Linear regression, OLS
def lm_ols(y_train, y_validate, y_test):
    # Create the object
    lm = LinearRegression(normalize=True)
    # Fit the object
    lm.fit(X_train_scaled, y_train.value)
    # Use the object
    y_train['price_pred_lm'] = lm.predict(X_train)
    rmse_train = mean_squared_error(y_train.value, y_train.price_pred_lm) ** (1/2)
    y_validate['price_pred_lm'] = lm.predict(X_validate)
    rmse_validate = mean_squared_error(y_validate.value, y_validate.price_pred_lm) ** (1/2)
    r2_train = explained_variance_score(y_train.value, y_train.price_pred_lm)
    r2_validate = explained_variance_score(y_validate.value, y_validate.price_pred_lm)
    p = print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train,2), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate,2),
      "\n",
      "\n",
      "R^2 using Mean\nTrain/In-Sample: ", round(r2_train, 2),
      "\nValidate/Out-of-Sample: ", round(r2_validate, 2))
    plot = sns.scatterplot(x=y_train['value'], y=y_train['price_pred_lm'])
    return p, plot
lm_ols(y_train, y_validate, y_test)
# create a function of the Lasso + Lars
def lars_lasso(y_train, y_validate, y_test):
    # Create the object
    lars = LassoLars(alpha=1)

    # Fit the model to train. 
    # We must specify the column in y_train, 
    # because we have converted it to a dataframe from a series!
    lars.fit(X_train, y_train.value)

    # predict train
    y_train['price_pred_lars'] = lars.predict(X_train)

    # evaluate using rmse
    rmse_train = mean_squared_error(y_train.value, y_train.price_pred_lars) ** (1/2)

    # predict validate
    y_validate['price_pred_lars'] = lars.predict(X_validate)
    y_test['price_pred_lars'] = lars.predict(X_test)

    # evaluate using rmse
    rmse_validate = mean_squared_error(y_validate.value, y_validate.price_pred_lars) ** (1/2)

    r2_train = explained_variance_score(y_train.value, y_train.price_pred_lars)
    r2_validate = explained_variance_score(y_validate.value, y_validate.price_pred_lars)

    p= print("RMSE for Lasso + Lars\nTraining/In-Sample: ", round(rmse_train,2), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate,2),
      "\n",
      "\n",
      "R^2 using Mean\nTrain/In-Sample: ", round(r2_train, 2),
      "\nValidate/Out-of-Sample: ", round(r2_validate, 2))
    plot = sns.scatterplot(x=y_train['value'], y=y_train['price_pred_lars'], alpha=0.5)
    return p, plot
lars_lasso(y_train, y_validate, y_test)
# create a function of the GLM
def glm_tweedie(y_train, y_validate, y_test):
    # Create the object
    glm = TweedieRegressor(power=1, alpha=0)

    # Fit the model to train. 
    # We must specify the column in y_train, 
    # becuase we  converted it to a dataframe from a series! 
    glm.fit(X_train, y_train.value)
    # predict train
    y_train['price_pred_glm'] = glm.predict(X_train)
    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.value, y_train.price_pred_glm) ** (1/2)
    # predict validate
    y_validate['price_pred_glm'] = glm.predict(X_validate)
    y_test['price_pred_glm'] = glm.predict(X_test)
    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.value, y_validate.price_pred_glm) ** (1/2)
    # evaluate: r2
    r2_train = explained_variance_score(y_train.value, y_train.price_pred_glm)
    r2_validate = explained_variance_score(y_validate.value, y_validate.price_pred_glm)
    
    p = print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", round(rmse_train,2), 
      "\nValidation/Out-of-Sample: ", round(rmse_validate,2),
      "\n",
      "\n",
      "R^2 using Mean\nTrain/In-Sample: ", round(r2_train, 2),
      "\nValidate/Out-of-Sample: ", round(r2_validate, 2))
    # plot
    plot = sns.scatterplot(x=y_train['value'], y=y_train['price_pred_glm'], alpha=0.5)
    plot2 = sns.scatterplot(x=y_test['value'], y=y_test['price_pred_lars'])

    # create subplots for the two plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].set_title('GLM Tweedie, power=1 & alpha=0')
    axs[0].set_xlabel('Actual Price')
    axs[0].set_ylabel('Predicted Price')
    axs[0].set_ylim(0, max(y_train.value) + 100)
    axs[0].set_xlim(0, max(y_train.value) + 100)
    axs[0].scatter(y_train.value, y_train.price_pred_glm, alpha=0.5)
    axs[1].set_title('Lasso + Lars')
    axs[1].set_xlabel('Actual Price')
    axs[1].set_ylabel('Predicted Price')
    axs[1].set_ylim(0, max(y_test.value) + 100)
    axs[1].set_xlim(0, max(y_test.value) + 100)
    axs[1].scatter(y_test.value, y_test.price_pred_lars, alpha=0.5)
    return p, plot, plot2
glm_tweedie(y_train, y_validate, y_test)
# create a function that predicts test data
def lars_test(X_test, y_test):
    y_test['price_pred_lars'] = lars.predict(X_test)

    # evaluate using rmse
    rmse_test = mean_squared_error(y_test.value, y_test.price_pred_lars) ** (1/2)
    r2_test = explained_variance_score(y_test.value, y_test.price_pred_lars)

    p = print("RMSE for Lasso + Lars\n",
      "Test/Out-of-Sample: ", round(rmse_test,2),
      "\n",
      "\n",
      "R^2 using Mean\n",
      "Test/Out-of-Sample: ", round(r2_test, 2))
    plot = sns.scatterplot(x=y_test['value'], y=y_test['price_pred_lars'])
    return p, plot
lars_test(X_test, y_test)




    
