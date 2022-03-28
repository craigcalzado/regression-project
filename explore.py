# imports
import seaborn as sns
import matplotlib.pyplot as plt
# function that plots variable pairs and returns a list of the plots
def plot_variable_pairs(train):
    sns.pairplot(train, hue="tenure")
    sns.set(style='whitegrid', palette='muted')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    return train
# function that converts months to complete years
def months_to_years(df):
    df['tenure_years'] = (df['tenure'] / 12).astype(int)
    return df
# function that accepts your dataframe and the name of the columns that hold the continuous and categorical features and outputs 3 different plots for visualizing a categorical variable and a continuous variable.
def plot_categorical_and_continuous_vars(df, continuous_var, categorical_var):
    # Plot the distribution of the continuous variable
    sns.distplot(df[continuous_var])
    plt.show()
    # Plot the distribution of the categorical variable
    sns.boxplot(df[categorical_var])
    plt.show()
    # Plot the bar plot of the continuous and categorical variables
    sns.swarmplot(x=categorical_var, y=continuous_var, data=df)
    plt.show()
    return df


