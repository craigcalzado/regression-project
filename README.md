# Regression Project - Zillow Analysis
<h2>Objectives:</h2>

- Document code, process, findings, and key takeaways in a Jupyter Notebook Final Report.
- Creation of modules the make that process repeatable and easy to follow.
- Ask exploratory questions of the data that will help understand more about the attributes and drivers of home value.
- Construct a model that perdicts assessed home value for single family properties using regression techniques.
- Make recommendations to a data science team about how to improve predictions.
- Deliver a Report in the form of a 5 minute presentation.
- Answer any questions the audience may have.

<h2>Buisness Goals:</h2>

- Construct an ML Regression model that predict propery tax assessed values of Single Family Properties using attributes of the properties.
- Find the key drivers of property value for single family properties.
- Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
- Make recommendations on what works or doesn't work in prediction these homes' values.

<h2>Audience:</h2>

- Target audience is Zillow data science team.

<h2>Deliverables:</h2>

- Up-to-date github repository with code, notebooks, and data. link: https://github.com/craigcalzado/regression-project
- Live presentation of the project. 
- README.md file with project description.
- Final report in the form of a Jupyter Notebook.
- Necessary modules for project repoduction. Acquire & Prepare Modules (.py)
- Instructions on how to replicate the project.

<h2>Context/Scenario:</h2>

We want to be able to predict the property tax assessed values of Single Family Properties that had a transaction during 2017.

We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.

One last thing, Zach lost the email that told us where these properties were located. Ugh, Zach :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

<h2>Data Dictionary</h2>

| Attribute | Definition | Data Type | Additional Info |
| --- | --- | --- | --- |
| parcelid | Unique identifier for each property | int64 | dropped after 'logerror' and 'transactiondate' concat |
| taxvaluedollarcnt | Total property tax assessed value | float64 |renamed to 'value' |
| taxamount | Total tax amount for the property | float64 | dropped after 'tax_rate' |
| yearbuilt | Original construction date | float64 | changed to 'year' |
| lotsizesquarefeet | Size of property in square feet | float64 | changed to lotsqft |
| bedroomcnt | Number of bedrooms | float64 | changed to bedrooms |
| calculatedfinishedsquarefeet | Size of property in square feet | float64 | changed to sqft |
| bathroomcnt | Number of bathrooms | float64 |changed to bedroom |
| taxamount | Total tax amount for the property | float64 | dropped after 'tax_rate' to prevent data leakage |
| logerror | Difference between property tax and property value | integer | |
| fips | County FIPS code | integer | float64 |
| transactiondate | Date of transaction | object | dropped due to no use |

<h2>Questions:</h2>

- Is there a relationship between square footage and the tax value?
- Is there a relationship between the number of bathrooms and tax value?
- Is there a relationship between the number of bedrooms and tax value?
- Is there a relationship between the lot size and tax value?

<h2>Hypotheses;</h2>

- Alpha = .05 (95% confidence level)

<h3>Hypothesis #1 Square footage vs. Value</h3>

-  $H_0:$ There is no correlation between sqft and the value.
-  $H_a:$ There is a correlation between sqft and the value.

<h3>Hypothesis #2 Bathrooms vs. Value</h3>

-  $H_0:$ There is no correlation between Bathrooms and the value.
-  $H_a:$ There is a correlation between Bathrooms and the value.

<h3>Hypothesis #3 Bedrooms vs. Value</h3>

-  $H_0:$ There is no correlation between Bedrooms and the value.
-  $H_a:$ There is a correlation between Bedrooms and the value.

<h3>Hypoesis #4 Lot Sqft vs. Value</h3>

-  $H_0:$ There is no correlation between Lot Sqft and the value.
-  $H_a:$ There is a correlation between Lot Sqft and the value. 

<h4>All the nulls were rejected on all hypothesis tests. The data is not linearly separable. Pearsonr correlation test and T-test were ran.</h4>

<h2>Executive Summary/Conclusion and Recommendations</h2>

Discovery:

- There were four drivers of property value; Square footage(sqft), Bedrooms, Bathrooms, and lot squar footage(lotsqft).
- LassaLars model was the best out performing model. The baseline was set at $328,336.00 off of actual value. LassaLars model was able to predict the value of the property with $208,614.82 off of the actual value. Thats approximately 120k improvement in property value prediction.
- LassaLars was best model as determined by RMSE and R^2.
- County had an effect on tax value, but was not one of the most important drivers.

<h2>Reproduction</h2>
github: https://github.com/craigcalzado/regression-project
Import modules:
    - wrangle.py (data wrangling)
    - prepare.py (data preparation)
    - project_models.py (All functions for modeling)
Run zillow_project.ipynb

