#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing import data_cleaned
import squarify  # pip install squarify (algorithm for treemap)
import seaborn as sns
import textwrap
from collections import Counter
import plotly.express as px
import scipy.stats as stats

#%%
def greater_than_0 (value): return value if value == 0 else 1

def univariate_analysis(variable):
    if variable.dtype == "float64":
        if variable.name == "drinks_consumed_last_30_days":
            fig, axes = plt.subplots(1, 2, figsize = (13,6))

            altered_data = pd.DataFrame(variable.apply(greater_than_0), dtype="str")
            ax1 = sns.histplot(altered_data, x=variable.name, ax=axes[0])
            ax1.set_xticks(range(len(ax1.get_xticklabels())))
            ax1.set_xlabel(variable.name)
            ax1.set_xticklabels(["0", "1 or Greater"])

            ax2 = sns.histplot(variable[variable>0], log_scale= True, ax = axes[1])
            ax2.set_xticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
            ax2.set_xticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
        else:
            fig, ax = plt.subplots(figsize=(13, 6))
            ax = sns.histplot(variable, log_scale= True)
            ax.set_title(variable.name)
            ax.set(xlabel=None)
    else:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.set_title(variable.name)
        ax.set(xlabel=None)
        ax = sns.countplot(variable, x = variable.values)
        labels = [textwrap.fill(label.get_text(), 10, break_long_words=False) for label in ax.get_xticklabels()]
        ax.set_xticks(list(range(0, len(labels))))
        if variable.name == "state":
            ax.set_xticklabels(labels, rotation=45, fontsize = 6)
        else:
            ax.set_xticklabels(labels)
    plt.show()

def bivariate_analysis(variable):
    target = "chd"
    data = pd.concat([variable, data_cleaned[target]], axis = 1)
    if variable.dtype == "float64":
        fig, ax = plt.subplots(figsize=(13, 6))
        sns.boxplot(data, x = target, y = variable)
        ax.set_title(variable.name)
        ax.set(xlabel=None)
    elif not(variable.name == target):
        fig, ax = plt.subplots(figsize=(13, 6))
        ax = sns.countplot(data, x = variable.name, hue = target)
        ax.set_title(variable.name)
        ax.set(xlabel=None)
        labels = [textwrap.fill(label.get_text(), 10, break_long_words=False) for label in ax.get_xticklabels()]
        ax.set_xticks(list(range(0, len(labels))))
        if variable.name == "state":
            ax.set_xticklabels(labels, rotation=45, fontsize = 6)
        else:
            ax.set_xticklabels(labels)


#%%
data_cleaned.apply(univariate_analysis)
data_cleaned.apply(bivariate_analysis)

#%%

def chi_square (variable):
    data = pd.crosstab(variable, data_cleaned["chd"])

    # Perform a chi-squared test
    chi2, p, _, _ = stats.chi2_contingency(data)

    # Check for statistical significance
    print(variable.name)
    if p < 0.05:
        print("There is a significant association between the variables.")
    else:
        print("There is no significant association between the variables.")
        

#testing
data_cleaned["health_insurance"].nunique()

data_cleaned.apply(chi_square)

# pd.crosstab( data_cleaned["health_insurance"], columns = data_cleaned["chd"])
# test = data_cleaned.loc[:, data_cleaned.columns != 'chd']






# df_dummies = pd.get_dummies(data_cleaned)

# observed = df_dummies.groupby('health_insurance').size().values.reshape(-1, 1)

# values = list(data_cleaned.columns.values)

# test_data = data_cleaned[['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking']]
# test = test_data.groupby(['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking'])['chd'].count()
# test1 = test_data.groupby(['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking']).count()

# test1['freq'] = list(test)

# test1.reset_index(inplace = True)
# test1

# for variable in data_cleaned.columns:
#     print (variable)
#     CrosstabResult = pd.crosstab(index=data_cleaned['chd'], columns=data_cleaned[variable])
#     ChiSqResult = chi2_contingency(CrosstabResult)
#     print('The P-Value of the ChiSq Test is:', ChiSqResult[1])

# print(CrosstabResult)
# data_cleaned.unique()
# data_cleaned.isna().sum()


# for variable in data_cleaned.columns:
#     print(data_cleaned[variable].value_counts())

# data_cleaned = data_cleaned.drop(["hpv_shot"], axis = 1)
# data_cleaned = data_cleaned.dropna()
# # importing the required function
# from scipy.stats import chi2_contingency

# # Performing Chi-sq test


# # P-Value is the Probability of H0 being True
# # If P-Value&gt;0.05 then only we Accept the assumption(H0)



# fig = px.sunburst(test1, path=['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking'],
#                     values='freq',
#                     color='freq',
#                     color_continuous_scale='rdbu_r',
#                     width=960, height=600
#                     )
# fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
# fig.show()

# fig = px.treemap(test1, path=[px.Constant("all"), 'chd', 'health_insurance', 'personal_physician',
#                                 'skin_cancer', 'smoking'],
#                     values='freq',
#                     color='freq',
#                     color_continuous_scale='viridis',
#                     width=960, height=600
#                     )
# fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
# fig.show()

