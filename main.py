#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from collections import Counter
import plotly.express as px
import scipy.stats as stats

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
    target = "other_cancer"
    data = pd.concat([variable, data_imputed[target]], axis = 1)
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
    plt.show()
#%%
data_imputed = pd.read_csv("../data/data_imputed.csv")
data_imputed = data_imputed[data_imputed.other_cancer != "Don't know / Not Sure / Refused / Missing"]
univariate_analysis(data_imputed["other_cancer"])
plots = data_imputed.apply(univariate_analysis)
plots = data_imputed.apply(bivariate_analysis)

#%% chi-squre (likely not used)
#run chi-square test with chd as target variable and return only p value
# def chi_square (variable):
#     data = pd.crosstab(variable, data_cleaned["chd"])

#     # Perform a chi-squared test
#     chi2, p_value, dof, expected = stats.chi2_contingency(data)

#     # Check for statistical significance
#     #display chi2 (rounded to 2 decimal points) and p-value (rounded to 3 decimal points) as well as if its significant or not (p-value < 0.05)
#     if p_value < 0.05:
#         return f"chi2: {round(chi2, 2)}, p-value: {round(p_value, 3)}, significant"
#     else:
#         return f"chi2: {round(chi2, 2)}, p-value: {round(p_value, 3)}, not significant"

# # Apply the chi_square function to each column in data_cleaned
# result = data_cleaned.apply(chi_square)

# # Print the result
# print(result)
#%%