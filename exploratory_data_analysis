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
            title = "Histogram of " + variable.name
            ax.set_title(title, fontsize = 20)
            ax.set(xlabel=None)
    else:
        fig, ax = plt.subplots(figsize=(13, 6))
        title = "Barplot of " + variable.name
        ax.set_title(title, fontsize = 20)
        ax.set(xlabel=None)
        if variable.name == "age":
            order = ['18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54' , '55 to 59',
                     '60 to 64', '65 to 69', '70 to 74', '75 to 79', '80 or older', "Don't know / Not Sure / Refused / Missing"]
            ax = sns.countplot(variable, x = variable.values, order = order)
        else:
            ax = sns.countplot(variable, x = variable.values)
        labels = [textwrap.fill(label.get_text(), 5, break_long_words=False) for label in ax.get_xticklabels()]
        if variable.name == "age":
            #keep all labels same except for last one
            labels = labels[:-1]
            labels.append("Don't \nknow /\n Not Sure /\n Refused /\n Missing")
        ax.set_xticks(list(range(0, len(labels))))
        if variable.name == "state":
            ax.set_xticklabels(labels, rotation=45, fontsize = 6)
        else:
            ax.set_xticklabels(labels, fontsize = 15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_ylabel("Count", fontsize = 20)
        
    plt.show()

def bivariate_analysis(variable):
    target = "other_cancer"
    data = pd.concat([variable, data_imputed[target]], axis = 1)
    if variable.dtype == "float64":
        fig, ax = plt.subplots(figsize=(13, 6))
        #do violin plot
        sns.violinplot(data, x = target, y = variable.name)
        # sns.boxplot(data, x = variable, y = target)
        title = "Violin Plot of " + variable.name + " and " + target
        ax.set_title(title, fontsize = 20)
        #make 10 y ticks, round min and max to nearest 10
        min = np.floor(variable.min()/10)*10
        max = np.ceil(variable.max()/10)*10
        yticks = np.linspace(min, max, 10)
        ax.set_yticks(yticks)

        # ax.set(xlabel=None)
    elif not(variable.name == target):
        fig, ax = plt.subplots(figsize=(13, 6))
        if variable.name == "age":
            print("test")
            order = ['18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54' , '55 to 59',
                     '60 to 64', '65 to 69', '70 to 74', '75 to 79', '80 or older', "Don't know / Not Sure / Refused / Missing"]
            ax = sns.countplot(data, x = variable.name, hue = target, order = order)
        else:
            ax = sns.countplot(data, x = variable.name, hue = target)
        title = "Countplot of " + variable.name + " and " + target
        ax.set_title(title, fontsize = 20)
        ax.set(xlabel=None)
        labels = [textwrap.fill(label.get_text(), 5, break_long_words=False) for label in ax.get_xticklabels()]
        if variable.name == "age":
            #keep all labels same except for last one
            labels = labels[:-1]
            labels.append("Don't know /\n Not Sure /\n Refused /\n Missing")
        ax.set_xticks(list(range(0, len(labels))))
        if variable.name == "state":
            ax.set_xticklabels(labels, rotation=45, fontsize = 6)
        else:
            ax.set_xticklabels(labels)
    plt.show()
#%%
data_imputed = pd.read_csv("../data/data_imputed_R.csv")

data_imputed = data_imputed[["other_cancer", "bmi", "age"]]

plots = data_imputed.apply(univariate_analysis)
plots = data_imputed.apply(bivariate_analysis)