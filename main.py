import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_processing
import squarify  # pip install squarify (algorithm for treemap)
import seaborn as sns
import textwrap
from collections import Counter
plt.ioff()


def univariate_analysis():
    for variable in data_cleaned.columns:
        fig, ax = plt.subplots(figsize=(13, 6))
        if variable == "state":
            data = data_cleaned.loc[:, variable].dropna()
            plt.axis("off")
            ax = squarify.plot(sizes=list(Counter(data).values()), label=list(Counter(data).keys()), alpha=.5,
                               text_kwargs={'fontsize': 5}, pad=0.1)

        elif data_cleaned[variable].dtype == "float64":
            if variable == "drinks_consumed_last_30_days":
                fig, axes = plt.subplots(1, 2)
                altered_data = data_cleaned[variable]
                altered_data = pd.DataFrame({"value": [value if value == 0 else 1 for value in altered_data]},
                                            dtype="str")
                ax1 = sns.histplot(altered_data, x="value", ax=axes[0])
                ax1.set_xticks(range(len(ax1.get_xticklabels())))
                ax1.set_xlabel(variable)
                ax1.set_xticklabels(["0", "1 or Greater"])
                ax2 = sns.histplot(data_cleaned, x=variable, log_scale=True, ax=axes[1])
                ax2.set_xticks([1, 100])
            else:
                ax = sns.histplot(data_cleaned, x=variable)
        else:
            ax = sns.countplot(data=data_cleaned, x=variable)
            labels = [textwrap.fill(label.get_text(), 10, break_long_words=False) for label in ax.get_xticklabels()]
            ax.set_xticks(list(range(0, len(labels))))
            ax.set_xticklabels(labels, rotation=0)
        ax.set_title(variable)
        ax.set(xlabel=None)


def bivariate_analysis():
    target = "chd"
    for variable in data_cleaned.columns:
        if variable == target or variable == "state":
            continue
        cross_tab = pd.crosstab(data_cleaned[variable], data_cleaned[target])
        fig, ax = plt.subplots(figsize=(13, 6))
        width = 0.25  # the width of the bars
        multiplier = 0
        x = np.arange(len(cross_tab.index))
        for attribute, measurement in cross_tab.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3, rotation=90)
            multiplier += 1

        ax.set_ylabel("Frequency")
        ax.set_title(variable)
        ax.set_xticks(x + width, cross_tab.index)
        ax.legend(loc="upper right", title="CHD?")
        labels = [textwrap.fill(label.get_text(), 10, break_long_words=False) for label in ax.get_xticklabels()]
        ax.set_xticks(list(range(0, len(labels))))
        ax.set_xticklabels(labels, rotation=0)
        plt.show()


if __name__ == '__main__':
    data_cleaned = data_processing.clean_data()

    univariate_analysis()
    bivariate_analysis()
