import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()
from collections import Counter
import data_processing
import squarify  # pip install squarify (algorithm for treemap)
import seaborn as sns
import textwrap


def univariate_analysis(variable):
    data = data_cleaned.loc[:, variable].dropna()
    frequency = list(Counter(data).values())
    values = list(Counter(data).keys())
    if variable == "state":
        plt.subplots(figsize=(13, 6))
        plt.axis("off")
        squarify.plot(sizes=frequency, label=values, alpha=.5,
                      text_kwargs={'fontsize': 5}, pad=0.1)
        plt.title(variable)
    elif variable == "drinks_consumed_last_30_days":
        table = pd.DataFrame.from_dict(Counter(data), orient='index').reset_index()
        table = table.rename(columns={"index": "values", 0: "frequency"})
        table = table.sort_values("values")

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.pie(table["frequency"])
        fig.subplots_adjust(right=0.4)
        labels = [f"{l} - {s}" for l, s in zip(table["values"], table["frequency"])]
        ax.legend(labels=labels, loc="best", ncols=7, fontsize=6, bbox_to_anchor=(1, 0.5, 0.5, 0.5),
                  title="Legend (Value - # of instances)")
        ax.set_title(variable)
    else:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(values, frequency, color="maroon", width=0.4)
        ax.set_ylabel("Frequency")
        ax.set_title(variable)
        if not (data.dtype == "float64"):
            percentage = [(x / sum(frequency) * 100) for x in frequency]
            offset = max(frequency) * 0.005
            wrap_labels(ax, 8)
            for i in range(len(values)):
                ax.annotate("{:.2f}%".format(percentage[i]), (i-.1, frequency[i] + offset))
    plt.show()


def bivariate_analysis(target, variable):
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
    wrap_labels(ax, 8)
    plt.show()


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=break_long_words))
    len(ax.get_xticklabels())
    ax.set_xticks(list(range(0, len(ax.get_xticklabels()))))
    ax.set_xticklabels(labels, rotation=0)


if __name__ == '__main__':
    data_cleaned = data_processing.clean_data()
    for i in range(len(data_cleaned.columns)):
        variable = data_cleaned.columns[i]
        univariate_analysis(variable)

    # for i in range(len(data_cleaned.columns)):
    #     variable = data_cleaned.columns[i]
    #     target = "chd"
    #     bivariate_analysis(target, variable)
    # bivariate_analysis("chd", "age")
    # univariate_analysis("drinks_consumed_last_30_days")
    # data_cleaned["drinks_consumed_last_30_days"].hist(log = True, bins = 50, edgecolor = "black")
    # data_cleaned.hist()
    # data_cleaned.plot.bar()