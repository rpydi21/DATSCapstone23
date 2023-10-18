import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_processing
import squarify  # pip install squarify (algorithm for treemap)
import seaborn as sns
import textwrap
from collections import Counter
plt.ioff()
import plotly.express as px


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
                ax2.set_xticks([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
                ax2.set_xticklabels([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
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
    df_dummies = pd.get_dummies(data_cleaned)

    observed = df_dummies.groupby('health_insurance').size().values.reshape(-1, 1)
    #TESTING
    values = list(data_cleaned.columns.values)

    test_data = data_cleaned[['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking']]
    test = test_data.groupby(['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking'])['chd'].count()
    test1 = test_data.groupby(['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking']).count()

    test1['freq'] = list(test)

    test1.reset_index(inplace = True)
    test1

    for variable in data_cleaned.columns:
        print (variable)
        CrosstabResult = pd.crosstab(index=data_cleaned['chd'], columns=data_cleaned[variable])
        ChiSqResult = chi2_contingency(CrosstabResult)
        print('The P-Value of the ChiSq Test is:', ChiSqResult[1])

    print(CrosstabResult)
    data_cleaned.unique()
    data_cleaned.isna().sum()


    for variable in data_cleaned.columns:
        print(data_cleaned[variable].value_counts())

    data_cleaned = data_cleaned.drop(["hpv_shot"], axis = 1)
    data_cleaned = data_cleaned.dropna()
    # importing the required function
    from scipy.stats import chi2_contingency

    # Performing Chi-sq test


    # P-Value is the Probability of H0 being True
    # If P-Value&gt;0.05 then only we Accept the assumption(H0)



    fig = px.sunburst(test1, path=['chd', 'health_insurance', 'personal_physician', 'skin_cancer', 'smoking'],
                      values='freq',
                      color='freq',
                      color_continuous_scale='rdbu_r',
                      width=960, height=600
                      )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig.show()

    fig = px.treemap(test1, path=[px.Constant("all"), 'chd', 'health_insurance', 'personal_physician',
                                  'skin_cancer', 'smoking'],
                     values='freq',
                     color='freq',
                     color_continuous_scale='viridis',
                     width=960, height=600
                     )
    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
    fig.show()

