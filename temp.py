import pandas as pd
import numpy as np
data = pd.read_csv("../data/data.csv")
data_select_variable = data[["_STATE", "DISPCODE", "PRIMINSR", "PERSDOC3",
            "MEDCOST1", "CHECKUP1", "CVDSTRK3", "CHCSCNC1", "CHCOCNC1", "CHCCOPD3", "ADDEPEV3",
            "CHCKDNY2", "DIABETE4", "EMPLOY1", "PNEUVAC4", "HIVRISK5", "HPVADVC4", "SHINGLE2",
            "_METSTAT", "GENHLTH", "_TOTINDA", "_MICHD", "_ASTHMS1", "_DRDXAR2", "_RACEPR1",
            "_SEX", "_AGEG5YR", "_BMI5", "_EDUCAG", "_INCOMG1", "_RFMAM22", "_HADCOLN",
            "_HADSIGM", "_SMOKER3", "ALCDAY4", "AVEDRNK3"]]
data_select_variable = data_select_variable[data_select_variable["DISPCODE"] == 1100]
data_select_variable = data_select_variable.drop(columns = 'DISPCODE')
data_select_variable = data_select_variable.rename(columns = {"_STATE": "state",
        "PRIMINSR" : "health_insurance", "PERSDOC3" : "personal_physician",
        "MEDCOST1" : "doctor_visit_ability", "CHECKUP1" : "last_visit",
        "CVDSTRK3" : "stroke", "CHCSCNC1" : "skin_cancer", "CHCOCNC1" : "other_cancer",
        "CHCCOPD3" : "copd" , "ADDEPEV3" : "depression", "CHCKDNY2" : "kidney_disease",
        "DIABETE4" : "diabetes", "EMPLOY1" : "employment", "PNEUVAC4" : "pneumonia_shot",
        "HIVRISK5" : "hiv_risk", "HPVADVC4" : "hpv_shot", "SHINGLE2" : "shingles_shot",
        "_METSTAT" : "metropolitan_status", "GENHLTH" : "health_status", "_TOTINDA" : "physical activity",
        "_MICHD" : "chd", "_ASTHMS1" : "asthma", "_DRDXAR2" : "arthritis",
        "_RACEPR1" : "race", "_SEX" : "sex", "_AGEG5YR" : "age", "_BMI5" : "bmi",
        "_EDUCAG" : "education", "_INCOMG1" : "income", "_RFMAM22" : "mammogram",
        "_HADCOLN" : "colonoscopy", "_HADSIGM" : "sigmoidoscopy", "_SMOKER3" : "smoking",
        "ALCDAY4" : "days_alcohol_consumed", "AVEDRNK3" : "avg_drink_consumed"
})
data_select_variable.reset_index(inplace=True, drop=True)

variable_name = "days_alcohol_consumed"
data_select_variable[variable_name] = data_select_variable[variable_name].replace(
    [777, 888, 999], [np.nan, 0, np.nan]
)

# data_select_variable['newcol'] = [int(str(value)[1:3]) * 4 if str(value)[0] == "1" else
#        int(str(value)[1:3]) if str(value)[0] == "2" else
#        value
#        for value in data_select_variable[variable_name]]

# def cleanthis(value):
#     return int(str(value)[1:3]) * 4 if str(value)[0] == "1" else int(str(value)[1:3]) if str(value)[0] == "2" else value
#
# data_select_variable['newcol2'] = data_select_variable[variable_name].apply(cleanthis)
#
#
# value_counts_all = data_select_variable.apply(pd.Series.value_counts).plot(kind = "bar")

import matplotlib.pyplot as plt
# def plot_bars_eda(series):
#     series.value_counts().plot(kind="bar")
#     plt.show()
#
# data_select_variable.apply(plot_bars_eda, axis=1)

for col in data_select_variable.columns:
    data_select_variable[col].value_counts().plot(kind="bar")
    plt.show()
