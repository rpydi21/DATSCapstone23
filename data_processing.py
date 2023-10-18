import pandas as pd
import numpy as np

def clean_data ():
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

    variable_name = "state"
    replace_dict = {
        1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
        8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
        12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
        18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
        23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
        27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
        31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
        35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
        39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
        44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
        47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
        53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
        66: "Guam", 72: "Puerto Rico", 78: "Virgin Islands"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "health_insurance"
    replace_dict = {
        1: "Purchased through employer",
        2: "Private nongovernmental plan",
        3: "Medicare",
        4: "Medigap",
        5: "Medicaid",
        6: "Children´s Health Insurance Program (CHIP)",
        7: "Military related health care",
        8: "Indian Health Service",
        9: "State sponsored health plan",
        10: "Other government program",
        88: "No coverage of any type",
        77: "Don’t know/Not Sure/Refused",
        99: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "personal_physician"
    replace_dict = {
        1: "Yes, only one",
        2: "More than one",
        3: "No",
        7: "Don’t know/Not Sure/Refused",
        9: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "doctor_visit_ability"
    replace_dict = {
        1: "Could not see doctor 1+ times",
        2: "Could see doctor all times",
        7: "Don't Know/Refused",
        9: "Don't Know/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "last_visit"
    replace_dict = {
        1: "Within past year",
        2: "Within 1 and 2 years",
        3: "Within 2 and 5 years",
        4: "5 or more years ago",
        7: "Don’t know/Not sure/Refused",
        8: "Never",
        9: "Don’t know/Not sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "stroke"
    replace_dict = {
        1: "Yes",
        2: "No",
        7: "Don’t know/Not sure/Refused",
        9: "Don’t know/Not sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "skin_cancer"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "other_cancer"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name ="copd"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "depression"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "kidney_disease"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "diabetes"
    replace_dict = {
        1: "Yes",
        2: "Yes, but female told only during pregnancy",
        3: "No",
        4: "Pre-diabetes or borderline diabetes",
        7: "Don’t know/Not Sure/Refused",
        9: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "employment"
    replace_dict = {
        1: "Employed for wages",
        2: "Self-employed",
        3: "Unemployed: 1 year or more",
        4: "Unemployed: less than 1 year",
        5: "A homemaker",
        6: "A student",
        7: "Retired",
        8: "Unable to work",
        9: "Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "pneumonia_shot"
    replace_dict = {
        1: "Yes",
        2: "No",
        7: "Don’t know/Not Sure/Refused",
        9: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "hiv_risk"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "hpv_shot"
    replace_dict = {
        1: "Yes",
        2: "No",
        3: "Doctor refused when asked",
        7: "Don’t know/Not Sure/Refused",
        9: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "shingles_shot"
    replace_dict = {
        1: "Yes",
        2: "No",
        7: "Don’t know/Not Sure/Refused",
        9: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "metropolitan_status"
    replace_dict = {
        1: "Metropolitan counties",
        2: "Nonmetropolitan counties"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "health_status"
    replace_dict = {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor",
        7: "Don’t know/Not Sure Or Refused/Missing",
        9: "Don’t know/Not Sure Or Refused/Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "physical activity"
    replace_dict = {
        1: "Had physical activity",
        2: "No physical activity in last 30 days",
        9: "Don’t know/Refused/Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "chd"
    replace_dict = {
        1.0: "Yes",
        2.0: "No"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "asthma"
    replace_dict = {
        1: "Current",
        2: "Former",
        3: "Never",
        9: "Don’t know/Not Sure Or Refused/Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "arthritis"
    replace_dict = {
        1: "Diagnosed",
        2: "Not diagnosed"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "race"
    replace_dict = {
        1: "White only, non-Hispanic",
        2: "Black only, non-Hispanic",
        3: "American Indian or Alaskan Native only, Non-Hispanic",
        4: "Asian only, non-Hispanic",
        5: "Native Hawaiian or other Pacific Islander only, Non-Hispanic",
        6: "Multiracial, non-Hispanic",
        7: "Hispanic"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "sex"
    replace_dict = {
        1: "Male",
        2: "Female"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "age"
    replace_dict = {
        1: "Age 18 to 24",
        2: "Age 25 to 29",
        3: "Age 30 to 34",
        4: "Age 35 to 39",
        5: "Age 40 to 44",
        6: "Age 45 to 49",
        7: "Age 50 to 54",
        8: "Age 55 to 59",
        9: "Age 60 to 64",
        10: "Age 65 to 69",
        11: "Age 70 to 74",
        12: "Age 75 to 79",
        13: "Age 80 or older",
        14: "Don’t know, Refused, or Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "bmi"
    data_select_variable[variable_name] = data_select_variable[variable_name]/100

    variable_name = "education"
    replace_dict = {
        1: "Did not graduate High School",
        2: "Graduated High School",
        3: "Attended College or Technical School",
        4: "Graduated from College or Technical School",
        9: "Don’t know/Not sure/Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "income"
    replace_dict = {
        1: "Less than $15,000",
        2: "$15,000 to < $25,000",
        3: "$25,000 to < $35,000",
        4: "$35,000 to < $50,000",
        5: "$50,000 to < $100,000",
        6: "$100,000 to < $200,000",
        7: "$200,000 or more",
        9: "Don’t know/Not sure/Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "mammogram"
    replace_dict = {
        1: "Yes",
        2: "No",
        9: "Don’t know/Not Sure/Refused"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "colonoscopy"
    replace_dict = {
        1: "Yes",
        2: "No"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "sigmoidoscopy"
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "smoking"
    replace_dict = {
        1: "Current everyday smoker",
        2: "Current occasional smoker",
        3: "Former smoker",
        4: "Never smoked",
        9: "Don’t know/Refused/Missing"
    }
    data_select_variable[variable_name] = data_select_variable[variable_name].map(replace_dict)

    variable_name = "days_alcohol_consumed"
    data_select_variable[variable_name] = data_select_variable[variable_name].replace(
        [777, 888, 999], [np.nan, 0, np.nan]
    )
    data_select_variable[variable_name] = [int(str(value)[1:3]) * 4 if str(value)[0] == "1" else
           int(str(value)[1:3]) if str(value)[0] == "2" else
           value
           for value in data_select_variable[variable_name]]

    variable_name = "avg_drink_consumed"
    data_select_variable[variable_name] = data_select_variable[variable_name].replace(
        [88, 77, 99], [0, np.nan, np.nan]
    )

    data_select_variable.loc[data_select_variable["days_alcohol_consumed"] == 0, "avg_drink_consumed"] = float(0)
    data_select_variable.loc[data_select_variable["days_alcohol_consumed"] == np.nan, "avg_drink_consumed"] = np.nan

    data_select_variable['drinks_consumed_last_30_days'] = (data_select_variable["days_alcohol_consumed"]
                                                            .multiply(data_select_variable["avg_drink_consumed"])
                                                            .values)
    data_select_variable = data_select_variable.drop(["days_alcohol_consumed", "avg_drink_consumed"], axis = 1)
    del(data, replace_dict, variable_name)

    return data_select_variable

