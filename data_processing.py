import pandas as pd
import numpy as np
def replace_values(data_select_variable, variable_name, original_values, new_values):
    data_select_variable[variable_name] = (data_select_variable[variable_name]
                                           .replace(to_replace = original_values,
                                                    value = new_values)
                                           )

def clean_data ():
    data = pd.read_csv(r"/Users/rohithpydi/Downloads/data.csv")
    data_select_variable = data[["_STATE", "DISPCODE", "PRIMINSR", "PERSDOC3",
                "MEDCOST1", "CHECKUP1", "CVDSTRK3", "CHCSCNC1", "CHCOCNC1", "CHCCOPD3", "ADDEPEV3",
                "CHCKDNY2", "DIABETE4", "EMPLOY1", "PNEUVAC4", "HIVRISK5", "HPVADVC4", "SHINGLE2",
                "_METSTAT", "_RFHLTH", "_TOTINDA", "_MICHD", "_ASTHMS1", "_DRDXAR2", "_RACEPR1",
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
            "_METSTAT" : "metropolitan_status", "_RFHLTH" : "health_status", "_TOTINDA" : "physical activity",
            "_MICHD" : "chd", "_ASTHMS1" : "asthma", "_DRDXAR2" : "arthritis",
            "_RACEPR1" : "race", "_SEX" : "sex", "_AGEG5YR" : "age", "_BMI5" : "bmi",
            "_EDUCAG" : "education", "_INCOMG1" : "income", "_RFMAM22" : "mammogram",
            "_HADCOLN" : "colonoscopy", "_HADSIGM" : "sigmoidoscopy", "_SMOKER3" : "smoking",
            "ALCDAY4" : "days_alcohol_consumed", "AVEDRNK3" : "avg_drink_consumed"
    })
    data_select_variable.reset_index(inplace=True, drop=True)
    variable_name = "state"
    original_values = [1,2,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48,49,50,51,53,54,55,56,66,72,78]
    new_values = ["Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming","Guam","Puerto Rico","Virgin Islands"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "health_insurance"
    original_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 88, 77, 99]
    new_values = ["A plan purchased through an employer or union (including plans purchased through another person´s employer)", "A private nongovernmental plan that you or another family member buys on your own", "Medicare", "Medigap", "Medicaid", "Children´s Health Insurance Program (CHIP)", "Military related health care: TRICARE (CHAMPUS) / VA health care / CHAMP- VA", "Indian Health Service", "State sponsored health plan", "Other government program", "No coverage of any type", "Don’t know/Not Sure", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "personal_physician"
    original_values = [1, 2, 3, 7, 9]
    new_values = ["Yes, only one", "More than one", "No", "Don’t know/Not Sure", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "doctor_visit_ability"
    original_values = [1, 2, 7, 9]
    new_values = ["Could not see doctor 1+ times", "Could see doctor all times", "Don't Know", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "last_visit"
    original_values = [1, 2, 3, 4, 7, 8, 9]
    new_values = ["Within past year (anytime less than 12 months ago)", "Within past 2 years (1 year but less than 2 years ago)", "Within past 5 years (2 years but less than 5 years ago)", "5 or more years ago", "Don’t know/Not sure", "Never", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "stroke"
    original_values = [1, 2, 7, 9]
    new_values = ["Yes", "No", "Don’t know / Not sure", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "skin_cancer"
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "other_cancer"
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name ="copd"
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "depression"
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "kidney_disease"
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "diabetes"
    original_values = [1, 2, 3, 4, 7, 9]
    new_values = ["Yes", "Yes, but female told only during pregnancy—Go to Section 08.01 AGE", "No—Go to Section 08.01 AGE", "No, pre-diabetes or borderline diabetes—Go to Section 08.01 AGE", "Don’t know/Not Sure—Go to Section 08.01 AGE", "Refused—Go to Section 08.01 AGE"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "employment"
    original_values = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    new_values = ["Employed for wages", "Self-employed", "Out of work for 1 year or more", "Out of work for less than 1 year", "A homemaker", "A student", "Retired", "Unable to work", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "pneumonia_shot"
    original_values = [1, 2, 7, 9]
    new_values =["Yes", "No", "Don’t know/Not Sure", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "hiv_risk"
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "hpv_shot"
    original_values = [1, 2, 3, 7, 9]
    new_values = ["Yes", "No—Go to Next Module", "Doctor refused when asked—Go to Next Module", "Don’t know/Not Sure—Go to Next Module", "Refused—Go to Next Module"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "shingles_shot"
    original_values = [1, 2, 7, 9]
    new_values = ["Yes", "No", "Don’t know/Not Sure", "Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "metropolitan_status"
    original_values = [1, 2]
    new_values = ["Metropolitan counties", "Nonmetropolitan counties"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "health_status"
    original_values = [1, 2, 9]
    new_values = ["Good or Better Health", "Fair or Poor Health", "Don’t know/Not Sure Or Refused/Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "physical activity"
    original_values = [1, 2, 9]
    new_values = ["Had physical activity or exercise", "No physical activity or exercise in last 30 days", "Don’t know/Refused/Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "chd"
    original_values = [1.0, 2.0]
    new_values = ["Yes", "No"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "asthma"
    original_values = [1, 2, 3, 9]
    new_values = ["Current", "Former", "Never", "Don’t know/Not Sure Or Refused/Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "arthritis"
    original_values = [1, 2]
    new_values = ["Diagnosed with arthritis", "Not diagnosed with arthritis"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "race"
    original_values = [1, 2, 3, 4, 5, 6, 7]
    new_values = ["White only, non-Hispanic", "Black only, non-Hispanic", "American Indian or Alaskan Native only, Non-Hispanic", "Asian only, non-Hispanic", "Native Hawaiian or other Pacific Islander only, Non-Hispanic", "Multiracial, non-Hispanic", "Hispanic"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "sex"
    original_values = [1, 2]
    new_values = ["Male", "Female"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "age"
    original_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    new_values = ["Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39", "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59", "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79", "Age 80 or older", "Don’t know, Refused, or Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "bmi"
    data_select_variable[variable_name] = data_select_variable[variable_name]/100

    variable_name = "education"
    original_values =[1, 2, 3, 4, 9]
    new_values = ["Did not graduate High School", "Graduated High School", "Attended College or Technical School", "Graduated from College or Technical School", "Don’t know/Not sure/Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "income"
    original_values = [1, 2, 3, 4, 5, 6, 7, 9]
    new_values = ["Less than $15,000", "$15,000 to < $25,000", "$25,000 to < $35,000", "$35,000 to < $50,000", "$50,000 to < $100,000", "$100,000 to < $200,000", "$200,000 or more", "Don’t know/Not sure/Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "mammogram"
    original_values = [1, 2, 9]
    new_values = ["Yes", "No", "Don’t know/Not Sure/Refused"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "colonoscopy"
    original_values = [1, 2]
    new_values = ["Have had a colonoscopy", "Have not had a colonoscopy"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "sigmoidoscopy"
    original_values = [1, 2]
    new_values = ["Have had a sigmoidoscopy", "Have not had a sigmoidoscopy"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "smoking"
    original_values = [1, 2, 3, 4, 9]
    new_values = ["Current smoker - now smokes every day", "Current smoker - now smokes some days", "Former smoker", "Never smoked", "Don’t know/Refused/Missing"]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "days_alcohol_consumed"
    for x in range(len(data_select_variable[variable_name])):
        if str(data_select_variable[variable_name][x])[0] == "1":
            new_value = int(str(data_select_variable[variable_name][x])[1:3]) * 4
            data_select_variable._set_value(x, variable_name, new_value)
        elif str(data_select_variable[variable_name][x])[0] == "2":
            new_value = int(str(data_select_variable[variable_name][x])[1:3])
            data_select_variable._set_value(x, variable_name, new_value)
    original_values = [777, 888, 999]
    new_values = [np.nan, 0, np.nan]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    variable_name = "avg_drink_consumed"
    original_values = [88, 77, 99, np.nan]
    new_values = [0, np.nan, np.nan, 0]
    replace_values(data_select_variable, variable_name, original_values, new_values)

    drinks_consumed_last_30_days = data_select_variable["days_alcohol_consumed"].multiply(data_select_variable["avg_drink_consumed"])
    data_select_variable['drinks_consumed_last_30_days'] = drinks_consumed_last_30_days.values
    data_select_variable = data_select_variable.drop(["days_alcohol_consumed", "avg_drink_consumed"], axis = 1)

    del(drinks_consumed_last_30_days, new_values, new_value, original_values, variable_name, x)
    return data_select_variable