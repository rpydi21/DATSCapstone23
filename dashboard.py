import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

undersampling_rf = joblib.load('../model/trained_undersampling_rf.joblib')
smoteNC_rf = joblib.load('../model/trained_smoteNC_rf.joblib')
loaded_label_encoders = joblib.load('label_encoders.joblib')

app = dash.Dash(__name__)

# Create an empty DataFrame to store selected answers
df_answers = pd.DataFrame(columns=['ID', 'Answer'])
order_list = pd.DataFrame(
    ["state", "health_insurance", "personal_physician", "doctor_visit_ability", "last_visit",
    "stroke", "skin_cancer", "copd", "depression", "kidney_disease", "diabetes", "employment",
    "pneumonia_shot", "hiv_risk", "metropolitan_status", "health_status", "physical activity",
    "chd", "asthma", "arthritis", "race", "sex", "colonoscopy", "sigmoidoscopy", "age", "bmi",
    "education", "income", "smoking", "drinks_consumed_last_30_days"], columns=['ID']
)


# Questions and their corresponding answer choices
dropdown_questions = [
    ("state", "What state do you currently live in?", [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
        'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
        'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
        'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
        'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
        'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
        'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
        'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
        'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Guam', 'Puerto Rico',
        'Virgin Islands'
    ]),
    ("health_insurance", "What is the current primary source of your health insurance?", [
        'Medicare', 'Purchased through employer', 'Military related health care',
        'Private nongovernmental plan', 'Medigap', 'Other government program',
        'No coverage of any type', 'Medicaid', 'State sponsored health plan',
        'Indian Health Service', 'Children´s Health Insurance Program (CHIP)',
        "Don't know / Not Sure / Refused / Missing"
    ]),
    ("personal_physician", "Do you have one person or a group of doctors that you think of as your personal health care provider?", [
        'Yes, only one', 'More than one', 'No', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("doctor_visit_ability", "Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it?", [
        'Could see doctor all times', 'Could not see doctor 1+ times', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("last_visit", "About how long has it been since you last visited a doctor for a routine checkup?", [
        'Within past year', 'Within 1 and 2 years', 'Within 2 and 5 years', '5 or more years ago', 'Never', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("stroke", "Have you ever been told you had a stroke?", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("skin_cancer", "Have you ever been told you had skin cancer that is not melanoma?", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("copd", "Have you ever been told you had C.O.P.D. (chronic obstructive pulmonary disease), emphysema or chronic bronchitis?", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("depression", "Have you ever been told you had a depressive disorder (including depression, major depression, dysthymia, or minor depression)?", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("kidney_disease", "Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("diabetes", "Have you ever been told you had diabetes? If you are a female and had diabetes, was this only when you were pregnant?. If so, respond with 'Yes, but female told only during pregnancy'", [
        'No', 'Yes', 'Pre-diabetes or borderline diabetes', 'Yes, but female told only during pregnancy', 
        "Don't know / Not Sure / Refused / Missing"        
    ]),
    ("employment", "Are you currently _______________?", [
        'Retired', 'Self-employed', 'A homemaker', 'Employed for wages',
        'Unable to work', 'Unemployed: less than 1 year', 'Unemployed: 1 year or more',
        'A student', 'Refused'
    ]),
    ("pneumonia_shot", "Have you ever had a pneumonia shot also known as a pneumococcal vaccine?", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("hiv_risk", "Do any of the following situations apply to you? You do not need to respond to which one. You have injected any drug other than those prescribed for you in the past year. You have been treated for a sexually transmitted disease or STD in the past year.  You have given or received money or drugs in exchange for sex in the past year.", [
        'No', 'Yes', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("metropolitan_status", "What type of county (metropolitan or nonmetropolitan) do you live in?", [
        'Metropolitan counties', 'Nonmetropolitan counties'
    ]),
    ("health_status", "Would you say that in general your health is:", [
        'Very Good', 'Excellent', 'Fair', 'Poor', 'Good', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("physical activity", "Did you do any physical activity or exercise during the past 30 days other than your regular job?", [
        'No physical activity in last 30 days', 'Had physical activity', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("chd", "Have you ever been told you had coronary heart disease (CHD) or a myocardial infarction (MI)?", [
        'No', 'Yes'
    ]),
    ("asthma", "What is your current asthma status?", [
        'Never', 'Current', 'Former', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("arthritis", "Have you ever been diagnosed by a doctor as having some form of arthritis?", [
        'Not diagnosed', 'Diagnosed'
    ]),
    ("race", "What is your race?", [
        'White only, non-Hispanic', 'Black only, non-Hispanic',
        'American Indian or Alaskan Native only, Non-Hispanic',
        'Multiracial, non-Hispanic', 'Hispanic',
        'Native Hawaiian or other Pacific Islander only, Non-Hispanic',
        'Asian only, non-Hispanic'
    ]),
    ("sex", "What is your sex", [
        'Female', 'Male'
    ]),
    ("colonoscopy", "Have you ever had a colonoscopy? If under 45, respond with ‘Age Less than 45’.", [
        'No', 'Yes', 'Age Less than 45'
    ]),
    ("sigmoidoscopy", "Have you ever had a sigmoidoscopy? If under 45, respond with ‘Age Less than 45’.", [
        'No', 'Yes', 'Age Less than 45'
    ]),
    ("age", "What is your age?", [
        '18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49',
        '50 to 54', '55 to 59', '60 to 64', '65 to 69', '70 to 74',
        '75 to 79', '80 or older', "Don't know / Not Sure / Refused / Missing"
    ]),
    # ("What is your height (in inches)?", []),
    # ("What is your weight (in lbs)?", []),
    ("education", "What is your highest level of education completed?", [
        'Graduated from College or Technical School', 'Graduated High School',
        'Attended College or Technical School', 'Did not graduate High School',
        "Don't know / Not Sure / Refused / Missing"
    ]),
    ("income", "What is your household income category?", [
        '$14,999 or less', '$15,000 to < $25,000', '$25,000 to < $35,000',
        '$35,000 to < $50,000', '$50,000 to < $100,000', '$100,000 to < $200,000',
        '$200,000 or more', "Don't know / Not Sure / Refused / Missing"
    ]),
    ("smoking", "What is your smoking status?", [
        'Never smoked', 'Current occasional smoker', 'Former smoker',
        'Current everyday smoker', "Don't know / Not Sure / Refused / Missing"
    ]),
    # ("During the past 30 days, how many days per week or per month did you have at least one drink of any alcoholic beverage?", []),
    # ("One drink is equivalent to a 12-ounce beer, a 5-ounce glass of wine, or a drink with one shot of liquor. During the past 30 days, on the days when you drank, about how many drinks did you drink on the average?  (A 40 ounce beer would count as 3 drinks, or a cocktail drink with 2 shots would count as 2 drinks.)", []),
]

numerical_questions = [
    ("height", "What is your height (in inches)?"),
    ("weight", "What is your weight (in lbs)?"),
    ("days_alc", "During the past 30 days, how many days per week or per month did you have at least one drink of any alcoholic beverage?"),
    ("drinks_alc", "One drink is equivalent to a 12-ounce beer, a 5-ounce glass of wine, or a drink with one shot of liquor. During the past 30 days, on the days when you drank, about how many drinks did you drink on the average?  (A 40 ounce beer would count as 3 drinks, or a cocktail drink with 2 shots would count as 2 drinks.)"),
]

# Flatten the list of questions and choices for dropdown creation
dropdown_options = [(id, question, choice) for id, question, choices in dropdown_questions for choice in choices]

app.layout = html.Div([
    html.H1("Machine Learning for non-Skin Cancer Assessment: Integrating Clinical, Socioeconomic, and Lifestyle Data"),
    html.H2("By: Rohith Pydi"),
    # Dynamic creation of dropdowns based on the specified questions and choices
    *[html.Div([
        html.H3(question),
        dcc.Dropdown(
            id=f'dropdown-{id.replace(" ", "-")}',
            options=[{'label': choice, 'value': choice} for choice in choices],
            placeholder=f'Select answer choice',
        ),
    ]) for id, question, choices in dropdown_questions],

    # Dynamic creation of numerical inputs based on the specified questions
    *[html.Div([
        html.H3(question),
        dcc.Input(
            id=f'input-{id}',
            type='number',
            placeholder=f'Enter answer'
        ),
    ]) for id, question in numerical_questions],

    html.Button('Submit Answers', id='button'),

    html.Div(id='output-text', style={'fontSize': 30}),  # Display the selected answers

    html.Hr(),  # Add a horizontal line

    # html.H2("Selected Answers DataFrame:"),

    # dash_table.DataTable(
    #     id='datatable',
    #     columns=[{'name': col, 'id': col} for col in df_answers.columns],
    #     data=df_answers.to_dict('records')
    # )
])

# Callback to update the output text and DataFrame based on the selected answers
@app.callback(
    # [Output('output-text', 'children'), 
    #  Output('datatable', 'data')],
    Output('output-text', 'children'),
    [Input('button', 'n_clicks')],
    [State(f'dropdown-{id.replace(" ", "-")}', 'value') for id, _, _ in dropdown_questions],
    [State(f'input-{id}', 'value') for id, _ in numerical_questions]
)

def update_output(n_clicks, *selected_answers):
    if n_clicks is not None:
        if all(selected_answers):
            # Display the selected answers in the output text
            # output_text = "You clicked the button! Chosen answers:\n"
            # + "\n".join([f"{question}: {answer} : {id}" for question, answer, id in zip([question for question, _, _ in questions], selected_answers, [id for _, _, id in questions])])

            # Append the selected answers to the DataFrame
            global df_answers
            df_answers = pd.DataFrame(columns=['ID', 'Answer'])
            df_answers = pd.concat([df_answers, 
                                    pd.DataFrame({'ID': [id for id, _, _ in dropdown_questions],
                                                'Answer': selected_answers[:len(dropdown_questions)]}), 
                                    pd.DataFrame({'ID': [id for id, _ in numerical_questions],
                                                'Answer': selected_answers[len(dropdown_questions):]})
                                                ], ignore_index=True)


            # # Define the data for each column
            # data = {
            #     "ID": ["state", "health_insurance", "personal_physician", "doctor_visit_ability", "last_visit",
            #     "stroke", "skin_cancer", "copd", "depression", "kidney_disease", "diabetes", "employment",
            #     "pneumonia_shot", "hiv_risk", "metropolitan_status", "health_status", "physical activity",
            #     "chd", "asthma", "arthritis", "race", "sex", "colonoscopy", "sigmoidoscopy", "age",
            #     "education", "income", "smoking", "height", "weight", "days_alc", "drinks_alc"],
            #     "Answer": ["Pennsylvania", "Purchased through employer", "Yes, only one", "Could see doctor all times", 
            #                "Within past year", "No", "No", "No", "Yes", "No", "No", "Employed for wages", "No", "No",
            #                  "Metropolitan counties", "Very Good", "Had physical activity", "No", "Never", "Not diagnosed", 
            #                  "White only, non-Hispanic", "Male", "Age Less than 45", "Age Less than 45", "35 to 39",  
            #                  "Graduated from College or Technical School", "$50,000 to < $100,000", "Former smoker", 70, 206.6,
            #                   2, 30]
            # }

            # # Create the DataFrame
            # df_answers = pd.DataFrame(data)
            
            #calculate bmi and append to df_answers
            height = df_answers[df_answers['ID'] == 'height']['Answer'].values[0]
            weight = df_answers[df_answers['ID'] == 'weight']['Answer'].values[0]
            bmi = round((weight / (height * height)) * 703, 2)
            df_answers = pd.concat([df_answers, 
                                    pd.DataFrame({'ID': 'bmi', 'Answer': [bmi]})], ignore_index=True)

            #calculate drinks_consumed_last_30_days and append to df_answers
            days_alc = df_answers[df_answers['ID'] == 'days_alc']['Answer'].values[0]
            drinks_alc = df_answers[df_answers['ID'] == 'drinks_alc']['Answer'].values[0]
            drinks_consumed_last_30_days = round(days_alc * drinks_alc, 2)
            df_answers = pd.concat([df_answers, 
                                    pd.DataFrame({'ID': 'drinks_consumed_last_30_days', 'Answer': [drinks_consumed_last_30_days]})], ignore_index=True)
            
            values = ['height', 'weight', 'days_alc', 'drinks_alc']
            df_answers = df_answers[~df_answers['ID'].isin(values)]

            df_answers = df_answers.set_index('ID')
            df_answers = df_answers.reindex(index=order_list['ID'])
            df_answers = df_answers.reset_index()

            #encode answers
            df_answers_encoded = df_answers.copy()
            df_answers_encoded = df_answers_encoded.set_index('ID')
            df_answers_encoded = df_answers_encoded.transpose()

            df_answers_encoded['bmi'] = df_answers_encoded['bmi'].astype(float)
            df_answers_encoded['drinks_consumed_last_30_days'] = df_answers_encoded['drinks_consumed_last_30_days'].astype(float)

            # Reset the index
            df_answers_encoded.reset_index(drop=True, inplace=True)

            # use loaded label encoder to transform data
            for column in df_answers_encoded.columns:
                if df_answers_encoded[column].dtype == 'object':
                    le = loaded_label_encoders[column]
                    df_answers_encoded[column] = le.transform(df_answers_encoded[column])
            
            patient = df_answers_encoded.iloc[0]
            # prediction1 = undersampling_rf.predict([patient])
            probability1 = undersampling_rf.predict_proba([patient])
            
            # prediction2 = smoteNC_rf.predict([patient])
            probability2 = smoteNC_rf.predict_proba([patient])

            probability = (probability1 + probability2) / 2.0
            prediction = probability.argmax(axis=1)
            
            if prediction == 0:
                output_text = "You are likely not at risk for cancer. Your probability of having cancer is " + str(round(probability[0][1] * 100, 2)) + "%."
            else:
                output_text = "You are likely at risk for cancer. Your probability of having cancer is " + str(round(probability[0][1] * 100, 2)) + "%."
            
            # Return updated output text and DataFrame data
            return output_text
            # , df_answers.to_dict('records')
        else:
            return "Please provide an answer for each question before clicking the button.", df_answers.to_dict('records')

    return '', df_answers.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True)
