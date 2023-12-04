# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
import joblib

data = pd.read_csv("../data/data_imputed.csv")
#drop values where other_cancer is equal to don't know
data = data[data.other_cancer != "Don't know / Not Sure / Refused / Missing"]

#get index of categorical features
cat_cols = data.select_dtypes(include='object').columns
cat_cols_idx = [col in cat_cols for col in data.columns]

label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
joblib.dump(label_encoders, 'label_encoders.joblib')
del column, label_encoders, le

target = 'other_cancer'
X = data.drop(target, axis=1)  # Features (all columns except the target)
y = data[target]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
def decision_tree(X_train, X_test, y_train, y_test, class_weight=None, return_clf = False):
    if class_weight is None:
        clf = DecisionTreeClassifier()
    else:
        clf = DecisionTreeClassifier(class_weight=class_weight)
    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    model_results(y_pred, y_test, y_pred_proba)

    if return_clf:
        return clf

def random_forest(X_train, X_test, y_train, y_test, class_weight=None, return_clf = False):
    if class_weight is None:
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight)

    # Train the model
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Make predictions
    y_pred = clf.predict(X_test)

    model_results(y_pred, y_test, y_pred_proba)

    if return_clf:
        return clf
        
def model_results(y_pred, y_test, y_pred_proba):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    #calculate f1 score
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 score:", f1)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print('ROC AUC score:', roc_auc)

    sensitivity = recall_score(y_test, y_pred)
    print('Sensitivity:', sensitivity)  

    precision = precision_score(y_test, y_pred)
    print('Precision:', precision)

    report = classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
def feature_importance(clf):
    importances = clf.feature_importances_

    # Print the name and importance of each feature
    for feature_name, importance in sorted(zip(importances, X.columns), reverse=True):
        print(f"{feature_name}: {importance}")

    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure()

    # Create plot title
    plt.title("Feature Importance")

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    # Show plot
    plt.show()
        
# %%
# Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

print("\nRandom Forest with Undersampling")
clf = random_forest(X_resampled, X_test, y_resampled, y_test, return_clf=True)

joblib.dump(clf, '../model/trained_undersampling_rf.joblib')

print ("Feature Importances:")
feature_importance(clf)
# %%
ratio = 2
majority_sample_size = int(len(data[data[target]==1]) * ratio)
majority_downsampled = resample(data[data[target]==0],
                                replace=False, # sample without replacement
                                n_samples= majority_sample_size, # number of samples to downsample to
                                random_state=42) # reproducible results

# Combine minority class and downsampled majority class
downsampled_data = pd.concat([majority_downsampled, data[data[target]==1]])

# Shuffle the data
downsampled_data = downsampled_data.sample(frac=1, random_state=42)

X_downsampled = downsampled_data.drop(target, axis=1)  # Features (all columns except the target)
y_downsampled = downsampled_data[target]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_downsampled, y_downsampled, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTENC(random_state=42, categorical_features = cat_cols_idx)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nRandom Forest with SMOTENC")
clf = random_forest(X_resampled, X_test, y_resampled, y_test, return_clf=True)

joblib.dump(clf, '../model/trained_smoteNC_rf.joblib')
#%%
print ("Feature Importances:")
feature_importance(clf)

# %% Potential use of importance threshold
# threshold = 0.1
# important_features = []
# for feature_name, importance in sorted(zip(clf.feature_importances_, X.columns), reverse=True):
#     if importance > threshold:
#         important_features.append(importance)
# print(f"Important features: {important_features}")


# %% Testing
# #show prediction for a single patient
# clf = joblib.load('trained_smoteNC_rf.joblib')
# clf = joblib.load('trained_undersampling_rf.joblib')
# print("Prediction for a single patient")
# num = 4
# patient = X_test.iloc[num]
# #remove the target row
# print(patient)
# print("Actual value:", y_test.iloc[num])
# print("Predicted value:", clf.predict([patient]))
# print("Predicted probability:", clf.predict_proba([patient]))

# #print values of patient as string
# def format_series(series):
#     formatted_values = ['"{}"'.format(value) for value in series]
#     result_string = ', '.join(formatted_values)
#     print(result_string)
# format_series(patient)
# %%
