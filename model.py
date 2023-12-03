# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import recall_score, precision_score

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

    report = classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print('ROC AUC score:', roc_auc)

    sensitivity = recall_score(y_test, y_pred)
    print('Sensitivity:', sensitivity)  

    precision = precision_score(y_test, y_pred)
    print('Precision:', precision)

def feature_importance(clf):
    importances = clf.feature_importances_

    # Print the name and importance of each feature 
    for feature_name, importance in zip(X.columns, importances):
        print(f"{feature_name}: {importance}")
    #sort features by importance
    sorted(zip(importances, X.columns), reverse=True)

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
        

# %% [markdown]
# 

# %%
print("Decision Tree without class weighting")
decision_tree(X_train, X_test, y_train, y_test)

print("\nRandom Forest without class weighting")
random_forest(X_train, X_test, y_train, y_test)

# %%
# from sklearn.tree import export_graphviz
# import graphviz
# import pydotplus

# dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('decision_tree.png')

# %%
# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print("Decision Tree with class weighting")
decision_tree(X_train, X_test, y_train, y_test, class_weights_dict)

print("\nRandom Forest without class weighting")
random_forest(X_train, X_test, y_train, y_test, class_weights_dict)

# %%
# Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

print("Decision Tree with Undersampling")
decision_tree(X_resampled, X_test, y_resampled, y_test)

print("\nRandom Forest with Undersampling")
random_forest(X_resampled, X_test, y_resampled, y_test)

# %%
# Oversample the minority class
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("Decision Tree with Oversampling")
decision_tree(X_resampled, X_test, y_resampled, y_test)

print("\nRandom Forest with Oversampling")
random_forest(X_resampled, X_test, y_resampled, y_test)

# %%
ratio = 4
majority_sample_size = int(len(data[data[target]==1]) * ratio)
# Downsample majority class to 50000 samples
majority_downsampled = resample(data[data[target]==0],
                                replace=False, # sample without replacement
                                n_samples= majority_sample_size, # number of samples to downsample to
                                random_state=42) # reproducible results

# Combine minority class and downsampled majority class
downsampled_data = pd.concat([majority_downsampled, data[data[target]==1]])

# Shuffle the data
downsampled_data = downsampled_data.sample(frac=1, random_state=42)

X = downsampled_data.drop(target, axis=1)  # Features (all columns except the target)
y = downsampled_data[target]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTENC(random_state=42, categorical_features = cat_cols_idx)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# %%
# print("Decision Tree with SMOTENC")
# clf = decision_tree(X_resampled, X_test, y_resampled, y_test, return_clf=True)
# print ("Feature Importances:")
# feature_importance(clf)

print("\nRandom Forest with SMOTENC")
clf = random_forest(X_resampled, X_test, y_resampled, y_test, return_clf=True)
print ("Feature Importances:")
feature_importance(clf)

# %%
#USE F1 MACRO INSTEAD OF ACCURACY
#RECALL
#ROC AND AOC
#more false positives than false negatives
#more sensitivisity than specificity instead of false positives/negatives


# %%
#consider converting numerical to categorical via bins
#random forest feature selection
#one hot encoding
#frequency encoding
#chi square
#try dividing data set by age