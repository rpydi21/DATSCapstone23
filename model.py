#%%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from data_processing import data_cleaned

data = data_cleaned.copy()
data.dropna(inplace = True)

label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
del column, label_encoders, le

X = data.drop('chd', axis=1)  # Features (all columns except the target)
y = data['chd']  # Target variable

#%%

#try clustering

#%% Run decision tree without class weighting (f1 = 0.5824665001243977)
print("Decision Tree without class weighting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

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

# %% Export Decision Tree to png
# from sklearn.tree import export_graphviz
# import graphviz
# import pydotplus

# dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png('decision_tree.png')

#%% Run decision tree with class weighting (f1 = 0.5798864818524276)
print("Decision Tree with class weighting")
from sklearn.utils import class_weight

# Split the data into features and target variable
X = data.drop("chd", axis=1)
y = data["chd"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Fit a decision tree classifier to the training data with class weighting
clf = DecisionTreeClassifier(class_weight=class_weights_dict)
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#calculate f1 score
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 score:", f1)

report = classification_report(y_test, y_pred)
print(report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


#%% Run undersampling to balance the data (f1 = 0.5303245214531039)
print("Undersampling")
from imblearn.under_sampling import RandomUnderSampler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Fit a decision tree classifier to the resampled data with class weighting
clf_resampled = DecisionTreeClassifier()
clf_resampled.fit(X_resampled, y_resampled)

# Make predictions on the testing data
y_pred_resampled = clf_resampled.predict(X_test)

# Evaluate the performance of the model
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
print("Accuracy (resampled):", accuracy_resampled)

#calculate f1 score
f1 = f1_score(y_test, y_pred_resampled, average='macro')
print("F1 score:", f1)

report_resampled = classification_report(y_test, y_pred_resampled)
print(report_resampled)

cm_resampled = confusion_matrix(y_test, y_pred_resampled)
print("Confusion Matrix (resampled):")
print(cm_resampled)

#%% Run oversampling to balance the data (f1 = 0.5816778196480517)
print("Oversampling")
from imblearn.over_sampling import RandomOverSampler

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample the minority class
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Fit a decision tree classifier to the resampled data
clf_resampled = DecisionTreeClassifier()
clf_resampled.fit(X_resampled, y_resampled)

# Make predictions on the testing data
y_pred_resampled = clf_resampled.predict(X_test)

# Evaluate the performance of the model
accuracy_resampled = accuracy_score(y_test, y_pred_resampled)
print("Accuracy (resampled):", accuracy_resampled)

#calculate f1 score
f1 = f1_score(y_test, y_pred_resampled, average='macro')
print("F1 score:", f1)

report_resampled = classification_report(y_test, y_pred_resampled)
print(report_resampled)

cm_resampled = confusion_matrix(y_test, y_pred_resampled)
print("Confusion Matrix (resampled):")
print(cm_resampled)

#%% Run SMOTENC to balance the data (f1 = 0.6565410412248198))
print("SMOTE-NC")
# Import the necessary libraries
from imblearn.over_sampling import SMOTENC
from sklearn.utils import resample

# Downsample majority class to 50000 samples
majority_downsampled = resample(data[data.chd==0],
                                replace=False, # sample without replacement
                                n_samples=50000, # number of samples to downsample to
                                random_state=42) # reproducible results

# Combine minority class and downsampled majority class
downsampled_data = pd.concat([majority_downsampled, data[data.chd==1]])

# Shuffle the data
downsampled_data = downsampled_data.sample(frac=1, random_state=42)

X = downsampled_data.drop('chd', axis=1)  # Features (all columns except the target)
y = downsampled_data['chd']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#get index of categorical features
cat_cols = data_cleaned.select_dtypes(include='object').columns
cat_cols_idx = [data_cleaned.columns.get_loc(col) for col in cat_cols]

# Apply SMOTE to the training data
smote = SMOTENC(random_state=42, categorical_features = cat_cols_idx)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Fit a decision tree classifier to the resampled data
clf = DecisionTreeClassifier()
clf.fit(X_resampled, y_resampled)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#calculate f1 score
f1 = f1_score(y_test, y_pred, average='macro')
print("F1 score:", f1)

report = classification_report(y_test, y_pred)
print(report)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
#%%
#USE F1 MACRO INSTEAD OF ACCURACY
#RECALL
#ROC AND AOC
#more false positives than false negatives
#more sensitivisity than specificity instead of false positives/negatives

#%% standup 10/30
#consider converting numerical to categorical via bins
#random forest feature selection
#one hot encoding
#frequency encoding
#chi square
#try dividing data set by age
