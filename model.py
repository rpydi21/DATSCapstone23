# %%
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
from sklearn.utils import class_weight, resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTENC
import joblib
import seaborn as sns
#%%
data = pd.read_csv("../data/data_imputed_R.csv")

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
def random_forest(X_train, X_test, y_train, y_test, return_clf = False):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Make predictions
    y_pred = clf.predict(X_test)

    model_results(y_pred, y_test, y_pred_proba)

    if return_clf:
        return clf
        
def model_results(y_pred, y_test, y_pred_proba):
    sensitivity = recall_score(y_test, y_pred)
    print('Sensitivity:', sensitivity)

    report = classification_report(y_test, y_pred)
    print(report)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    print(f'True Positives: {tp}')
    print(f'True Negatives: {tn}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')

    cm = np.flip(cm, axis=0)
    cm = np.flip(cm, axis=1)
    cm_copy = cm.copy()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #create heatmap for confusion matrix
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'flare', annot_kws={"size": 20})


    labels = (np.asarray(["{0:.0f}\n".format(value) for value in cm_copy.flatten()]))
    #reshape label as per the confusion matrix dimensions
    labels = labels.reshape(2,2)
    #add label to each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i][j] > 0.240:
                text_color = "white"
            else:
                text_color = "black"
            plt.annotate("Raw Count: " + labels[i][j], xy=(j+0.5, i+0.64), horizontalalignment='center', verticalalignment='center', size = 17, color = text_color)

    plt.ylabel('Actual label', fontsize = 20)
    plt.xlabel('Predicted label', fontsize = 20)
    #change x-axis labels to be more readable
    plt.xticks([0.5, 1.5], ['Yes', 'No'], fontsize = 20)
    #change y-axis labels to be more readable
    plt.yticks([0.5, 1.5], ['Yes', 'No'], fontsize = 20)
    all_sample_title = 'Recall Score: {0:.4f}'.format(sensitivity)
    plt.title(all_sample_title, size = 25)
    plt.show()
    
def feature_importance(clf):
    importances = clf.feature_importances_

    # Print the name and importance of each feature
    for feature_name, importance in sorted(zip(importances, X.columns), reverse=True):
        print(f"{feature_name}: {importance}")

    # Sort the feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]


    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)

    # Show plot
    plt.show()

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
#%%
print("Decision Tree without class weighting")
decision_tree(X_train, X_test, y_train, y_test)

print("\nRandom Forest without class weighting")
random_forest(X_train, X_test, y_train, y_test)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print("Decision Tree with class weighting")
decision_tree(X_train, X_test, y_train, y_test, class_weights_dict)

print("\nRandom Forest with class weighting")
random_forest(X_train, X_test, y_train, y_test, class_weights_dict)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print("Decision Tree with Oversampling")
decision_tree(X_resampled, X_test, y_resampled, y_test)

print("\nRandom Forest with Oversampling")
random_forest(X_resampled, X_test, y_resampled, y_test)

# %%
# Undersample the majority class
rus = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

print("Decision Tree with Undersampling")
decision_tree(X_train_resampled, X_test, y_train_resampled, y_test)

print("\nRandom Forest with Undersampling")
clf = random_forest(X_train_resampled, X_test, y_train_resampled, y_test, return_clf=True)

#save undersampled model
joblib.dump(clf, '../model/trained_undersampling_rf.joblib')

print ("Feature Importances:")
feature_importance(clf)
# %%
ratio = 2
minority_class = X_train[y_train==1]
majority_downsampled = resample(X_train[y_train==0],
                                replace=False, # sample without replacement
                                n_samples= int(len(minority_class) * ratio), # number of samples to downsample to
                                random_state=42) # reproducible results

# Combine minority class and downsampled majority class
X_train_downsampled = pd.concat([majority_downsampled, minority_class])
# Shuffle the data
X_train_downsampled = X_train_downsampled.sample(frac=1, random_state=42)

y_train_downsampled = y_train.loc[X_train_downsampled.index]

# Apply SMOTE to the downsampled training data
smote = SMOTENC(random_state=42, categorical_features = cat_cols_idx)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_downsampled, y_train_downsampled)

#export to csv
X_train_resampled.to_csv("../data/X_train_resampled.csv", index=False)
y_train_resampled.to_csv("../data/y_train_resampled.csv", index=False)

# # #Load from csv
# X_train_resampled = pd.read_csv("../data/X_train_resampled.csv")
# y_train_resampled = pd.read_csv("../data/y_train_resampled.csv")

print("Decision Tree with SMOTENC")
decision_tree(X_train_resampled, X_test, y_train_resampled, y_test, return_clf=True)

print("\nRandom Forest with SMOTENC")
clf = random_forest(X_train_resampled, X_test, y_train_resampled, y_test, return_clf=True)

#save SMOTE model
joblib.dump(clf, '../model/trained_smoteNC_rf.joblib')

print ("Feature Importances:")
feature_importance(clf)