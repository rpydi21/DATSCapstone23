# %%
import pandas as pd
import numpy as np
from data_processing import data_cleaned
from sklearn.preprocessing import LabelEncoder
# %%
from sklearn.neighbors import NearestNeighbors
def hot_deck_imputation(df, k_neighbors=1):
    # Iterate through each row with missing values
    for row_index, row in df.iterrows():
        if row.isnull().any():
            # Find similar cases using k-nearest neighbors
            X = df.dropna()  # Use non-missing rows for comparison
            y = row.dropna()
            
            if len(X) > k_neighbors:
                # Fit a nearest neighbors model
                nn = NearestNeighbors(n_neighbors=k_neighbors)
                nn.fit(X)
                
                # Find the nearest neighbors
                _, indices = nn.kneighbors([y])
                
                # Replace missing values with values from the nearest neighbor(s)
                for col_index, col in enumerate(df.columns):
                    if pd.isna(row[col]):
                        nearest_neighbor_index = indices[0][0]
                        df.at[row_index, col] = X.iloc[nearest_neighbor_index][col]
    
    return df
#%%

data = data_cleaned.select_dtypes(include=['object'])

label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
del column, label_encoders, le

#count na in each column
print(data.isnull().sum())
imputed_data = hot_deck_imputation(data, k_neighbors=1)

print(imputed_data)
df = data.copy()
from sklearn.ensemble import RandomForestClassifier

# Separate the data into two sets: one with missing values and one without
df_missing = df[df['chd'].isnull()]
df_not_missing = df[~df['chd'].isnull()]

# Fit a model to predict the missing values
model = RandomForestClassifier()
model.fit(df_not_missing.drop('chd', axis=1), df_not_missing['chd'])

# Predict missing values
missing_values = model.predict(df_missing.drop('chd', axis=1))

# Fill missing values with the predicted values
df.loc[df['chd'].isnull(), 'chd'] = missing_values


#%%
columns_with_missing = df.columns[df.isnull().any()].tolist()

# Create a label encoder for categorical columns
label_encoders = {}
for col in columns_with_missing:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col].astype(str))

# Separate the data into two sets: one with missing values and one without
df_missing = df[df.isnull().any(axis=1)]
df_not_missing = df[~df.isnull().any(axis=1)]

# Fit a model to predict the missing values
model = RandomForestClassifier()
model.fit(df_not_missing.drop(columns_with_missing, axis=1), df_not_missing[columns_with_missing])

# Predict missing values
missing_values = model.predict(df_missing.drop(columns_with_missing, axis=1))

# Fill missing values with the predicted values
df.loc[df.isnull().any(axis=1), columns_with_missing] = missing_values



data = data_cleaned.select_dtypes(include=['object'])

column = "chd"
le = LabelEncoder()
data[column] = le.fit_transform(data[column])
label_encoders[column] = le
data[column] = le.inverse_transform(data[column])

label_mapping = {label: index for index, label in enumerate(le.classes_)}
#list parts of label_mapping
label_mapping[np.nan]

print("Label Mapping:")
print(label_mapping)

data = data["chd"]
#count na
data.isnull().sum()
data = label_encoding(data)

#%%
data = data_cleaned.select_dtypes(include=['object'])
data = data.apply(label_encoding)
test = data["chd"].apply(label_encoding)
test.name

def label_encoding(variable):
    column = variable.name
    
    label_encoders = {}
    le = LabelEncoder()
    variable = le.fit_transform(variable)
    label_encoders[column] = le
    label_mapping = {label: index for index, label in enumerate(le.classes_)}
    variable = pd.Series(variable)
    variable = variable.replace(label_mapping[np.nan], np.nan)
    return variable


