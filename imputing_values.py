# %%
import pandas as pd
import numpy as np
from data_processing import data_cleaned
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

#%%
def label_encoding(variable):
    if variable.dtype == 'object':
        column = variable.name
        variable[pd.isnull(variable)] = "NaN"

        
        le = LabelEncoder()
        variable = pd.Series(le.fit_transform(variable))
        label_encoders[column] = le
        label_mapping = {label: index for index, label in enumerate(le.classes_)}
        variable.unique()

        if (label_mapping.get("NaN", "Missing") != "Missing"):
            #replace label mapping value in variable for "NaN" with np.nan
            variable.replace(label_mapping.get("NaN"), np.nan, inplace=True)
        
    return variable

label_encoders = {}
data_encoded = data_cleaned.apply(label_encoding)

# %%
df = data_encoded.copy()

df.isnull().sum()
df["chd"].unique()

row = df.iloc[209]
row_index = 209
row.isnull().any()

row = df.iloc[11]
row_index = 11
row.isnull().sum()

col = df["chd"]
#check if 


#find row with at least one na
def impute_column (col, indices, row_index):
    #check if column has na
    if col.isnull().any():
        # col_index = df.columns.get_loc(col.name)
        nearest_neighbor_index = indices[0][0]
        col[row_index] = df.iloc[nearest_neighbor_index][col.name]
    return col



def impute_row (row, k_neighbors):
    print(row.name)
    print(row.isnull().any())
    #check if series has any null
    
    if row.isnull().any():
        row_index = row.name
        # Find similar cases using k-nearest neighbors
        #get all columns without missing value

        colnames_with_missing = row.index[row.isnull()]
        print(colnames_with_missing)
        X = df.dropna().drop(colnames_with_missing, axis = 1) # Use non-missing rows for comparison
        y = row.dropna()
        
        if len(X) > k_neighbors:
            # Fit a nearest neighbors model
            nn = NearestNeighbors(n_neighbors=k_neighbors)
            nn.fit(X)
            
            # Find the nearest neighbors
            _, indices = nn.kneighbors([y])
            print(indices)

            row = row.apply(impute_column, indices = indices, row_index = row_index)

    return row



def hot_deck_imputation(df, k_neighbors):
    # Iterate through each row with missing values
    #get list of all rows with missing data
    # rows_with_missing = df[df.isnull().any(axis=1)]
    k_neighbors = 1
    df = df.apply(impute_row, axis=1, k_neighbors = k_neighbors)
    #iterate through each row
    # for row_index, row in rows_with_missing.iterrows():
    #     if row.isnull().any():
    #         # Find similar cases using k-nearest neighbors
    #         #get all columns without missing value
    #         colnames_with_missing = row.index[row.isnull()]
    #         X = df.dropna().drop(colnames_with_missing, axis = 1) # Use non-missing rows for comparison
    #         y = row.dropna()
            
    #         if len(X) > k_neighbors:
    #             # Fit a nearest neighbors model
    #             nn = NearestNeighbors(n_neighbors=k_neighbors)
    #             nn.fit(X)
                
    #             # Find the nearest neighbors
    #             _, indices = nn.kneighbors([y])

    #             #get entire columns with missing value
    #             columns_with_missing = df[colnames_with_missing]
    #             col = "colonoscopy"
    #             for col in columns_with_missing.columns:
    #                 col_index = df.columns.get_loc(col)
    #                 nearest_neighbor_index = indices[0][0]
    #                 df.at[row_index, col] = df.iloc[nearest_neighbor_index][col]

    
                # col = df["chd"]
                # col_index = df.columns.get_loc("chd")
                # Replace missing values with values from the nearest neighbor(s)
                # for col_index, col in enumerate(df.columns):
                #     if pd.isna(row[col]):
                #         nearest_neighbor_index = indices[0][0]
                #         df.at[row_index, col] = df.iloc[nearest_neighbor_index][col_index]
    
    return df
#%%
#count na in each column
print(data_encoded.isnull().sum())
imputed_data = hot_deck_imputation(data_encoded, k_neighbors=1)

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
def label_encoding(variable):
    column = variable.name
    variable[pd.isnull(variable)] = "NaN"

    label_encoders = {}
    le = LabelEncoder()
    variable = pd.Series(le.fit_transform(variable))
    label_encoders[column] = le
    label_mapping = {label: index for index, label in enumerate(le.classes_)}
    variable.unique()

    if (label_mapping.get("NaN", "Missing") != "Missing"):
        #replace label mapping value in variable for "NaN" with np.nan
        variable.replace(label_mapping.get("NaN"), np.nan, inplace=True)
    
    return variable

data = data_cleaned.select_dtypes(include=['object'])

data_encoded = data.apply(label_encoding)
#%%
