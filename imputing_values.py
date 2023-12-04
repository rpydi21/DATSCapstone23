# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import math

#%%
def label_encoding(variable):
    if variable.dtype == 'object':
        column = variable.name
        variable[pd.isnull(variable)] = "NaN"

        le = LabelEncoder()
        variable = pd.Series(le.fit_transform(variable))
        label_encoders[column] = le

        label_mapping = {label: index for index, label in enumerate(le.classes_)}

        if (label_mapping.get("NaN", "Missing") != "Missing"):
            #replace label mapping value in variable for "NaN" with np.nan
            variable.replace(label_mapping.get("NaN"), np.nan, inplace=True)
        
    return variable

def impute_column (col, indices, k_neighbors, X, y):
    nearest_neighbor_index = indices[0][0]
    while pd.isnull(data_imputed.iloc[nearest_neighbor_index][col.name]):
        k_neighbors += 1
        nn = NearestNeighbors(n_neighbors=k_neighbors)
        nn.fit(X)
        _, indices = nn.kneighbors([y])
        nearest_neighbor_index = indices[0][-1]
    col.iloc[0] = data_imputed.iloc[nearest_neighbor_index][col.name]
    return col

def impute_row (row, k_neighbors):
    print(row.name)

    # Find similar cases using k-nearest neighbors
    colnames_with_missing = row.index[row.isnull()]
    X = data_imputed.dropna().drop(colnames_with_missing, axis = 1) # Use non-missing rows for comparison
    y = row.dropna()
    X.columns = range(X.shape[1]) # set column names to be valid feature names

    # Fit a nearest neighbors model
    nn = NearestNeighbors(n_neighbors=k_neighbors)
    nn.fit(X)
    
    # Find the nearest neighbors
    _, indices = nn.kneighbors([y])
    
    row = pd.DataFrame(row)

    #only rows of row with missing values
    columns_with_missing = row.loc[colnames_with_missing]
    columns_with_missing = columns_with_missing.apply(impute_column, axis = 1, indices = indices, 
                                                      k_neighbors = k_neighbors, X = X, y = y)

    #find next nearest neighbor
    # nn = NearestNeighbors(n_neighbors=k_neighbors)
    # if columns_with_missing.isnull().sum() > 0:


    #set corresponding rows in row to columns_with_missing
    row.loc[colnames_with_missing] = columns_with_missing
    row = row.squeeze()
    return row
#%%
data_cleaned = pd.read_csv("../data/data_cleaned.csv")

label_encoders = {}
#select columns that are objects
cat_columns = data_cleaned.select_dtypes(include=['object']).columns
data_encoded = data_cleaned.apply(label_encoding)

data_imputed = data_encoded.copy()
rows_with_missing = data_imputed[data_imputed.isnull().any(axis=1)]
rows_with_missing = rows_with_missing.apply(impute_row, axis=1, k_neighbors = 1)

#replace rows in data_imputed with rows_with_missing
data_imputed.loc[rows_with_missing.index] = rows_with_missing

for column in cat_columns:
    data_imputed[column] = label_encoders[column].inverse_transform(data_imputed[column].astype(int))

#export to csv
data_imputed.to_csv("../data/data_imputed.csv", index=False)