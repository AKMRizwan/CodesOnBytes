###################: TASK 2  :##################
import pandas as pd
# Loading datasheet
data=pd.read_csv('D:\Download\dataset - netflix1.csv')
# Checking number of rows and columns
num_rows, num_columns=data.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_columns}")
#Checking first five rows with its data types and columns
print(data.head())
print(data.columns)
print(data.dtypes)
# Checking missing values
missing_data=data.isna()
missing_count=missing_data.sum()
print(missing_count)
# There are no missing values
# Checking for duplicate values
duplicates=data.duplicated()
num_duplicates=duplicates.sum()
print(f"Number of duplicate rows: {num_duplicates}")
# There are no duplicate values
# Checking outliers with IQR method
q1=data.select_dtypes(include=['int','float']).quantile(0.25)
q3=data.select_dtypes(include=['int','float']).quantile(0.75)
IQR=q3-q1
outliers = ((data.select_dtypes(include = ['int', 'float'])<(q1-1.5*IQR)) | (data.select_dtypes(include=['int', 'float'])>(q3+1.5*IQR))).any(axis=1)
print(data[outliers])
num_outliers=outliers.sum()
print(f"Number of outliers: {num_outliers}")
# Now removing outliers
data_cleaned=data[~outliers]
print("Number of rows after removing outliers:", data_cleaned.shape[0])
print(data_cleaned)
# converting columns names
output_file='D:\data science\cleaneddata'
data_cleaned.to_csv(output_file, index=False)
print(f"Cleaned dataset saved to {output_file}.")
