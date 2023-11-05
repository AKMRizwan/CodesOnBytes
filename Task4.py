import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('D:/data science/Train_dataset.csv', delimiter='\t')
test_data = pd.read_csv('D:\\data science\\Test_dataset.csv', delimiter='\t')
num_rows , num_columns = train_data.shape
print(f"Number of rows in train_data: {num_rows}")
print(f"Number of columns in train_data: {num_columns}")
num_rows , num_columns = test_data.shape
print(f"Number of rows in test_data: {num_rows}")
print(f"Number of columns in test_data: {num_columns}")
print("\n")
print("**** Analyzing the Train dataset ****")
print("     ---------------------------     ")
print("\n")
print("Missing values: ")
missing_data = train_data.isna()
missing_count = missing_data.sum()
print(missing_count)
print("Cleared data:")
missing_data_cleaned = train_data.dropna()
missing_data = missing_data_cleaned.isna()
missing_count = missing_data.sum()
print(missing_count)
# Checking for duplicate data
duplicate_values = missing_data_cleaned.duplicated()
num_duplicate = duplicate_values.sum()
print(f"Number of duplicate values: {num_duplicate}")
# Checking outlier with IQR method
q1 = missing_data_cleaned.select_dtypes(include=['int', 'float']).quantile(0.25)
q2 = missing_data_cleaned.select_dtypes(include=['int', 'float']).quantile(0.75)
IQR = q2-q1
outliers = ((missing_data_cleaned.select_dtypes(include=['int' , 'float']) < (q1-1.5*IQR)) |
            (missing_data_cleaned.select_dtypes(include=['int' , 'float']) > (q2+1.5*IQR))).any(axis=1)
print(missing_data_cleaned[outliers])
num_outliers = outliers.sum()
print(f"Number of outliers: {num_outliers}")
# Now saving clean data into a CSV file
output_file = 'D:/data science/train_data_cleaned_task4'
missing_data_cleaned.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}.")
print("\n")
print("**** Analyzing the test dataset ****")
print("     --------------------------     ")
print("\n")
print("Missing values: ")
test_missing_data = test_data.isna()
test_missing_count = test_missing_data.sum()
print(test_missing_count)
print("Duplicated values: ")
test_duplicate_values = test_data.duplicated()
test_num_duplicate = test_duplicate_values.sum()
print(f"Number of duplicate values: {test_num_duplicate}")
# IQR method for test data
q3 = test_data.select_dtypes(include=['int', 'float']).quantile(0.25)
q4 = test_data.select_dtypes(include=['int', 'float']).quantile(0.75)
IQR = q4-q3
outliers = ((test_data.select_dtypes(include=['int' , 'float']) < (q3-1.5*IQR)) |
            (test_data.select_dtypes(include=['int' , 'float']) > (q4+1.5*IQR))).any(axis=1)
print(test_data[outliers])
test_num_outliers = outliers.sum()
print(f"Number of outliers: {test_num_outliers}")
# Now saving clean data into a CSV file
test_output_file = 'D:/data science/test_data_cleaned_task4'
test_data.to_csv(test_output_file , index=False)
print(f"Cleaned data saved to {test_output_file}.")
# Now our Simple Linear Regression Model
# Now loading cleaned data for regression model
train_data_cleaned = pd.read_csv('D:/data science/train_data_cleaned_task4')
test_data_cleaned = pd.read_csv('D:/data science/test_data_cleaned_task4')
# Splitting of data
x_train = train_data_cleaned[['x']]
y_train = train_data_cleaned['y']
x_test = test_data_cleaned[['x']]
y_test = test_data_cleaned['y']
model = LinearRegression()
model.fit(x_train , y_train)
y_prediction = model.predict(x_test)
mse = mean_squared_error(y_test , y_prediction)
r2 = r2_score(y_test , y_prediction)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Now visualization of data
sns.set(style = "darkgrid")
plt.figure(figsize=(10,6))
plt.xticks(rotation = 90)
plt.scatter(x_train , y_train , label = 'Training Data')
plt.scatter(x_test , y_prediction , color = 'red' , label = 'Prediction Data')
plt.plot(x_test , y_prediction , color = 'blue' , linewidth = 2 , label = 'Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig('Regression_model.png')
plt.show()