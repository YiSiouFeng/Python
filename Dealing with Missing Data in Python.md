

# 1. The Problem With Missing Data

## Steps for treating missing values
```
Arrange the statements in the order of steps to be taken for dealing with missing data.

A. Evaluate & compare the performance of the treated/imputed dataset.
B. Convert all missing values to null values.
C. Appropriately delete or impute missing values.
D. Analyze the amount and type of missingness in the data.

B, D, C, A
```

## Null value operations
```
# Sum two None values and print the output.
try:
  # Print the sum of two None's
  print("Add operation output of 'None': ", None+None)

except TypeError:
  # Print if error
  print("'None' does not support Arithmetic Operations!!")

# Sum two np.nan and print the output.
try:
  # Print the sum of two np.nan's
  print("Add operation output of 'np.nan': ", np.nan + np.nan)

except TypeError:
  # Print if error
  print("'np.nan' does not support Arithmetic Operations!!")
  
# Print the output of logical or of two None.
try:
  # Print the output of logical OR of two None's
  print("OR operation output of 'None': ", None or None)

except TypeError:
  # Print if error
  print("'None' does not support Logical Operations!!")
  
# Print the output of logical or of two np.nan.
try:
  # Print the output of logical OR of two np.nan's
  print("OR operation output of 'np.nan': ", np.nan or np.nan)

except TypeError:
  # Print if error
  print("'np.nan' does not support Logical Operations!!")
```

## Finding Null values
* Compare two None using == and print the output
```
try:
  # Print the comparison of two 'None's
  print("'None' comparison output: ", None == None)

except TypeError:
  # Print if error
  print("'None' does not support this operation!!")
```
* Compare two np.nan using == and print the output.
```python
try:
  # Print the comparison of two 'np.nan's
  print("'np.nan' comparison output: ", np.nan==np.nan)

except TypeError:
  # Print if error  
  print("'np.nan' does not support this operation!!")
```
* Print whether None is not a number.
```python
try:
  # Check if 'None' is 'NaN'
  print("Is 'None' same as nan? ", np.isnan(None))

except TypeError:
  # Print if error
  print("Function 'np.isnan()' does not support this Type!!")
```
* Print whether np.nan is not a number
```python
try:
  # Check if 'np.nan' is 'NaN'
  print("Is 'np.nan' same as nan? ", np.isnan(np.nan))

except TypeError:
  # Print if error
  print("Function 'np.isnan()' does not support this Type!!")
```

## Detecting missing values
* Read the CSV version of the dataset into a pandas DataFrame.
```python
# Read the dataset 'college.csv'
college = pd.read_csv('college.csv')
print(college.head())
```
* Print the DataFrame information.
```python
# Read the dataset 'college.csv'
college = pd.read_csv('college.csv')
print(college.head())

# Print the info of college
print(college.info())
```
* Store the unique values of the csat column to csat_unique.
```python
# Read the dataset 'college.csv'
college = pd.read_csv('college.csv')
print(college.head())

# Print the 'info' of 'college'
print(college.info())

# Store unique values of 'csat' column to 'csat_unique'
csat_unique = college['csat'].unique()
```
* Sort csat_unique and print the output.
```python
# Read the dataset 'college.csv'
college = pd.read_csv('college.csv')
print(college.head())

# Print the info of college
print(college.info())

# Store unique values of 'csat' column to 'csat_unique'
csat_unique = college.csat.unique()

# Print the sorted values of csat_unique
print(np.sort(csat_unique))
```

## Replacing missing values
* Load the dataset 'college.csv' to the DataFrame college while setting the appropriate missing values.
```python
# Read the dataset 'college.csv' with na_values set to '.'
college = pd.read_csv('college.csv',na_values='.')
print(college.head())
```
* Print the DataFrame information.
```python
# Read the dataset 'college.csv' with na_values set to '.'
college = pd.read_csv('college.csv', na_values='.')
print(college.head())

# Print the info of college
print(college.info())
```
## Replacing hidden missing values
* Describe the basic statistics of diabetes.
```python
# Print the description of the data
print(diabetes.describe())
```
* Isolate the values of BMI which are equal to 0 and store them in zero_bmi.
```python
# Print the description of the data
print(diabetes.describe())

# Store all rows of column 'BMI' which are equal to 0 
zero_bmi = diabetes.BMI[diabetes.BMI == 0]
print(zero_bmi)
```
* Set all the values in the column BMI that are equal to 0 to np.nan.
```python
# Print the description of the data
print(diabetes.describe())

# Store all rows of column 'BMI' which are equal to 0 
zero_bmi = diabetes.BMI[diabetes.BMI == 0]
print(zero_bmi)

# Set the 0 values of column 'BMI' to np.nan
diabetes.BMI[diabetes['BMI']==0] = np.nan
```
* Print the rows with NaN values in BMI.
```python
# Print the description of the data
print(diabetes.describe())

# Store all rows of column 'BMI' which are equal to 0 
zero_bmi = diabetes.BMI[diabetes.BMI == 0]
print(zero_bmi)

# Set the 0 values of column 'BMI' to np.nan
diabetes.BMI[diabetes.BMI == 0] = np.nan

# Print the 'NaN' values in the column BMI
print(diabetes.BMI[diabetes['BMI'].isna()])
```

## Analyzing missingness percentage
* Load 'air-quality.csv' into a pandas DataFrame while parsing the 'Date' column and setting it to the index column as well.
```python
# Load the airquality dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')
```
* Find the number of missing values in airquality and store it into airquality_nullity.
```python
# Load the airquality dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')

# Create a nullity DataFrame airquality_nullity
airquality_nullity = airquality.isnull()
print(airquality_nullity.head())
```
* Calculate the number of missing values in airquality.
```python
# Load the airquality dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')

# Create a nullity DataFrame airquality_nullity
airquality_nullity = airquality.isnull()
print(airquality_nullity.head())

# Calculate total of missing values
missing_values_sum = airquality_nullity.sum()
print('Total Missing Values:\n', missing_values_sum)
```
* Calculate the percentage of missing values in airquality.
```python
# Load the airquality dataset
airquality = pd.read_csv('air-quality.csv', parse_dates=['Date'], index_col='Date')

# Create a nullity DataFrame airquality_nullity
airquality_nullity = airquality.isnull()
print(airquality_nullity.head())

# Calculate total of missing values
missing_values_sum = airquality_nullity.sum()
print('Total Missing Values:\n', missing_values_sum)

# Calculate percentage of missing values
missing_values_percent = airquality_nullity.mean() * 100
print('Percentage of Missing Values:\n', missing_values_percent)
```

## Visualize missingness
* Plot a bar chart of the missing values in airquality.
```python
# Import missingno as msno
import missingno as msno

# Plot amount of missingness
msno.bar(airquality)

# Display bar chart of missing values
display("/usr/local/share/datasets/bar_chart.png")
```

* Plot the nullity matrix of airquality.
```python
# Import missingno as msno
import missingno as msno

# Plot nullity matrix of airquality
msno.matrix(airquality)

# Display nullity matrix
display("/usr/local/share/datasets/matrix.png")
```

* Plot the nullity matrix of airquality across a monthly frequency.
```python
# Import missingno as msno
import missingno as msno

# Plot nullity matrix of airquality with frequency 'M'
msno.matrix(airquality,freq='M')

# Display nullity matrix
display("/usr/local/share/datasets/matrix_frequency.png")
```

* Splice airquality from 'May-1976' to 'Jul-1976' and plot its nullity matrix.
```python
# Import missingno as msno
import missingno as msno

# Plot the sliced nullity matrix of airquality with frequency 'M'
msno.matrix(airquality.loc['May-1976':'Jul-1976'],freq='M')

# Display nullity matrix
display("/usr/local/share/datasets/matrix_sliced.png")
```

# 2. Does Missingness Have A Pattern?
## Guess the missingness type
* Import the missingno package as msno and plot the missingness summary of diabetes.
```python
# Import missingno as msno
import missingno as msno

# Visualize the missingness summary
msno.matrix(diabetes)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_diabetes.png")
```
* Question
```python
Identify which of the following statements are FALSE given the diabetes data. Use the missingness summary on your right which you just created. You can zoom into the image to properly visualize the missing data. Thoroughly read through each of the statements before answering.

1. Glucose is not missing at random.
2. BMI is missing completely at random.
3. Serum_Insulin is missing at random.
4. Diastolic_BP is missing at random.

1 and 3.
```
## Deduce MNAR
```python
# Import missingno as msno
import missingno as msno

# Sort diabetes dataframe on 'Serum Insulin'
sorted_values = diabetes.sort_values('Serum_Insulin')

# Visualize the missingness summary of sorted
msno.matrix(sorted_values)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_sorted.png")
``` 
## Finding correlations in your data
* Create a missingness heatmap for the diabetes DataFrame.
```python
# Import missingno
import missingno as msno

# Plot missingness heatmap of diabetes
msno.heatmap(diabetes)

# Show plot
plt.show()
```
* Create a missingness dendogram for the diabetes DataFrame.
```python
# Import missingno
import missingno as msno

# Plot missingness heatmap of diabetes
msno.heatmap(diabetes)
# Plot missingness dendrogram of diabetes
msno.dendrogram(diabetes)

# Show plot
plt.show()
```
## Identify the missingness type
```python
Identify the missingness types of Skin_Fold. You can use the console for your experimentation and analysis. The missingno package has been imported for you as msno.

Skin_Fold is MNAR
```
## Fill dummy values
* Calculate the number of missing values in each column of the dummy DataFrame.
```python
def fill_dummy_values(df):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    col_null = col.isnull()    
    # Calculate number of missing values in column 
    num_nulls = col_null.sum()
  return df_dummy
```
* Calculate the range of the column.
```python
def fill_dummy_values(df):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    col_null = col.isnull()    
    # Calculate number of missing values in column 
    num_nulls = col_null.sum()
    # Calculate column range
    col_range = col.max()-col.min()
  return df_dummy
```
* Calculate random values with the size of num_nulls.
```python
def fill_dummy_values(df):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    col_null = col.isnull()    
    # Calculate number of missing values in column 
    num_nulls = col_null.sum()
    # Calculate column range
    col_range = col.max() - col.min()
    # Calculate random values of size num_nulls
    dummy_values = (rand(num_nulls) - 2)
  return df_dummy
```
* In your function definition, set the default value of scaling_factor to be 0.075
* Next, scale your dummy values by scaling_factor and multiply them by col_range. 
    The minimum col.min() has been already added for you.
```python
def fill_dummy_values(df, scaling_factor=0.075):
  df_dummy = df.copy(deep=True)
  for col_name in df_dummy:
    col = df_dummy[col_name]
    col_null = col.isnull()    
    # Calculate number of missing values in column 
    num_nulls = col_null.sum()
    # Calculate column range
    col_range = col.max() - col.min()
    # Scale the random values to scaling_factor times col_range
    dummy_values = (rand(num_nulls) - 2) * scaling_factor * col_range + col.min()
    col[col_null] = dummy_values
  return df_dummy
```
## Generate scatter plot with missingness
```python
# Fill dummy values in diabetes_dummy
diabetes_dummy = fill_dummy_values(diabetes)

# Sum the nullity of Skin_Fold and BMI
nullity = diabetes['BMI'].isnull()+ diabetes['Skin_Fold'].isnull()

# Create a scatter plot of Skin Fold and BMI 
diabetes_dummy.plot(x='Skin_Fold', y="BMI", kind='scatter', alpha=0.5, 
                    
                    # Set color to nullity of BMI and Skin_Fold
                    c=nullity, 
                    cmap='rainbow')

plt.show()
```

## Delete MCAR
* Visualize the missingness matrix of diabetes.
```python
# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_diabetes.png")
```
* Print the number of missing values in Glucose.
```python
# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Print the number of missing values in Glucose
print(diabetes['Glucose'].isnull().sum())
```
* Perform listwise deletion on the missing values of the Glucose column and visualize the nullity matrix.
```python
# Visualize the missingness of diabetes prior to dropping missing values
msno.matrix(diabetes)

# Print the number of missing values in Glucose
print(diabetes['Glucose'].isnull().sum())

# Drop rows where 'Glucose' has a missing value
diabetes.dropna(subset=['Glucose'], how='any', inplace=True)

# Visualize the missingness of diabetes after dropping missing values
msno.matrix(diabetes)

display("/usr/local/share/datasets/glucose_dropped.png")
```
## Will you delete?
* Visualize the missingness matrix of diabetes.
```python
# Visualize the missingness in the data
msno.matrix(diabetes)

# Display nullity matrix
display("/usr/local/share/datasets/matrix_diabetes.png")
```
* Visualize the missingness heatmap of diabetes.
```python
# Visualize the missingness in the data
msno.matrix(diabetes)

# Visualize the correlation of missingness between variables
msno.heatmap(diabetes)

# Show heatmap
plt.show()
```
* Question
```
From the graphs that you just created, keenly observe the missingness of the variable BMI and select the best option you deem fit is the reason for missingness.


BMI values are Missing Completely at Random(MCAR). Therefore, we should delete it.
```
* Drop the all the cases where BMI is missing.
```python
# Visualize the missingness in the data
msno.matrix(diabetes)

# Visualize the correlation of missingness between variables
msno.heatmap(diabetes)

# Show heatmap
plt.show()

# Drop rows where 'BMI' has a missing value
diabetes.dropna(subset=['BMI'], how='all', inplace=True)
```

# 3. Imputation Techniques
## Mean & median imputation
* Create a SimpleImputer() object while performing mean imputation.
* Impute the copied DataFrame.
```python
# Make a copy of diabetes
diabetes_mean = diabetes.copy(deep=True)

# Create mean imputer object
mean_imputer = SimpleImputer(strategy = 'mean')

# Impute mean values in the DataFrame diabetes_mean
diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)
```
* Create a SimpleImputer() object while performing median imputation.
* Impute the copied DataFrame.
```python
# Make a copy of diabetes
diabetes_median = diabetes.copy(deep=True)

# Create median imputer object
median_imputer = SimpleImputer(strategy='median')

# Impute median values in the DataFrame diabetes_median
diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)
```
## Mode and constant imputation
* Create a SimpleImputer() object while performing mode (or most frequent) imputation.
* Impute the copied DataFrame.
```python
# Make a copy of diabetes
diabetes_mode = diabetes.copy(deep=True)

# Create mode imputer object
mode_imputer = SimpleImputer(strategy='most_frequent')

# Impute using most frequent value in the DataFrame mode_imputer
diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)
```
* Create a SimpleImputer() object while filling missing values to 0.
* Impute the copied DataFrame.
```python
# Make a copy of diabetes
diabetes_constant = diabetes.copy(deep=True)

# Create median imputer object
constant_imputer = SimpleImputer(strategy='constant', fill_value=0)

# Impute missing values to 0 in diabetes_constant
diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)
```
## Visualize imputations
* Create 4 subplots by making a plot with 2 rows and 2 columns.
* Create the dictionary imputations by mapping each key with its matching DataFrame.
* Loop over axes and imputations, and plot each DataFrame in imputations.
* Set the color to the nullity and the title for each subplot to the name of the imputation.
```python
# Set nrows and ncols to 2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
nullity = diabetes.Serum_Insulin.isnull()+diabetes.Glucose.isnull()

# Create a dictionary of imputations
imputations = {'Mean Imputation': diabetes_mean, 'Median Imputation': diabetes_median, 
               'Most Frequent Imputation': diabetes_mode, 'Constant Imputation': diabetes_constant}

# Loop over flattened axes and imputations
for ax, df_key in zip(axes.flatten(), imputations):
    # Select and also set the title for a DataFrame
    imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter', 
                          alpha=0.5, c=nullity, cmap='rainbow', ax=ax, 
                          colorbar=False, title=df_key)
plt.show()
```
## Filling missing time-series data
* Impute missing values using the forward fill method.
```python
# Print prior to imputing missing values
print(airquality[30:40])

# Fill NaNs using forward fill
airquality.fillna(method='ffill', inplace=True)

# Print after imputing missing values
print(airquality[30:40])
```
* Impute missing values using the backward fill method.
```python
# Print prior to imputing missing values
print(airquality[30:40])

# Fill NaNs using backward fill
airquality.fillna(method='bfill', inplace=True)

# Print after imputing missing values
print(airquality[30:40])
```
## Impute with interpolate method
* Interpolate missing values with the linear method.
```python
# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs linearly
airquality.interpolate(method='linear', inplace=True)

# Print after interpolation
print(airquality[30:40])
```
* Interpolate missing values with the quadratic method.
```python
# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs quadratically
airquality.interpolate(method='quadratic', inplace=True)

# Print after interpolation
print(airquality[30:40])
```
* Interpolate missing values with the nearest method.
```python
# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs with nearest value
airquality.interpolate(method='nearest',inplace=True)

# Print after interpolation
print(airquality[30:40])
```
## Visualize forward fill imputation
* Impute airquality DataFrame with the frontward fill method.
```python
# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill',inplace=True)
```
* Impute airquality DataFrame with the frontward fill method.
```python
# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill')

# Plot the imputed DataFrame ffill_imp in red dotted style 
ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

plt.show()
```
* Overlay the airquality DataFrame on top of your plot.
* Set the title to 'Ozone' and set marker to 'o'.
```python
# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill')

# Plot the imputed DataFrame ffill_imp in red dotted style 
ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

# Plot the airquality DataFrame with title
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

plt.show()
```
## Visualize backward fill imputation
* Impute airquality with the backward fill method.
* Create a red colored line plot of bfill_imp with a 'dotted' line style with 'o' for markers.
* Overlay the airquality DataFrame on top of your plot.
* Set the title to 'Ozone' and set marker to 'o'.
```python
# Impute airquality DataFrame with bfill method
bfill_imputed = airquality.fillna(method='bfill')

# Plot the imputed DataFrame bfill_imp in red dotted style 
bfill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

# Plot the airquality DataFrame with title
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

plt.show()
```
## Plot interpolations
* Create 3 subplots for each plot, using 3 rows and 1 column.
* Create the interpolations dictionary by mapping each DataFrame to the corresponding
* interpolation technique.
* Loop over axes and interpolations.
* Select each DataFrame in interpolations and set the title for a DataFrame using df_key.
```python
# Set nrows to 3 and ncols to 1
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 20))

# Create a dictionary of interpolations
interpolations = {'Linear Interpolation': linear, 'Quadratic Interpolation': quadratic, 
                  'Nearest Interpolation': nearest}

# Loop over axes and interpolations
for ax, df_key in zip(axes, interpolations):
  # Select and also set the title for a DataFrame
  interpolations[df_key].Ozone.plot(color='red', marker='o', 
                                 linestyle='dotted', ax=ax)
  airquality.Ozone.plot(title=df_key + ' - Ozone', marker='o', ax=ax)
  
plt.show()
```
# 4. Advanced Imputation Techniques
## KNN imputation
* Import KNN from fancyimpute.
* Copy diabetes to diabetes_knn_imputed.
* Create a KNN() object and assign it to knn_imputer.
* Impute the diabetes_knn_imputed DataFrame.
```python
# Import KNN from fancyimpute
from fancyimpute import KNN

# Copy diabetes to diabetes_knn_imputed
diabetes_knn_imputed = diabetes.copy(deep=True)

# Initialize KNN
knn_imputer = KNN()

# Impute using fit_tranform on diabetes_knn_imputed
diabetes_knn_imputed.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn_imputed)
```
## MICE imputation
* Import IterativeImputer from fancyimpute.
* Copy diabetes to diabetes_mice_imputed.
* Create an IterativeImputer() object and assign it to mice_imputer.
* Impute the diabetes DataFrame.
```python
# Import IterativeImputer from fancyimpute
from fancyimpute import IterativeImputer

# Copy diabetes to diabetes_mice_imputed
diabetes_mice_imputed = diabetes.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer()

# Impute using fit_tranform on diabetes
diabetes_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(diabetes)
```
## Ordinal encoding of a categorical column
* Create the ordinal encoder object and assign it to ambience_ord_enc.
* Select the non-missing values of the 'ambience' column in users.
* Reshape ambience_not_null to shape (-1, 1).
* Replace the non-missing values of ambience with its encoded values.
```python
# Create Ordinal encoder
ambience_ord_enc = OrdinalEncoder()

# Select non-null values of ambience column in users
ambience = users['ambience']
ambience_not_null = ambience[ambience.notnull()]

# Reshape ambience_not_null to shape (-1, 1)
reshaped_vals = ambience_not_null.values.reshape(-1,1)

# Ordinally encode reshaped_vals
encoded_vals = ambience_ord_enc.fit_transform(reshaped_vals)

# Assign back encoded values to non-null values of ambience in users
users.loc[ambience.notnull(), 'ambience'] = np.squeeze(encoded_vals)
```
## Ordinal encoding of a DataFrame
* Define an empty dictionary ordinal_enc_dict.
* Create an Ordinal Encoder object for each column.
* Select non-null values of column in users and encode them.
* Assign back the encoded values to non-null values of each column (col_name) in users.
```python
# Create an empty dictionary ordinal_enc_dict
ordinal_enc_dict = {}

for col_name in users:
    # Create Ordinal encoder for col
    ordinal_enc_dict[col_name] = OrdinalEncoder()
    col = users[col_name]
    
    # Select non-null values of col
    col_not_null = col[col.notnull()]
    reshaped_vals = col_not_null.values.reshape(-1, 1)
    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
    
    # Store the values to non-null values of the column in users
    users.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)
```
## KNN imputation of categorical values
* Initialize the KNN() imputer.
* Impute the users DataFrame and round the results.
* Iterate over columns in users and perform .inverse_tranform() on the ordinally encoded columns.
```python
# Create KNN imputer
KNN_imputer = KNN()

# Impute and round the users DataFrame
users.iloc[:, :] = np.round(KNN_imputer.fit_transform(users))

# Loop over the column names in users
for col_name in users:
    
    # Reshape the data
    reshaped = users[col_name].values.reshape(-1, 1)
    
    # Perform inverse transform of the ordinally encoded columns
    users[col_name] = ordinal_enc_dict[col_name].inverse_transform(reshaped)
```
## Analyze the summary of linear model
* Set all features in the DataFrame diabetes_cc as X by adding a constant, while excluding and setting the 'Class' column as y.
* Print the summary of the linear model lm.
* Print the adjusted R-squared score of linear model lm.
* Print the .params of the linear model.
```python
# Add constant to X and set X & y values to fit linear model
X = sm.add_constant(diabetes_cc.iloc[:, :-1])
y = diabetes_cc['Class']
lm = sm.OLS(y, X).fit()

# Print summary of lm
print('\nSummary: ', lm.summary())

# Print R squared score of lm
print('\nAdjusted R-squared score: ', lm.rsquared_adj)

# Print the params of lm
print('\nCoefficcients:\n', lm.params)
```
## Comparing R-squared and coefficients
* Create the r_squared DataFrame by mapping each model's adjusted R-squared to the imputation name.
```python
# Store the Adj. R-squared scores of the linear models
r_squared = pd.DataFrame({'Complete Case': lm.rsquared_adj, 
                          'Mean Imputation': lm_mean.rsquared_adj, 
                          'KNN Imputation': lm_KNN.rsquared_adj, 
                          'MICE Imputation': lm_MICE.rsquared_adj}, 
                         index=['Adj. R-squared'])

print(r_squared)
```
* Create the coeff DataFrame by mapping each model's coefficients to the imputation name.
```python
# Store the coefficients of the linear models
coeff = pd.DataFrame({'Complete Case': lm.params, 
                      'Mean Imputation': lm_mean.params, 
                      'KNN Imputation': lm_KNN.params, 
                      'MICE Imputation': lm_MICE.params})

print(coeff)
```
* Select the best imputation based on the R-squared score.
```
r_squares = {'Mean Imputation': lm_mean.rsquared_adj, 
             'KNN Imputation': lm_KNN.rsquared_adj, 
             'MICE Imputation': lm_MICE.rsquared_adj}

# Select best R-squared
best_imputation = max(r_squares, key=r_squares.get)

print("The best imputation technique is: ", best_imputation)
```
## Comparing density plots
* Plot a density plot for the 'Skin_Fold' column for each DataFrame.
* Set the labels using the labels list.
* Set the label for the x-axis to 'Skin Fold'.
```python
# Plot graphs of imputed DataFrames and the complete case
diabetes_cc['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)
diabetes_mean_imputed['Skin_Fold'].plot(kind='kde')
diabetes_knn_imputed['Skin_Fold'].plot(kind='kde')
diabetes_mice_imputed['Skin_Fold'].plot(kind='kde')

# Create labels for the four DataFrames
labels = ['Baseline (Complete Case)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']
plt.legend(labels)

# Set the x-label as Skin Fold
plt.xlabel('Skin Fold')

plt.show()
```
## *Finished by 2021/08/17*


