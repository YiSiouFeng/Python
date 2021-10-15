# 1. Time Series and Machine Learning Primer
## Plotting a time series (I)
* Print the first five rows of data.
* Print the first five rows of data2
```py
# Print the first 5 rows of data
print(data.head())
# Print the first 5 rows of data2
print(data2.head())
```
* Plot the values column of both the data sets on top of one another, one per axis object.
```py
# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()
```
## Plotting a time series (II)
* Plot data and data2 on top of one another, one per axis object.
* The x-axis should represent the time stamps and the y-axis should represent the dataset values.
```py
# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x='time', y='data_values', ax=axs[0])
data2.iloc[:1000].plot(x='time', y='data_values', ax=axs[1])
plt.show()
```
## Fitting a simple model: classification
* Extract the "petal length (cm)" and "petal width (cm)" columns of data and assign it to X.
Fit a model on X and y.
```py
from sklearn.svm import LinearSVC

# Construct data for the model
X = data[['petal length (cm)','petal width (cm)']]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)
```
## Predicting using a classification model
* Predict the flower type using the array X_predict.
* Run the given code to visualize the predictions.
```py
# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()
```
## Fitting a simple model: regression
* Prepare X and y DataFrames using the data in boston.
* X should be the proportion of houses built prior to 1940, y average number of rooms per dwelling.
* Fit a regression model that uses these variables (remember to shape the variables correctly!).
* Don't forget that each variable must be the correct shape for scikit-learn to use it!
```py
from sklearn import linear_model

# Prepare input and output DataFrames
X = boston['AGE'].reshape([-1,1])
y = boston['RM'].reshape([-1,1])

# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)
```
## Predicting using a regression model
* Review new_inputs in the shell.
* Reshape new_inputs appropriately to generate predictions.
* Run the given code to visualize the predictions.
```py
# Generate predictions with the model using those inputs
predictions = model.predict(
new_inputs.reshape([-1,1]))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()
```
## Inspecting the classification data
* Use glob to return a list of the .wav files in data_dir directory.
* Import the first audio file in the list using librosa.
* Generate a time array for the data.
* Plot the waveform for this file, along with the time array.
```py
import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()
```
## Inspecting the regression data
* Import the data with Pandas (stored in the file 'prices.csv').
* Convert the index of data to datetime.
* Loop through each column of data and plot the the column's values over time.
```py
# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data.columns:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()
```
# 2. Time Series as Inputs to a Model
## Many repetitions of sounds
* First, create the time array for these audio files (all audios are the same length).
* Then, stack the values of the two DataFrames together (normal and abnormal, in that order) so that you have a single array of shape (n_audio_files, n_times_points).
* Finally, use the code provided to loop through each list item / axis, and plot the audio over time in the corresponding axis object.
* You'll plot normal heartbeats in the left column, and abnormal ones in the right column
```py
fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)

# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()
```
## Invariance in time
* Average across the audio files contained in normal and abnormal, leaving the time dimension.
* Visualize these averages over time.
```py
# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()
```
## Build a classification model
* Create an instance of the Linear SVC model and fit the model using the training data.
* Use the testing data to generate predictions with the model.
* Score the model using the provided code.
```py
from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train,y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))
```
## Calculating the envelope of sound
* Rectify the audio.
* Smooth the audio file by applying a rolling mean.
* Plot the result.
```
# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()
```
## Calculating features from the envelope
* Calculate the mean, standard deviation, and maximum value for each heartbeat sound.
* Column stack these stats in the same order.
* Use cross-validation to fit a model on each CV iteration.
```py
# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))
```
## Derivative features: The tempogram
* Use librosa to calculate a tempogram of each heartbeat audio.
* Calculate the mean, standard deviation, and maximum of each tempogram (this time using DataFrame methods)
```py
# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempo
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)
```
* Column stack these tempo features (mean, standard deviation, and maximum) in the same order.
* Score the classifier with cross-validation.
```py
# Create the X and y arrays
X = np.column_stack([means, stds, maxs,tempos_mean,tempos_std,tempos_max])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))
```
## Spectrograms of heartbeat audio
* Import the short-time fourier transform (stft) function from librosa.core.
* Calculate the spectral content (using the short-time fourier transform function) of audio.
```py
# Import the stft function
from librosa.core import stft

# Prepare the STFT
HOP_LENGTH = 2**4
spec = stft(audio, hop_length=HOP_LENGTH, n_fft=2**7)
```
* Convert the spectogram (spec) to decibels.
* Visualize the spectogram.
```py
from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert into decibels
spec_db = amplitude_to_db(spec)

# Compare the raw audio to the spectrogram of the audio
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, audio)
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
plt.show()
```
## Engineering spectral features
* Calculate the spectral bandwidth as well as the spectral centroid of the spectrogram by using functions in librosa.feature.
```py
import librosa as lr

# Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]
```
* Convert the spectrogram to decibels for visualization.
* Plot the spectrogram over time.
```py
from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert spectrogram to decibels for visualization
spec_db = amplitude_to_db(spec)

# Display these features on top of the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
ax = specshow( spec_db, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()
```
## Combining many features in a classifier
* Loop through each spectrogram, calculating the mean spectral bandwidth and centroid of each.
```py
# Loop through each spectrogram
bandwidths = []
centroids = []

for spec in spectrograms:
    # Calculate the mean spectral bandwidth
    this_mean_bandwidth = np.mean(lr.feature.spectral_bandwidth(S=spec))
    # Calculate the mean spectral centroid
    this_mean_centroid = np.mean(lr.feature.spectral_centroid(S=spec))
    # Collect the values
    bandwidths.append(this_mean_bandwidth)  
    centroids.append(this_mean_centroid)
```
* Column stack all the features to create the array X.
* Score the classifier with cross-validation.
```py
# Create X and y arrays
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths, centroids])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))
```
# 3. Predicting Time Series Data
## Introducing the dataset
* Plot the data in prices. Pay attention to any irregularities you notice.
* Generate a scatter plot with the values of Ebay on the x-axis, and Yahoo on the y-axis. Look up the symbols for both companies from the column names of the DataFrame.
* Finally, encode time as the color of each datapoint in order to visualize how the relationship between these two variables changes.
```py
# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c=prices.index, 
                    cmap=plt.cm.viridis, colorbar=False)
plt.show()
```
## Fitting a simple regression model
* Create the X and y arrays by using the column names provided.
* The input values should be from the companies "ebay", "nvidia", and "yahoo"
* The output values should be from the company "apple"
* Use the data to train and score the model with cross-validation.
```py
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Use stock symbols to extract training data
X = all_prices[['EBAY','NVDA','YHOO',]]
y = all_prices[['AAPL']]

# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)
```
## Visualizing predicted values
* Split the data (X and y) into training and test sets.
* Use the training data to train the regression model.
* Then use the testing data to generate predictions for the model.
```py
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split our data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=.8, shuffle=False, random_state=1)

# Fit our model and generate predictions
model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(score)
```
* Plot a time series of the predicted and "actual" values of the testing data.
```py
# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()
```
## Visualizing messy data
* Visualize the time series data using Pandas.
* Calculate the number of missing values in each time series. Note any irregularities that you can see. What do you think they are?
```py
# Visualize the dataset
prices.plot(legend=False)
plt.tight_layout()
plt.show()

# Count the missing values of each time series
missing_values = prices.isna().sum()
print(missing_values)
```
## Imputing missing values
* Create a boolean mask for missing values and interpolate the missing values using the interpolation argument of the function.
* Interpolate using the latest non-missing value and plot the results. Recall that interpolate_and_plot's second input is a string specifying the kind of interpolation to use.
* Interpolate linearly and plot the results.
* Interpolate with a quadratic function and plot the results.
```py
# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate(interpolation)

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)
    
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()
    
# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices, interpolation_type)

# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# Interpolate quadratic
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)
```
## Transforming raw data
* Define a percent_change function that takes an input time series and does the following:
* Extract all but the last value of the input series (assigned to previous_values) and the only the last value of the timeseries ( assigned to last_value)
* Calculate the percentage difference between the last value and the mean of earlier values.
* Using a rolling window of 20, apply this function to prices, and visualize it using the given code.
```py
# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()
```
## Handling outliers
* Define a function that takes an input series and does the following:
* Calculates the absolute value of each datapoint's distance from the series mean, then creates a boolean mask for datapoints that are three times the standard deviation from the mean.
* Use this boolean mask to replace the outliers with the median of the entire series.
* Apply this function to your data and visualize the results using the given code.
```py
def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()
```
## Engineering multiple rolling features at once
* Define a list consisting of four features you will calculate: the minimum, maximum, mean, and standard deviation (in that order).
* Using the rolling window (prices_perc_rolling) we defined for you, calculate the features from features_to_calculate.
* Plot the results over time, along with the original time series using the given code.
```py
# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()
```
## Percentiles and partial functions
* Import partial from functools.
* Use the partial() function to create several feature generators that calculate percentiles of your data using a list comprehension.
* Using the rolling window (prices_perc_rolling) we defined for you, calculate the quantiles using percentile_functions.
* Visualize the results using the code given to you.
```py
# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()
```
## Using "date" information
* Calculate the day of the week, week number in a year, and month number in a year.
* Add each one as a column to the prices_perc DataFrame, under the names day_of_week, week_of_year and month_of_year, respectively.
```py
# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.weekday
prices_perc['week_of_year'] = prices_perc.index.weekofyear
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)
```
# 4.Validating and Inspecting Time Series Models
## Creating time-shifted features
* Use a dictionary comprehension to create multiple time-shifted versions of prices_perc using the lags specified in shifts.
* Convert the result into a DataFrame.
Use the given code to visualize the results.
```py
# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()
```
## Special case: Auto-regressive models
* Replace missing values in prices_perc_shifted with the median of the DataFrame and assign it to X.
* Replace missing values in prices_perc with the median of the series and assign it to y.
Fit a regression model using the X and y arrays.
```py
# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)
```
## Visualize regression coefficients
* Define a function (called visualize_coefficients) that takes as input an array of coefficients, an array of each coefficient's name, and an instance of a Matplotlib axis object. It should then generate a bar plot for the input coefficients, with their names on the x-axis.
* Use this function (visualize_coefficients()) with the coefficients contained in the model variable and column names of prices_perc_shifted.
```py
def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax
    
# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_,prices_perc_shifted.columns , ax=axs[1])
plt.show()
```
## Auto-regression with a smoother time series
* Using the function (visualize_coefficients()) you created in the last exercise, generate a plot with coefficients of model and column names of prices_perc_shifted.
```py
# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()
```
## Cross-validation with shuffling
* Initialize a ShuffleSplit cross-validation object with 10 splits.
* Iterate through CV splits using this object. On each iteration:
* Fit a model using the training indices.
* Generate predictions using the test indices, score the model () using the predictions, and collect the results.
```py
# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)
```
## Cross-validation without shuffling
* Instantiate another cross-validation object, this time using KFold cross-validation with 10 splits and no shuffling.
* Iterate through this object to fit a model using the training indices and generate predictions using the test indices.
* Visualize the predictions across CV splits using the helper function (visualize_predictions()) we've provided.
```py
# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=False, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr],y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)
```
## Time-based cross-validation
* Import TimeSeriesSplit from sklearn.model_selection.
* Instantiate a time series cross-validation iterator with 10 splits.
* Iterate through CV splits. On each iteration, visualize the values of the input data that would be used to train the model for that iteration.
```py
# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()
```
## Bootstrapping a confidence interval
* The function should loop over the number of bootstraps (given by the parameter n_boots) and:
* Take a random sample of the data, with replacement, and calculate the mean of this random sample
* Compute the percentiles of bootstrap_means and return it
```py
from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles
```
##Calculating variability in model coefficients 
* Initialize a TimeSeriesSplit cross-validation object
* Create an array of all zeros to collect the coefficients.
* Iterate through splits of the cross-validation object. On each iteration:
* Fit the model on training data
* Collect the model's coefficients for analysis later
```py
# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_
```
* Initialize a TimeSeriesSplit cross-validation object
* Create an array of all zeros to collect the coefficients.
* Iterate through splits of the cross-validation object. On each iteration:
* Fit the model on training data
* Collect the model's coefficients for analysis later
```py
# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(feature_names, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()
```
## Visualizing model score variability over time
* Calculate the cross-validated scores of the model on the data (using a custom scorer we defined for you, my_pearsonr along with cross_val_score).
* Convert the output scores into a pandas Series so that you can treat it as a time series.
* Bootstrap a rolling confidence interval for the mean score using bootstrap_interval().
```py
from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))
```
## Accounting for non-stationarity
* Create an empty DataFrame to collect the results.
* Iterate through multiple window sizes, each time creating a new TimeSeriesSplit object.
* Calculate the cross-validated scores (using a custom scorer we defined for you, my_pearsonr) of the model on training data.
```py
# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model,X,y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores
```
    

