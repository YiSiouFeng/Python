# 1. ARMA Models
## Exploration
* Import matplotlib.pyplot giving it the alias plt and import pandas giving it the alias pd.
* Load in the candy production time series 'candy_production.csv' using pandas, set the index to the'date'column, parse the dates and assign it to the variable candy.
* Plot the time series onto the axis ax using the DataFrame's .plot() method. Then show the plot.
```py
# Import modules
import matplotlib.pyplot as plt
import pandas as pd

# Load in the time series
candy = pd.read_csv('candy_production.csv', 
            index_col='date',
            parse_dates=True)

# Plot and show the time series on axis ax
fig, ax = plt.subplots()
candy.plot(ax=ax)
plt.show()
```
## Train-test splits
* Split the time series into train and test sets by slicing with datetime indexes. Take the train set as everything up to the end of 2006 and the test set as everything from the start of 2007.
* Make a pyplot axes using the subplots() function.
* Use the DataFrame's .plot() method to plot the train and test sets on the axis ax.
```py
# Split the data into a train and test set
candy_train = candy.loc[:'2006']
candy_test = candy.loc['2007':]

# Create an axis
fig, ax = plt.subplots()

# Plot the train and test sets on the axis ax
candy_train.plot(ax=ax)
candy_test.plot(ax=ax)
plt.show()
```
## Augmented Dicky-Fuller
* Import the augmented Dicky-Fuller function adfuller() from statsmodels.
* Run the adfuller() function on the 'earthquakes_per_year' column of the earthquake DataFrame and assign the result to result.
* Print the test statistic, the p-value and the critical values.
```py
# Import augmented dicky-fuller test function
from statsmodels.tsa.stattools import adfuller

# Run test
result = adfuller(earthquake['earthquakes_per_year'])

# Print test statistic
print(result[0])

# Print p-value
print(result[1])

# Print critical values
print(result[4]) 
```
## Taking the difference
* Run the augmented Dicky-Fuller on the 'city_population' column of city.
* Print the test statistic and the p-value.
```py
# Run the ADF test on the time series
result = adfuller(city['city_population'])

# Plot the time series
fig, ax = plt.subplots()
city.plot(ax=ax)
plt.show()

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
* Take the first difference of city dropping the NaN values. Assign this to city_stationary and run the test again.
```py
# Calculate the first difference of the time series
city_stationary = city.diff().dropna()

# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])

# Plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
* Take the second difference of city, by applying the .diff() method twice and drop the NaN values.
```py
# Calculate the second difference of the time series
city_stationary = city.diff().diff().dropna()

# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])

# Plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```
## Other tranforms
* Calculate the first difference of the time series amazon to test for stationarity and drop the NaNs.
* Calculate the log return on the stocks time series amazon to test for stationarity.
```py
# Calculate the first difference and drop the nans
amazon_diff = amazon.diff()
amazon_diff = amazon_diff.dropna()

# Run test and print
result_diff = adfuller(amazon_diff['close'])
print(result_diff)

# Calculate log-return and drop nans
amazon_log = np.log(amazon)
amazon_log = amazon_log.dropna()

# Run test and print
result_log = adfuller(amazon_log['close'])
print(result_log)
```
## Generating ARMA data
* Set ar_coefs and ma_coefs for an MA(1) model with MA lag-1 coefficient of -0.7.
Generate a time series of 100 values.
```py
# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(1)

# Set coefficients
ar_coefs = [1]
ma_coefs = [1,-0.7]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)


plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()
```
* Set the coefficients for an AR(2) model with AR lag-1 and lag-2 coefficients of 0.3 and 0.2 respectively.
```py
# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(2)

# Set coefficients
ar_coefs = [1,-0.3,-0.2]
ma_coefs = [1]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()
```
* Set the coefficients for a model with form 
```py
# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(3)

# Set coefficients
ar_coefs = [1,0.2]
ma_coefs = [1,0.3,0.4]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()
```
## Fitting Prelude
* Import the ARMA model class from the statsmodels.tsa.arima_model submodule.
* Create a model object, passing it the time series y and the model order (1,1). Assign this to the variable model.
* Use the model's .fit() method to fit to the data.
```py
# Import the ARMA model
from statsmodels.tsa.arima_model import ARMA

# Instantiate the model
model = ARMA(y, order=(1,1))

# Fit the model
results = model.fit()
```
# 2. Fitting the Future
## Fitting AR and MA models
* Fit an AR(2) model to the 'timeseries_1' column of sample.
* Print a summary of the fitted model.
```py
# Instantiate the model
model = ARMA(sample['timeseries_1'], order=(2,0))

# Fit the model
results = model.fit()

# Print summary
print(results.summary())
```
* Fit an MA(3) model to the 'timeseries_2' column of sample.
```py
# Instantiate the model
model = ARMA(sample['timeseries_2'], order=(0,3))

# Fit the model
results = model.fit()

# Print summary
print(results.summary())
```
## Fitting an ARMA model
* Instantiate an ARMA(3,1) model and pass it the earthquakes dataset.
* Fit the model.
* Print the summary of the model fit.
```py
# Instantiate the model
model = ARMA(earthquake,order=(3,1))

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())
```
## Fitting an ARMAX model
* Instantiate an ARMAX(2,1) model to train on the 'wait_times_hrs' column of hospital using the 'nurse_count' column as an exogenous variable.
* Fit the model.
* Print the summary of the model fit.
```py
# Instantiate the model
model = ARMA(hospital['wait_times_hrs'],order=(2,1),exog=hospital['nurse_count'])

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())
```
## Generating one-step-ahead predictions
* Use the results object to make one-step-ahead predictions over the latest 30 days of data and assign the result to one_step_forecast.
* Assign your mean predictions to mean_forecast using one of the attributes of the one_step_forecast object.
* Extract the confidence intervals of your predictions from the one_step_forecast object and assign them to confidence_intervals.
* Print your mean predictions.
```py
# Generate predictions
one_step_forecast = results.get_prediction(start=-30)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean

# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate  predictions
print(mean_forecast)
```
## Plotting one-step-ahead predictions
* Plot the amazon data, using the amazon.index as the x coordinates.
* Plot the mean_forecast prediction similarly, using mean_forecast.index as the x-coordinates.
* Plot a shaded area between lower_limits and upper_limits of your confidence interval. Use the index of lower_limits as the x coordinates.
```py
# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits,upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()
```
## Generating dynamic forecasts
* Use the results object to make a dynamic predictions for the latest 30 days and assign the result to dynamic_forecast.
* Assign your predictions to a new variable called mean_forecast using one of the attributes of the dynamic_forecast object.
* Extract the confidence intervals of your predictions from the dynamic_forecast object and assign them to a new variable confidence_intervals.
* Print your mean predictions.
```py
# Generate predictions
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = dynamic_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = dynamic_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate predictions
print(mean_forecast)
```
## Plotting dynamic forecasts
* Plot the amazon data using the dates in the index of this DataFrame as the x coordinates and the values as the y coordinates.
* Plot the mean_forecast predictions similarly.
* Plot a shaded area between lower_limits and upper_limits of your confidence interval. Use the index of one of these DataFrames as the x coordinates.
```py
# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean forecast
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()
```
## Differencing and fitting ARMA
* Use the .diff() method of amazon to make the time series stationary by taking the first difference. Don't forget to drop the NaN values using the .dropna() method.
* Create an ARMA(2,2) model using the SARIMAX class, passing it the stationary data.
* Fit the model.
```py
# Take the first difference of the data
amazon_diff = amazon.diff().dropna()

# Create ARMA(2,2) model
arma = SARIMAX(amazon_diff,order=(2,0,2))

# Fit model
arma_results = arma.fit()

# Print fit summary
print(arma_results.summary())
```
## Unrolling ARMA forecast
* Use the .get_forecast() method of the arma_results object and select the predicted mean of the next 10 differences.
* Use the np.cumsum() function to integrate your difference forecast.
* Add the last value of the original DataFrame to make your forecast an absolute value.
```py
# Make arma forecast of next 10 differences
arma_diff_forecast = arma_results.get_forecast(steps=10).predicted_mean

# Integrate the difference forecast
arma_int_forecast = np.cumsum(arma_diff_forecast)

# Make absolute value forecast
arma_value_forecast = arma_int_forecast + amazon.iloc[-1,0]

# Print forecast
print(arma_value_forecast)
```
## Fitting an ARIMA model
* Create an ARIMA(2,1,2) model, using the SARIMAX class, passing it the Amazon stocks data amazon.
* Fit the model.
* Make a forecast of mean values of the Amazon data for the next 10 time steps. Assign the result to arima_value_forecast.
```py
# Create ARIMA(2,1,2) model
arima = SARIMAX(amazon,order=(2,1,2))

# Fit ARIMA model
arima_results = arima.fit()

# Make ARIMA forecast of next 10 values
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean

# Print forecast
print(arima_value_forecast)
```
# 3. The Best of the Best Models  
## AR or MA
* Import the plot_acf and plot_pacf functions from statsmodels.
* Plot the ACF and the PACF for the series df for the first 10 lags but not the zeroth lag.
```py
# Import
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(df, lags=10, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(df, lags=10, zero=False, ax=ax2)

plt.show()
```
## Order of earthquakes
* Plot the ACF and the PACF of the earthquakes time series earthquake up to a lag of 15 steps and don't plot the zeroth lag.
* Create and train a model object for the earthquakes time series.
```py
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

# Plot ACF and PACF
plot_acf(earthquake, lags=15, zero=False, ax=ax1)
plot_pacf(earthquake,lags=15,zero=False, ax=ax2)

# Show plot
plt.show()

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

# Plot ACF and PACF
plot_acf(earthquake, lags=10, zero=False, ax=ax1)
plot_pacf(earthquake, lags=10, zero=False, ax=ax2)

# Show plot
plt.show()

# Instantiate model
model = SARIMAX(earthquake,order=(1,0,0))

# Train model
results = model.fit()
```
## Searching over model order
* Loop over values of p from 0-2.
* Loop over values of q from 0-2.
* Train and fit an ARMA(p,q) model.
* Append a tuple of (p,q, AIC value, BIC value) to order_aic_bic.
```py
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over q values from 0-2
    for q in range(3):
      	# create and fit ARMA(p,q) model
        model = SARIMAX(df, order=(p,0,q))
        results = model.fit()
        
        # Append order and results tuple
        order_aic_bic.append((p,q,results.aic,results.bic))
```
## Choosing order with AIC and BIC
* Create a DataFrame to hold the order search information in the order_aic_bic list. Give it the column names ['p', 'q', 'AIC', 'BIC'].
* Print the DataFrame in order of increasing AIC and then BIC.
```py
# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p','q','AIC','BIC'])

# Print order_df in order of increasing AIC
print(order_df.sort_values('AIC'))

# Print order_df in order of increasing BIC
print(order_df.sort_values('BIC'))
```
## AIC and BIC vs ACF and PACF
* Loop over orders of p and q between 0 and 2.
* Inside the loop try to fit an ARMA(p,q) to earthquake on each loop.
* Print p and q alongside AIC and BIC in each loop.
* If the model fitting procedure fails print p, q, None, None.
```py
# Loop over p values from 0-2
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):
      
        try:
            # create and fit ARMA(p,q) model
            model = SARIMAX(earthquake,order=(p,0,q))
            results = model.fit()
            
            # Print order and results
            print(p, q, results.aic,results.bic)
            
        except:
            print(p, q, None, None)     
```
## Mean absolute error
* Use np functions to calculate the Mean Absolute Error (MAE) of the .resid attribute of the results object.
* Print the MAE.
* Use the DataFrame's .plot() method with no arguments to plot the earthquake time series..
```py
# Fit model
model = SARIMAX(earthquake, order=(1,0,1))
results = model.fit()

# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))

# Print mean absolute error
print(mae)

# Make plot of time series for comparison
earthquake.plot()
plt.show()
```
## Diagnostic summary statistics
* Fit an ARMA(3,1) model to the time series df.
* Print the model summary.
```py
# Create and fit model
model1 = SARIMAX(df, order=(3,0,1))
results1 = model1.fit()

# Print summary
print(results1.summary())
```
* Fit an AR(2) model to the time series df.
* Print the model summary.
```py
# Create and fit model
model2 = SARIMAX(df, order=(2,0,0))
results2 = model2.fit()

# Print summary
print(results2.summary())
```

## Plot diagnostics
* Fit an ARIMA(1,1,1) model to the time series df.
* Create the 4 diagnostic plots.
```py
# Create and fit model
model = SARIMAX(df, order=(1,1,1))
results=model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()
```

## Identification
* Plot the time series using the DataFrame's .plot() method.
* Apply the Dicky-Fuller test to the 'savings' column of the savings DataFrame and assign the test outcome to result.
* Print the Dicky-Fuller test statistics and the associated p-value.
```py
# Plot time series
savings.plot()
plt.show()

# Run Dicky-Fuller test
result = adfuller(savings.savings)

# Print test statistic
print(result)

# Print p-value
print(result[4])
```

## Identification II
* Make a plot of the ACF, for lags 1-10 and plot it on axis ax1.
* Do the same for the PACF.
```py
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of savings on ax1
plot_acf(savings,ax=ax1,zero=False,lags=10)

# Plot the PACF of savings on ax2
plot_pacf(savings,ax=ax2,zero=False,lags=10)

plt.show()
```
## Estimation
* Loop over values of from 0 to 3 and values of q from 0 to 3.
* Inside the loop create an ARMA(p,q) model with a constant trend.
* Then fit the model to the time series savings.
* At the end of each loop print the values of p and q and the AIC and BIC.
```py
# Loop over p values from 0-3
for p in range(4):
  
  # Loop over q values from 0-3
    for q in range(4):
      try:
        # Create and fit ARMA(p,q) model
        model = SARIMAX(savings, order=(p,0,q), trend='c')
        results = model.fit()
        
        # Print p, q, AIC, BIC
        print(p,q,results.aic,results.bic)
        
      except:
        print(p, q, None, None)
```
## Diagnostics
* Retrain the ARMA(1,2) model on the time series, setting the trend to constant.
* Create the 4 standard diagnostics plots.
* Print the model residual summary statistics.
```py
# Create and fit model
model = SARIMAX(savings,order=(1,0,2),trend='c')
results = model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()

# Print summary
print(results.summary())
```

# 4. Seasonal ARIMA Models
## Seasonal decompose
* Import the seasonal_decompose() function from statsmodels.tsa.seasonal.
* Decompose the 'pounds_per_cow' column of milk_production using an additive model and period of 12 months.
* Plot the decomposition.
```py
# Import seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform additive decomposition
decomp = seasonal_decompose(milk_production['pounds_per_cow'], 
                            period=12)

# Plot decomposition
decomp.plot()
plt.show()
```
## Seasonal ACF and PACF
* Plot the ACF of the 'water_consumers' column of the time series up to 25 lags.
```py
# Create figure and subplot
fig, ax1 = plt.subplots()

# Plot the ACF on ax1
plot_acf(water['water_consumers'], lags=25, zero=False,  ax=ax1)

# Show figure
plt.show()
```
* Subtract a 15 step rolling mean from the original time series and assign this to water_2
Drop the NaN values from water_2
```py
# Subtract the rolling mean
water_2 = water-water.rolling(15).mean()

# Drop the NaN values
water_2 = water_2.dropna()

# Create figure and subplots
fig, ax1 = plt.subplots()

# Plot the ACF
plot_acf(water_2['water_consumers'], lags=25, zero=False, ax=ax1)

# Show figure
plt.show()
```
## Fitting SARIMA models
* Create a SARIMAX(1,0,0)(1,1,0) model and fit it to df1.
* Print the model summary table.
```py
# Create a SARIMAX model
model = SARIMAX(df1, order=(1,0,0), seasonal_order=(1,1,0,7))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())
```
* Create a SARIMAX(2,1,1)(1,0,0) model and fit it to df2.
```py
# Create a SARIMAX model
model = SARIMAX(df2,order=(2,1,1),seasonal_order=(1,0,0,4))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())
```
* Create a SARIMAX(1,1,0)(0,1,1) model and fit it to df3.
```py
# Create a SARIMAX model
model = SARIMAX(df3,order=(1,1,0),seasonal_order=(0,1,1,12))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())
```
## Choosing SARIMA order
* Take the first order difference and the seasonal difference of the aus_employment and drop the NaN values. The seasonal period is 12 months.
```py
# Take the first and seasonal differences and drop NaNs
aus_employment_diff = aus_employment.diff().diff(12).dropna()
```
* Plot the ACF and PACF of aus_employment_diff up to 11 lags.
```py
# Create the figure 
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))

# Plot the ACF on ax1
plot_acf(aus_employment_diff,lags=11,zero=False,ax=ax1)

# Plot the PACF on ax2
plot_pacf(aus_employment_diff,lags=11,zero=False,ax=ax2)

plt.show()
```
* Make a list of the first 5 seasonal lags and assign the result to lags.
* Plot the ACF and PACF of aus_employment_diff for the first 5 seasonal lags.
```py
# Make list of lags
lags = [12, 24, 36, 48, 60]

# Create the figure 
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))

# Plot the ACF on ax1
plot_acf(aus_employment_diff,lags=lags,ax=ax1)

# Plot the PACF on ax2
plot_pacf(aus_employment_diff,lags=lags,ax=ax2)

plt.show()
```
## SARIMA vs ARIMA forecasts
* Create a forecast object, called arima_pred, for the ARIMA model to forecast the next 25 steps after the end of the training data.
* Extract the forecast .predicted_mean attribute from arima_pred and assign it to arima_mean.
* Repeat the above two steps for the SARIMA model.
* Plot the SARIMA and ARIMA forecasts and the held out data wisconsin_test.
```py
# Create ARIMA mean forecast
arima_pred = arima_results.get_forecast(steps=25)
arima_mean = arima_pred.predicted_mean

# Create SARIMA mean forecast
sarima_pred = sarima_results.get_forecast(steps=25)
sarima_mean = sarima_pred.predicted_mean

# Plot mean ARIMA and SARIMA predictions and observed
plt.plot(dates, sarima_mean, label='SARIMA')
plt.plot(dates, arima_mean, label='ARIMA')
plt.plot(wisconsin_test, label='observed')
plt.legend()
plt.show()
```
## Automated model selection
* Import the pmdarima package as pm.
```py
# Import pmdarima as pm
import pmdarima as pm
```
* Model the time series df1 with period 7 days and set first order seasonal differencing and no non-seasonal differencing.
```py
# Create auto_arima model
model1 = pm.auto_arima(df1,
                      seasonal=True, m=7,
                      d=0, D=1, 
                 	  max_p=2, max_q=2,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model1.summary())
```
* Create a model to fit df2. Set the non-seasonal differencing to 1, the trend to a constant and set no seasonality.
```py
# Create model
model2 = pm.auto_arima(df2,
                      d=1,
                      seasonal=False,
                      trend='c',
                 	  max_p=2, max_q=2,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model2.summary())
```
* Fit a SARIMAX(p,1,q)(P,1,Q) model to the data setting start_p, start_q, max_p, max_q, max_P and max_Q to 1.
```py
# Create model for SARIMAX(p,1,q)(P,1,Q)7
model3 = pm.auto_arima(df3,
                      seasonal=True, m=7,
                      d=1, D=1, 
                      start_p=1, start_q=1,
                      max_p=1, max_q=1,
                      max_P=1, max_Q=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model3.summary())
```
## Saving and updating models
* Import the joblib package and use it to save the model to "candy_model.pkl".
```py
# Import joblib
import joblib

# Set model name
filename = "candy_model.pkl"

# Pickle it
joblib.dump(model,filename)
```
* Use the joblib package to load the model back in as loaded_model.
```py
# Import
import joblib

# Set model name
filename = "candy_model.pkl"

# Load the model back in
loaded_model = joblib.load(filename)
```
* Update the loaded model with the data df_new.
```py
# Update the model
loaded_model.update(df_new)
```
## SARIMA model diagnostics
* Fit a SARIMA(1, 1, 1)(0, 1, 1) model to the data and set the trend to constant.
```py
# Import model class
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create model object
model = SARIMAX(co2, 
                order=(1,1,1), 
                seasonal_order=(0,1,1,12), 
                trend='c')
# Fit model
results = model.fit()
```
* Create the common diagnostics plots for the results object.
```py
# Plot common diagnostics
results.plot_diagnostics()
plt.show()
```
## SARIMA forecast
* Create a forecast object for the next 136 steps - the number of months until Jan 2030.
* Assign the .predicted_mean of the forecast to the variable mean.
* Compute the confidence intervals and assign this DataFrame to the variable conf_int.
```py
# Create forecast object
forecast_object = results.get_forecast(steps=136)

# Extract predicted mean attribute
mean = forecast_object.predicted_mean

# Calculate the confidence intervals
conf_int = forecast_object.conf_int()

# Extract the forecast dates
dates = mean.index
```
* Plot the mean predictions against the dates.
* Shade the area between the values in the first two columns of DataFrame conf_int using dates as the x-axis values.
```py
plt.figure()

# Plot past CO2 levels
plt.plot(co2.index, co2, label='past')

# Plot the prediction means as line
plt.plot(dates, mean, label='predicted')

# Shade between the confidence intervals
plt.fill_between(dates, conf_int['lower CO2_ppm'], conf_int['upper CO2_ppm'], alpha=0.2)

# Plot legend and show figure
plt.legend()
plt.show()
```
* Print the final predicted mean of the forecast.
* Print the final row of the confidence interval conf_int.
* Remember to select the correct elements by using .iloc[____] on both.
```py
# Print last predicted mean
print(mean.tail(1)[0])
# print(mean.iloc[-1]) also work

# Print last confidence interval
print(conf_int.iloc[-1,:])
```
*Finished by 2021/09/03*