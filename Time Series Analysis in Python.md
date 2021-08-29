# 1. Correlation and Autocorrelation
## A "Thin" Application of Time Series
* Convert the date index to datetime using pandas's to_datetime().
```py
# Import pandas and plotting modules
import pandas as pd
import matplotlib.pyplot as plt

# Convert the date index to datetime
diet.index = pd.to_datetime(diet.index)
```
* Plot the time series and set the argument grid to True to better see the year-ends.
```py
# From previous step
diet.index = pd.to_datetime(diet.index)

# Plot the entire time series diet and show gridlines
diet.plot(grid=True)
plt.show()
```
* Slice the diet dataset to keep only values from 2012, assigning to diet2012.
* Plot the diet2012, again creating gridlines with the grid argument.
```py
# From previous step
diet.index = pd.to_datetime(diet.index)

# Slice the dataset to keep only 2012
diet2012 = diet['2012']

# Plot 2012 data
diet2012.plot(grid=True)
plt.show()
```
## Merging Time Series With Different Dates
* Convert the dates in the stocks.index and bonds.index into sets.
* Take the difference of the stock set minus the bond set to get those dates where the stock market has data but the bond market does not.
* Merge the two DataFrames into a new DataFrame, stocks_and_bonds using the .join() method, which has the syntax df1.join(df2).
* To get the intersection of dates, use the argument how='inner'.
```py
# Import pandas
import pandas as pd

# Convert the stock index and bond index into sets
set_stock_dates = set(stocks.index)
set_bond_dates = set(bonds.index)

# Take the difference between the sets and print
print(set_stock_dates - set_bond_dates)

# Merge stocks and bonds DataFrames using join()
stocks_and_bonds = stocks.join(bonds,how='inner')
```
## Correlation of Stocks and Bonds
* Compute percent changes on the stocks_and_bonds DataFrame using the .pct_change() method and call the new DataFrame returns.
* Compute the correlation of the columns SP500 and US10Y in the returns DataFrame using the .corr() method for Series which has the syntax series1.corr(series2).
* Show a scatter plot of the percentage change in stock and bond yields.
```py
# Compute percent change using pct_change()
returns = stocks_and_bonds.pct_change()
# Compute correlation using corr()
correlation = returns['SP500'].corr(returns['US10Y'])
print("Correlation of stocks and interest rates: ", correlation)

# Make scatter plot
plt.scatter(returns['SP500'],returns['US10Y'])
plt.show()
```
## Flying Saucers Aren't Correlated to Flying Markets
* Calculate the correlation of the columns DJI and UFO.
* Create a new DataFrame of changes using the .pct_change() method.
* Re-calculate the correlation of the columns DJI and UFO on the changes.
```py
# Compute correlation of levels
correlation1 = levels['DJI'].corr(levels['UFO'])
print("Correlation of levels: ", correlation1)

# Compute correlation of percent changes
changes = levels.pct_change()
correlation2 = changes['DJI'].corr(changes['UFO'])
print("Correlation of changes: ", correlation2)
```
## Looking at a Regression's R-Squared
* Compute the correlation between x and y using the .corr() method.
Run a regression:
* First convert the Series x to a DataFrame dfx.
* Add a constant using sm.add_constant(), assigning it to dfx1
* Regress y on dfx1 using sm.OLS().fit().
* Print out the results of the regression and compare the R-squared with the correlation.
```py
# Import the statsmodels module
import statsmodels.api as sm

# Compute correlation of x and y
correlation = x.corr(y)
print("The correlation between x and y is %4.2f" %(correlation))

# Convert the Series x to a DataFrame and name the column x
dfx = pd.DataFrame(x, columns=['x'])

# Add a constant to the DataFrame dfx
dfx1 = sm.add_constant(dfx)

# Regress y on dfx1
result = sm.OLS(y, dfx1).fit()

# Print out the results and look at the relationship between R-squared and the correlation above
print(result.summary())
```
## A Popular Strategy Using Autocorrelation
* Use the .resample() method with rule='W' and how='last'to convert daily data to weekly data.
* The argument how in .resample() has been deprecated.
* The new syntax .resample().last() also works.
* Create a new DataFrame, returns, of percent changes in weekly prices using the .pct_change() method.
* Compute the autocorrelation using the .autocorr() method on the series of closing stock prices, which is the column 'Adj Close' in the DataFrame returns.
```py
# Convert the daily data to weekly data
MSFT = MSFT.resample('W').last()

# Compute the percentage change of prices
returns = MSFT.pct_change()

# Compute and print the autocorrelation of returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))
```
## Are Interest Rates Autocorrelated?
* Create a new DataFrame, daily_diff, of changes in daily rates using the .diff() method.
* Compute the autocorrelation of the column 'US10Y' in daily_diff using the .autocorr() method.
* Use the .resample() method with arguments rule='A' to convert to annual frequency and how='last'.
* The argument how in .resample() has been deprecated.
* The new syntax .resample().last() also works.
* Create a new DataFrame, yearly_diff of changes in annual rates and compute the autocorrelation, as above.
```py
# Compute the daily change in interest rates 
daily_diff = daily_rates.diff()

# Compute and print the autocorrelation of daily changes
autocorrelation_daily = daily_diff['US10Y'].autocorr()
print("The autocorrelation of daily interest rate changes is %4.2f" %(autocorrelation_daily))

# Convert the daily data to annual data
yearly_rates = daily_rates.resample('A').last()

# Repeat above for annual data
yearly_diff = yearly_rates.diff()
autocorrelation_yearly = yearly_diff['US10Y'].autocorr()
print("The autocorrelation of annual interest rate changes is %4.2f" %(autocorrelation_yearly))
```
## Taxing Exercise: Compute the ACF
* Import the acf module and plot_acf module from statsmodels.
* Compute the array of autocorrelations of the quarterly earnings data in DataFrame HRB.
* Plot the autocorrelation function of the quarterly earnings data in HRB, and pass the  argument alpha=1 to suppress the confidence interval.
```py
# Import the acf module and the plot_acf module from statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

# Compute the acf array of HRB
acf_array = acf(HRB)
print(acf_array)

# Plot the acf function
plot_acf(HRB,alpha=1)
plt.show()
```
## Are We Confident This Stock is Mean Reverting?
* Recompute the autocorrelation of weekly returns in the Series 'Adj Close' in the returns DataFrame.
* Find the number of observations in the returns DataFrame using the len() function.
* Approximate the 95% confidence interval of the estimated autocorrelation. The math function sqrt() has been imported and can be used.
* Plot the autocorrelation function of returns using plot_acf that was imported from statsmodels. Set alpha=0.05 for the confidence intervals (that's the default) and lags=20.
```py
# Import the plot_acf module from statsmodels and sqrt from math
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt

# Compute and print the autocorrelation of MSFT weekly returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly MSFT returns is %4.2f" %(autocorrelation))

# Find the number of observations by taking the length of the returns DataFrame
nobs = len(returns)

# Compute the approximate confidence interval
conf = 1.96/ sqrt(nobs)
print("The approximate confidence interval is +/- %4.2f" %(conf))

# Plot the autocorrelation function with 95% confidence intervals and 20 lags using plot_acf
plot_acf(returns, alpha=0.05, lags=20)
plt.show()
```
## Can't Forecast White Noise
* Generate 1000 random normal returns using np.random.normal() with mean 2% (0.02) and standard deviation 5% (0.05), where the argument for the mean is loc and the argument for the standard deviation is scale.
* Verify the mean and standard deviation of returns using np.mean() and np.std().
* Plot the time series.
* Plot the autocorrelation function using plot_acf with lags=20.
```py
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Simulate white noise returns
returns = np.random.normal(loc=.02, scale=.05, size=1000)

# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f" %(mean,std))

# Plot returns series
plt.plot(returns)
plt.show()

# Plot autocorrelation function of white noise returns
plot_acf(returns, lags=20)
plt.show()
```
## Generate a Random Walk
* Generate 500 random normal "steps" with mean=0 and standard deviation=1 using np.random.normal(), where the argument for the mean is loc and the argument for the standard deviation is scale.
* Simulate stock prices P:
* Cumulate the random steps using the numpy .cumsum() method
* Add 100 to P to get a starting stock price of 100.
* Plot the simulated random walk
```py
# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)

# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0

# Simulate stock prices, P with a starting price of 100
P = 100 + np.cumsum(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk")
plt.show()
```
## Get the Drift
* Generate 500 random normal multiplicative "steps" with mean 0.1% and standard deviation 1% using np.random.normal(), which are now returns, and add one for total return.
* Simulate stock prices P:
Cumulate the product of the steps using the numpy .cumprod() method.
Multiply the cumulative product of total returns by 100 to get a starting value of 100.
* Plot the simulated random walk with drift.
```py
# Generate 500 random steps
steps = np.random.normal(loc=.001, scale=.01, size=500) + 1

# Set first element to 1
steps[0]=1

# Simulate the stock price, P, by taking the cumulative product
P = 100 * np.cumprod(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk with Drift")
plt.show()
```
## Are Stock Prices a Random Walk?
* Import the adfuller module from statsmodels.
* Run the Augmented Dickey-Fuller test on the series of closing stock prices, which is the column 'Adj Close' in the AMZN DataFrame.
* Print out the entire output, which includes the test statistic, the p-values, and the critical values for tests with 1%, 10%, and 5% levels.
* Print out just the p-value of the test (results[0] is the test statistic, and results[1] is the p-value).
```py
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Run the ADF test on the price series and print out the results
results = adfuller(AMZN['Adj Close'])
print(results)

# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))
```
## How About Stock Returns?
* Import the adfuller module from statsmodels.
* Create a new DataFrame of AMZN returns by taking the percent change of prices using the method .pct_change().
* Eliminate the NaN in the first row of returns using the .dropna() method on the DataFrame.
* Run the Augmented Dickey-Fuller test on the 'Adj Close' column of AMZN_ret, and print out the p-value in results[1].
```py
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Create a DataFrame of AMZN returns
AMZN_ret = AMZN.pct_change().dropna()

# Run the ADF test on the return series and print out the p-value
results = adfuller(AMZN_ret['Adj Close'])
print('The p-value of the test on returns is: ' + str(results[1]))
```
## Seasonal Adjustment During Tax Season
* Create a new DataFrame of seasonally adjusted earnings by taking the lag-4 difference of quarterly earnings using the .diff() method.
* Examine the first 10 rows of the seasonally adjusted DataFrame and notice that the first four rows are NaN.
* Drop the NaN rows using the .dropna() method.
* Plot the autocorrelation function of the seasonally adjusted DataFrame.
```py
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Seasonally adjust quarterly earnings
HRBsa = HRB.diff(4)

# Print the first 10 rows of the seasonally adjusted series
print(HRBsa.head(10))

# Drop the NaN data in the first four rows
HRBsa = HRBsa.dropna()

# Plot the autocorrelation function of the seasonally adjusted series
plot_acf(HRBsa)
plt.show()
```
## Autoregressive (AR) Models
* Import the class ArmaProcess in the arima_process module.
* Plot the simulated AR processes:
Let ar1 represent an array of the AR parameters [1, ] as explained above. For now, the MA parameter array, ma1, will contain just the lag-zero coefficient of one.
With parameters ar1 and ma1, create an instance of the class ArmaProcess(ar,ma) called AR_object1.
Simulate 1000 data points from the object you just created, AR_object1, using the method .generate_sample(). Plot the simulated data in a subplot.
* Repeat for the other AR parameter.
```py
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: AR parameter = +0.9
plt.subplot(2,1,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: AR parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.show()
```
## Compare the ACF for Several AR Time Series
* Compute the autocorrelation function for each of the three simulated datasets using the plot_acf function with 20 lags (and suppress the confidence intervals by setting alpha=1).
```py
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: AR parameter = +0.9
plot_acf(simulated_data_1, alpha=1, lags=20)
plt.show()

# Plot 2: AR parameter = -0.9
plot_acf(simulated_data_2, alpha=1, lags=20)
plt.show()

# Plot 3: AR parameter = +0.3
plot_acf(simulated_data_3, alpha=1, lags=20)
plt.show()
```
## Estimating an AR Model
* Import the class ARMA in the module statsmodels.tsa.arima_model.
* Create an instance of the ARMA class called mod using the simulated data simulated_data_1 and the order (p,q) of the model (in this case, for an AR(1)), is order=(1,0).
* Fit the model mod using the method .fit() and save it in a results object called res.
* Print out the entire summary of results using the .summary() method.
* Just print out an estimate of the constant and  using the .params attribute (no parentheses).
```py
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for phi
print("When the true phi=0.9, the estimate of phi (and the constant) are:")
print(res.params)
```
## Forecasting with an AR Model
* Import the class ARMA in the module statsmodels.tsa.arima_model
* Create an instance of the ARMA class called mod using the simulated data simulated_data_1 and the order (p,q) of the model (in this case, for an AR(1) order=(1,0)
* Fit the model mod using the method .fit() and save it in a results object called res
* Plot the in-sample and out-of-sample forecasts of the data using the plot_predict() method
* Start the forecast 10 data points before the end of the 1000 point series at 990, and end the forecast 10 data points after the end of the series at point 1010
```py
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first AR(1) model
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()
res.plot_predict(start=990, end=1010)
plt.show()
```
## Let's Forecast Interest Rates
* Import the class ARMA in the module statsmodels.tsa.arima_model.
* Create an instance of the ARMA class called mod using the annual interest rate data and choosing the order for an AR(1) model.
* Fit the model mod using the method .fit() and save it in a results object called res.
* Plot the in-sample and out-of-sample forecasts of the data using the .plot_predict() method.
Pass the arguments start=0 to start the in-sample forecast from the beginning, and choose end to be '2022' to forecast several years in the future.
* Note that the end argument 2022 must be in quotes here since it represents a date and not an integer position.
```py
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast interest rates using an AR(1) model
mod = ARMA(interest_rate_data, order=(1,0))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start=0,end='2022')
plt.legend(fontsize=8)
plt.show()
```
## Compare AR Model with Random Walk
* Import plot_acf function from the statsmodels module
* Create two axes for the two subplots
* Plot the autocorrelation function for 12 lags of the interest rate series interest_rate_data in the top plot
* Plot the autocorrelation function for 12 lags of the interest rate series simulated_data in the bottom plot
```py
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot the interest rate series and the simulated random walk series side-by-side
fig, axes = plt.subplots(2,1)

# Plot the autocorrelation of the interest rate series in the top plot
fig = plot_acf(interest_rate_data, alpha=1, lags=12, ax=axes[0])

# Plot the autocorrelation of the simulated random walk series in the bottom plot
fig = plot_acf(simulated_data, alpha=1, lags=12, ax=axes[1])

# Label axes
axes[0].set_title("Interest Rate Data")
axes[1].set_title("Simulated Random Walk Data")
plt.show()
```
## Estimate Order of Model: PACF
* Import the modules for simulating data and for plotting the PACF
* Simulate an AR(1) with  (remember that the sign for the AR parameter is reversed)
* Plot the PACF for simulated_data_1 using the plot_pacf function
* Simulate an AR(2) with 
 (again, reverse the signs)
* Plot the PACF for simulated_data_2 using the plot_pacf function
```py
# Import the modules for simulating data and for plotting the PACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf

# Simulate AR(1) with phi=+0.6
ma = np.array([1])
ar = np.array([1, -0.6])
AR_object = ArmaProcess(ar, ma)
simulated_data_1 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(1)
plot_pacf(simulated_data_1, lags=20)
plt.show()

# Simulate AR(2) with phi1=+0.6, phi2=+0.3
ma = np.array([1])
ar = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar, ma)
simulated_data_2 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(2)
plot_pacf(simulated_data_2, lags=20)
plt.show()
```
## Estimate Order of Model: Information Criteria
* Import the ARMA module for estimating the parameters and computing BIC.
* Initialize a numpy array BIC, which we will use to store the BIC for each AR(p) model.
* Loop through order p for p = 0,…,6.
* For each p, fit the data to an AR model of order p.
* For each p, save the value of BIC using the .bic attribute (no parentheses) of res.
* Plot BIC as a function of p (for the plot, skip p=0 and plot for p=1,…6).
```py
# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARMA(simulated_data_2, order=(p,0))
    res = mod.fit()
# Save BIC for AR(p)    
    BIC[p] = res.bic
    
# Plot the BIC as a function of p
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.show()
```
# 4. Moving Average (MA) and ARMA Models
* Import the class ArmaProcess in the arima_process module.
* Plot the simulated MA(1) processes
* Let ma1 represent an array of the MA parameters [1, ] as explained above. The AR parameter array will contain just the lag-zero coefficient of one.
* With parameters ar1 and ma1, create an instance of the class ArmaProcess(ar,ma) called MA_object1.
* Simulate 1000 data points from the object you just created, MA_object1, using the method .generate_sample(). Plot the simulated data in a subplot.
* Repeat for the other MA parameter.
```py
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: MA parameter = -0.9
plt.subplot(2,1,1)
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: MA parameter = +0.9
plt.subplot(2,1,2)
ar2 = np.array([1])
ma2 = np.array([1, .9])
MA_object2 = ArmaProcess(ar2, ma2)
simulated_data_2 = MA_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)

plt.show()
```
## Compute the ACF for Several MA Time Series
* simulated_data_1 is the first simulated time series with an MA parameter of .
* Compute the autocorrelation function of simulated_data_1 using the plot_acf function with 20 lags.
```py
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: MA parameter = -0.9
plot_acf(simulated_data_1, lags=20)
plt.show()
```
* simulated_data_2 is the second simulated time series with an MA parameter of .
* Compute the autocorrelation function using the plot_acf function with lags=20.
```py
# Plot 2: MA parameter = 0.9
plot_acf(simulated_data_2,lags=20)
plt.show()
```
* simulated_data_3 is the third simulated time series with an MA parameter of .
* Compute the autocorrelation function using the plot_acf() function with 20 lags.
```py
# Plot 3: MA parameter = -0.3
plot_acf(simulated_data_3, lags=20)
plt.show()
```
## Estimating an MA Model
* Import the class ARMA in the module statsmodels.tsa.arima_model.
* Create an instance of the ARMA class called mod using the simulated data simulated_data_1 and the order (p,q) of the model (in this case, for an MA(1)), is order=(0,1).
* Fit the model mod using the method .fit() and save it in a results object called res.
* Print out the entire summary of results using the .summary() method.
Just print out an estimate of the constant and theta parameter using the .params attribute (no arguments).
```py
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Fit an MA(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(0,1))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for theta
print("When the true theta=-0.9, the estimate of theta (and the constant) are:")
print(res.params)
```
## Forecasting with MA Model
* Import the class ARMA in the module statsmodels.tsa.arima_model
* Create an instance of the ARMA class called mod using the simulated data simulated_data_1 and the (p,q) order of the model (in this case, for an MA(1), order=(0,1)
* Fit the model mod using the method .fit() and save it in a results object called res
* Plot the in-sample and out-of-sample forecasts of the data using the .plot_predict() method
* Start the forecast 10 data points before the end of the 1000 point series at 990, and end the forecast 10 data points after the end of the series at point 1010
```py
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first MA(1) model
mod = ARMA(simulated_data_1, order=(0,1))
res = mod.fit()
res.plot_predict(start=990, end=1010)
plt.show()
```
## High Frequency Stock Prices
* Manually change the first date to zero using .iloc[0,0].
* Change the two column headers to 'DATE' and 'CLOSE' by setting intraday.columns equal to a list containing those two strings.
* Use the pandas attribute .dtypes (no parentheses) to see what type of data are in each column.
* Convert the 'DATE' column to numeric using the pandas function to_numeric().
* Make the 'DATE' column the new index of intraday by using the pandas method .set_index(), which will take the string 'DATE' as its argument (not the entire column, just the name of the column).
```py
# import datetime module
import datetime

# Change the first date to zero
intraday.iloc[0,0] = 0

# Change the column headers to 'DATE' and 'CLOSE'
intraday.columns = ['DATE','CLOSE']

# Examine the data types for each column
print(intraday.dtypes)

# Convert DATE column to numeric
intraday['DATE'] = pd.to_numeric(intraday['DATE'])

# Make the `DATE` column the new index
intraday = intraday.set_index('DATE')
```
## More Data Cleaning: Missing Data
* Print out the length of intraday using len().
```py
# Notice that some rows are missing
print("If there were no missing rows, there would be 391 rows of minute data")
print("The actual length of the DataFrame is:", len(intraday))
```
* Find the missing rows by making range(391) into a set and then subtracting the set of the intraday index, intraday.index.
```py
# Everything
set_everything = set(range(391))

# The intraday index as a set
set_intraday = set(intraday.index)

# Calculate the difference
set_missing = set_everything - set_intraday

# Print the difference
print("Missing rows: ", set_missing)
```
* Fill in the missing rows using the .reindex() method, setting the index equal to the full range(391) and forward filling the missing data by setting the method argument to 'ffill'.
* Change the index to times using pandas function date_range(), starting with '2017-09-01 9:30' and ending with '2017-09-01 16:00' and passing the argument freq='1min'.
Plot the data and include gridlines.
```py
# From previous step
intraday = intraday.reindex(range(391), method='ffill')

# Change the index to the intraday times
intraday.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00', freq='1min')

# Plot the intraday time series
intraday.plot(grid=True)
plt.show()
```
## Applying an MA Model
* Import plot_acf and ARMA modules from statsmodels
* Compute minute-to-minute returns from prices:
* Compute returns with the .pct_change() method
* Use the pandas method .dropna() to drop the first row of returns, which is NaN
* Plot the ACF function with lags up to 60 minutes
* Fit the returns data to an MA(1) model and print out the MA(1) parameter
```py
# Import plot_acf and ARMA modules from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARMA

# Compute returns from prices and drop the NaN
returns = intraday.pct_change()
returns = returns.dropna()

# Plot ACF of returns with lags up to 60 minutes
plot_acf(returns, lags=60)
plt.show()

# Fit the data to an MA(1) model
mod = ARMA(returns, order=(0,1))
res = mod.fit()
print(res.params)
```
## Equivalence of AR(1) and MA(infinity)
* Import the modules for simulating data and plotting the ACF from statsmodels
* Use a list comprehension to build a list with exponentially decaying MA parameters: 
* Simulate 5000 observations of the MA(30) model
* Plot the ACF of the simulated series
```py
# import the modules for simulating data and plotting the ACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf

# Build a list MA parameters
ma = [0.8**i for i in range(30)]

# Simulate the MA(30) model
ar = np.array([1])
AR_object = ArmaProcess(ar,ma)
simulated_data = AR_object.generate_sample(nsample=5000)

# Plot the ACF
plot_acf(simulated_data, lags=30)
plt.show()
```
# 5. Putting It All Together
## A Dog on a Leash? (Part 1)
* Plot Heating Oil, HO, and Natural Gas, NG, on the same subplot
Make sure you multiply the HO price by 7.25 to match the units of NG
Plot the spread on a second subplot
The spread will be 7.25*HO - NG
```py
# Plot the prices separately
plt.subplot(2,1,1)
plt.plot(7.25*HO, label='Heating Oil')
plt.plot(NG, label='Natural Gas')
plt.legend(loc='best', fontsize='small')

# Plot the spread
plt.subplot(2,1,2)
plt.plot(7.25*HO-NG, label='Spread')
plt.legend(loc='best', fontsize='small')
plt.axhline(y=0, linestyle='--', color='k')
plt.show()
```
## A Dog on a Leash? (Part 2)
* Perform the adfuller test on HO and on NG separately, and save the results (results are a list)
* The argument for adfuller must be a series, so you need to include the column 'Close'
* Print just the p-value (item [1] in the list)
* Do the same thing for the spread, again converting the units of HO, and using the column 'Close' of each DataFrame
```py
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Compute the ADF for HO and NG
result_HO = adfuller(HO['Close'])
print("The p-value for the ADF test on HO is ", result_HO[1])
result_NG = adfuller(NG['Close'])
print("The p-value for the ADF test on NG is ", result_NG[1])

# Compute the ADF of the spread
result_spread = adfuller(7.25 * HO['Close'] - NG['Close'])
print("The p-value for the ADF test on the spread is ", result_spread[1])
```
## Are Bitcoin and Ethereum Cointegrated?
* Import the statsmodels module for regression and the adfuller function
* Add a constant to the ETH DataFrame using sm.add_constant()
* Regress BTC on ETH using sm.OLS(y,x).fit(), where y is the dependent variable and x is the independent variable, and save the results in result.
* The intercept is in result.params[0] and the slope in result.params[1]
* Run ADF test on BTC  ETH
```py
# Import the statsmodels module for regression and the adfuller function
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Regress BTC on ETH
ETH = sm.add_constant(ETH)
result = sm.OLS(BTC,ETH).fit()

# Compute ADF
b = result.params[1]
adf_stats = adfuller(BTC['Price'] - b*ETH['Price'])
print("The p-value for the ADF test is ", adf_stats[1])
```
## Is Temperature a Random Walk (with Drift)?
* Convert the index of years into a datetime object using pd.to_datetime(), and since the data is annual, pass the argument format='%Y'.
* Plot the data using .plot()
* Compute the p-value the Augmented Dickey Fuller test using the adfuller function.
* Save the results of the ADF test in result, and print out the p-value in result[1].
```py
# Import the adfuller function from the statsmodels module
from statsmodels.tsa.stattools import adfuller

# Convert the index to a datetime object
temp_NY.index = pd.to_datetime(temp_NY.index, format='%Y')

# Plot average temperatures
temp_NY.plot()
plt.show()

# Compute and print ADF p-value
result = adfuller(temp_NY['TAVG'])
print("The p-value for the ADF test is ", result[1])
```
## Getting "Warmed" Up: Look at Autocorrelations
* Import the modules for plotting the sample ACF and PACF
* Take first differences of the DataFrame temp_NY using the pandas method .diff()
* Create two subplots for plotting the ACF and PACF
* Plot the sample ACF of the differenced series
* Plot the sample PACF of the differenced series
```py
# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Take first difference of the temperature Series
chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
plot_acf(chg_temp, lags=20, ax=axes[0])

# Plot the PACF
plot_pacf(chg_temp, lags=20, ax=axes[1])
plt.show()
```
## Which ARMA Model is Best?
* For each ARMA model, create an instance of the ARMA class, passing the data and the order=(p,q). p is the autoregressive order; q is the moving average order.
* Fit the model using the method .fit().
* Print the AIC value, found in the .aic element of the results.
```py
# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(1) model and print AIC:
mod_ar1 = ARMA(chg_temp, order=(1, 0))
res_ar1 = mod_ar1.fit()
print("The AIC for an AR(1) is: ", res_ar1.aic)

# Fit the data to an AR(2) model and print AIC:
mod_ar2 = ARMA(chg_temp, order=(2, 0))
res_ar2 = mod_ar2.fit()
print("The AIC for an AR(2) is: ", res_ar2.aic)

# Fit the data to an ARMA(1,1) model and print AIC:
mod_arma11 = ARMA(chg_temp,order=(1,1))
res_arma11 = mod_arma11.fit()
print("The AIC for an ARMA(1,1) is: ", res_arma11.aic)
```
## Don't Throw Out That Winter Coat Yet
* Create an instance of the ARIMA class called mod for an integrated ARMA(1,1) model
* The d in order(p,d,q) is one, since we first differenced once
* Fit mod using the .fit() method and call the results res
* Forecast the series using the plot_predict() method on res
* Choose the start date as 1872-01-01 and the end date as 2046-01-01
```py
# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima_model import ARIMA

# Forecast temperatures using an ARIMA(1,1,1) model
mod = ARIMA(temp_NY, order=(1,1,1))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start='1872-01-01', end='2046-01-01')
plt.show()
```
*Finished by 2021/08/29*