# 1.Univariate Investment Risk and Returns
## Financial timeseries data
* Import pandas as pd.
* Use pd.read_csv() to read in the file from fpath_csv and make sure you parse the 'Date' column correctly using the parse_dates argument.
* Ensure the DataFrame is sorted by the 'Date' column.
* Print the first five rows of StockPrices.
```py
# Import pandas as pd
import pandas as pd

# Read in the csv file and parse dates
StockPrices = pd.read_csv(fpath_csv, parse_dates=['Date'])

# Ensure the prices are sorted by Date
StockPrices = StockPrices.sort_values(by='Date')

# Print only the first five rows of StockPrices
print(StockPrices.head())
```
## Calculating financial returns
* Calculate the simple returns of the stock on the 'Adjusted' column and save it to the 'Returns' column.
* Print the first five rows of StockPrices
* Use the Pandas .plot() method to plot the 'Returns' column over time.
```py
# Calculate the daily returns of the adjusted close price
StockPrices['Returns'] = StockPrices['Adjusted'].pct_change()

# Check the first five rows of StockPrices
print(StockPrices.head())

# Plot the returns column over time
StockPrices['Returns'].plot()
plt.show()
```
## Return distributions
* Convert the 'Returns' column from decimal to percentage returns and assign it to percent_return.
* Drop the missing values (represented as NaN) from percent_return and save the result to returns_plot.
* Plot a histogram of returns_plot with 75 bins using matplotlib's hist() function.
```py
# Convert the decimal returns into percentage returns
percent_return = StockPrices['Returns']*100

# Drop the missing values
returns_plot = percent_return.dropna()

# Plot the returns histogram
plt.hist(returns_plot, bins=75)
plt.show()
```
## First moment: Mu
* Import numpy as np.
* Calculate the mean of the 'Returns' column to estimate the first moment (  ) and set it equal to mean_return_daily.
* Use the formula to derive the average annualized return assuming 252 trading days per year. Remember that exponents in Python are calculated using the ** operator.
```py
# Import numpy as np
import numpy as np

# Calculate the average daily return of the stock
mean_return_daily = np.mean(StockPrices['Returns'])
print(mean_return_daily)

# Calculate the implied annualized average return
mean_return_annualized = ((1+mean_return_daily)**252)-1
print(mean_return_annualized)
```
## Second moment: Variance
* Calculate the daily standard deviation of the 'Returns' column and set it equal to sigma_daily.
* Derive the daily variance (second moment, ) by squaring the standard deviation.
```py
# Calculate the standard deviation of daily return of the stock
sigma_daily = np.std(StockPrices['Returns'])
print(sigma_daily)

# Calculate the daily variance
variance_daily = sigma_daily**2
print(variance_daily)
```
## Annualizing variance
* Annualize sigma_daily by multiplying by the square root of 252 (the number of trading days in a years).
* Once again, square sigma_annualized to derive the annualized variance.
```py
# Annualize the standard deviation
sigma_annualized = sigma_daily*np.sqrt(252)
print(sigma_annualized)

# Calculate the annualized variance
variance_annualized = sigma_annualized**2
print(variance_annualized)
```
## Third moment: Skewness
* Import skew from scipy.stats.
* Drop missing values in the 'Returns' column to prevent errors.
* Calculate the skewness of clean_returns
```py
# Import skew from scipy.stats
from scipy.stats import skew

# Drop the missing values
clean_returns = StockPrices['Returns'].dropna()

# Calculate the third moment (skewness) of the returns distribution
returns_skewness = skew(clean_returns)
print(returns_skewness)
```
## Fourth moment: Kurtosis
* Import kurtosis from scipy.stats.
* Use clean_returns to calculate the excess_kurtosis.
* Derive the fourth_moment from excess_kurtosis.
```py
# Import kurtosis from scipy.stats
from scipy.stats import kurtosis

# Calculate the excess kurtosis of the returns distribution
excess_kurtosis = kurtosis(clean_returns)
print(excess_kurtosis)

# Derive the true fourth moment of the returns distribution
fourth_moment = excess_kurtosis+3
print(fourth_moment)
```
## Statistical tests for normality
* Import shapiro from scipy.stats.
* Run the Shapiro-Wilk test on clean_returns.
* Extract the p-value from the shapiro_results tuple.
```py
# Import shapiro from scipy.stats
from scipy.stats import shapiro

# Run the Shapiro-Wilk test on the stock returns
shapiro_results = shapiro(clean_returns)
print("Shapiro results:", shapiro_results)

# Extract the p-value from the shapiro_results
p_value = shapiro_results[1]
print("P-value: ", p_value)
```
## Calculating portfolio returns
* Finish defining the numpy array of model portfolio_weights with the values according to the table above.
* Use the .mul() method to multiply the portfolio_weights across the rows of StockReturns to get weighted stock returns.
* Then use the .sum() method across the rows on the WeightedReturns object to calculate the portfolio returns.
* Finally, review the plot of cumulative returns over time.
```py
# Finish defining the portfolio weights as a numpy array
portfolio_weights = np.array([0.12, 0.15, 0.08, 0.05, 0.09, 0.10, 0.11, 0.14, 0.16])

# Calculate the weighted stock returns
WeightedReturns = StockReturns.mul(portfolio_weights, axis=1)

# Calculate the portfolio returns
StockReturns['Portfolio'] = WeightedReturns.sum(axis=1)

# Plot the cumulative portfolio returns over time
CumulativeReturns = ((1+StockReturns["Portfolio"]).cumprod()-1)
CumulativeReturns.plot()
plt.show()
```
## Equal weighted portfolios
* Set numstocks equal to 9, which is the number of stocks in your portfolio.
* Use np.repeat() to set portfolio_weights_ew equal to an array with an equal weights for each of the 9 stocks.
* Use the .iloc accessor to select all rows and the first 9 columns when calculating the portfolio return.
* Finally, review the plot of cumulative returns over time.
```py
# How many stocks are in your portfolio?
numstocks = 9

# Create an array of equal weights across all assets
portfolio_weights_ew = np.repeat(1/numstocks,numstocks)

# Calculate the equally-weighted portfolio returns
StockReturns['Portfolio_EW'] = StockReturns.iloc[:,:numstocks].mul(portfolio_weights_ew, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW'])
```
## Market-cap weighted portfolios
* Finish defining the market_capitalizations array of market capitalizations in billions according to the table above.
* Calculate mcap_weights array such that each element is the ratio of market cap of the company to the total market cap of all companies.
* Use the .mul() method on the mcap_weights and returns to calculate the market capitalization weighted portfolio returns.
* Finally, review the plot of cumulative returns over time.
```py
# Create an array of market capitalizations (in billions)
market_capitalizations = np.array([601.51, 469.25, 349.5, 310.48, 299.77, 356.94, 268.88, 331.57, 246.09])

# Calculate the market cap weights
mcap_weights = market_capitalizations/market_capitalizations.sum()

# Calculate the market cap weighted portfolio returns
StockReturns['Portfolio_MCap'] = StockReturns.iloc[:, 0:9].mul(mcap_weights, axis=1).sum(axis=1)
cumulative_returns_plot(['Portfolio', 'Portfolio_EW', 'Portfolio_MCap'])
```
## The correlation matrix
* Calculate the correlation_matrix of the StockReturns DataFrame.
```py
# Calculate the correlation matrix
correlation_matrix = StockReturns.corr()

# Print the correlation matrix
print(correlation_matrix)
```
## Import seaborn as sns.
* Use seaborn's heatmap() function to create a heatmap map correlation_matrix.
```py
# Import seaborn as sns
import seaborn as sns

# Create a heatmap
sns.heatmap(correlation_matrix,
            annot=True,
            cmap="YlGnBu", 
            linewidths=0.3,
            annot_kws={"size": 8})

# Plot aesthetics
plt.xticks(rotation=90)
plt.yticks(rotation=0) 
plt.show()
```
## The co-variance matrix
* Calculate the co-variance matrix of the StockReturns DataFrame.
Annualize the co-variance matrix by multiplying it with 252, the number of trading days in a year.
```py
# Calculate the covariance matrix
cov_mat = StockReturns.cov()

# Annualize the co-variance matrix
cov_mat_annual = cov_mat*252

# Print the annualized co-variance matrix
print(cov_mat_annual)
```
## Portfolio standard deviation
* Calculate the portfolio volatility assuming you use the portfolio_weights by following the formula above.
```py
# Import numpy as np
import numpy as np

# Calculate the portfolio standard deviation
portfolio_volatility = np.sqrt(np.dot(portfolio_weights.T, np.dot(cov_mat_annual, portfolio_weights)))
print(portfolio_volatility)
```
## Sharpe ratios
* Assume a risk_free rate of 0 for this exercise.
* Calculate the Sharpe ratio for each asset by subtracting the risk free rate from returns and then dividing by volatility.
```py
# Risk free rate
risk_free = 0

# Calculate the Sharpe Ratio for each asset
RandomPortfolios['Sharpe'] = RandomPortfolios['Returns']/RandomPortfolios['Volatility']

# Print the range of Sharpe ratios
print(RandomPortfolios['Sharpe'].describe()[['min', 'max']])
```
## The MSR portfolio
* Sort RandomPortfolios with the highest Sharpe value, ranking in descending order.
* Multiply MSR_weights_array across the rows of StockReturns to get weighted stock returns.
* Finally, review the plot of cumulative returns over time.
```py
# Sort the portfolios by Sharpe ratio
sorted_portfolios = RandomPortfolios.sort_values(by=['Sharpe'], ascending=False)

# Extract the corresponding weights
MSR_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the MSR weights as a numpy array
MSR_weights_array = np.array(MSR_weights)

# Calculate the MSR portfolio returns
StockReturns['Portfolio_MSR'] = StockReturns.iloc[:, 0:numstocks].mul(MSR_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR'])
```
## The GMV portfolio
* Sort RandomPortfolios with the lowest volatility value, ranking in ascending order.
* Multiply GMV_weights_array across the rows of StockReturns to get weighted stock returns.
* Finally, review the plot of cumulative returns over time.
```py
# Sort the portfolios by volatility
sorted_portfolios = RandomPortfolios.sort_values(by=['Volatility'], ascending=True)

# Extract the corresponding weights
GMV_weights = sorted_portfolios.iloc[0, 0:numstocks]

# Cast the GMV weights as a numpy array
GMV_weights_array = np.array(GMV_weights)

# Calculate the GMV portfolio returns
StockReturns['Portfolio_GMV'] = StockReturns.iloc[:, 0:numstocks].mul(GMV_weights_array, axis=1).sum(axis=1)

# Plot the cumulative returns
cumulative_returns_plot(['Portfolio_EW', 'Portfolio_MCap', 'Portfolio_MSR', 'Portfolio_GMV'])
```
# 3. Factor Investing
## excess returns
* Calculate excess portfolio returns by subtracting the risk free ('RF') column from the 'Portfolio' column in FamaFrenchData.
* Review the plot of returns and excessive returns.
```py
# Calculate excess portfolio returns
FamaFrenchData['Portfolio_Excess'] = FamaFrenchData['Portfolio']-FamaFrenchData['RF']

# Plot returns vs excess returns
CumulativeReturns = ((1+FamaFrenchData[['Portfolio','Portfolio_Excess']]).cumprod()-1)
CumulativeReturns.plot()
plt.show()
```
## Calculating beta using co-variance
* Generate a co-variance matrix between 'Portfolio_Excess' and 'Market_Excess' columns.
* Calculate the variance of 'Market_Excess' column.
* Calculate the portfolio beta.
```py
# Calculate the co-variance matrix between Portfolio_Excess and Market_Excess
covariance_matrix = FamaFrenchData[['Portfolio_Excess', 'Market_Excess']].cov()

# Extract the co-variance co-efficient
covariance_coefficient = covariance_matrix.iloc[0, 1]
print(covariance_coefficient)

# Calculate the benchmark variance
benchmark_variance = FamaFrenchData['Market_Excess'].var()
print(benchmark_variance)

# Calculating the portfolio market beta
portfolio_beta = covariance_coefficient/benchmark_variance
print(portfolio_beta)
```
## Calculating beta with CAPM
* First, you will need to import statsmodels.formula.api as smf.
* Define a regression model that explains Portfolio_Excess as a function of Market_Excess.
* Extract and print the adjusted r-squared of the fitted regression model.
* Extract the market beta of your portfolio.
```py
# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
CAPM_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess', data=FamaFrenchData)

# Print adjusted r-squared of the fitted regression
CAPM_fit = CAPM_model.fit()
print(CAPM_fit.rsquared_adj)

# Extract the beta
regression_beta = CAPM_fit.params['Market_Excess']
print(regression_beta)
```
## Adjusted R-squared
* Model 3 has the highest adjusted r-squared.

## The Fama French 3-factor model
* Define a regression model that explains Portfolio_Excess as a function of Market_Excess, SMB, and HML.
* Extract the adjusted r-squared value from FamaFrench_fit.
```py
# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
FamaFrench_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML', data=FamaFrenchData)

# Fit the regression
FamaFrench_fit = FamaFrench_model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = FamaFrench_fit.rsquared_adj
print(regression_adj_rsq)
```
## p-values and coefficients
* Extract the p-value for 'SMB'.
* Extract the regression coefficient for 'SMB'.
```py
# Extract the p-value of the SMB factor
smb_pval = FamaFrench_fit.pvalues['SMB']

# If the p-value is significant, print significant
if smb_pval < 0.05:
    significant_msg = 'significant'
else:
    significant_msg = 'not significant'

# Print the SMB coefficient
smb_coeff = FamaFrench_fit.params['SMB']
print("The SMB coefficient is ", smb_coeff, " and is ", significant_msg)
```
## Economic intuition in factor modeling
* Small-cap + Value stocks should have the highest returns and risk.

## The efficient market and alpha
* Extract the coefficient of your intercept and assign it to portfolio_alpha.
* Annualize your portfolio_alpha return by assuming 252 trading days in a year.
```py
# Calculate your portfolio alpha
portfolio_alpha = FamaFrench_fit.params['Intercept']
print(portfolio_alpha)

# Annualize your portfolio alpha
portfolio_alpha_annualized = ((1+portfolio_alpha)**252)-1
print(portfolio_alpha_annualized)
```
## The 5-factor model
* Use what you've learned from the previous exercises to define the FamaFrench5_model regression model for Portfolio_Excess against the original 3 Fama-French factors (Market_Excess, SMB, HML) in addition to the two new factors (RMW, CMA).
* Fit the regression model and store the results in FamaFrench5_fit.
* Extract the adjusted r-squared value and assign it to regression_adj_rsq.
```py
# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
FamaFrench5_model = smf.ols(formula='Portfolio_Excess ~ Market_Excess + SMB + HML + RMW + CMA ', data=FamaFrenchData)

# Fit the regression
FamaFrench5_fit = FamaFrench5_model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = FamaFrench5_fit.rsquared_adj
print(regression_adj_rsq)
```
## Alpha vs R-squared
* The Fama-French 5 Factor model explains the most variability, so alpha is lower

# 4. Value at Risk
## Historical drawdown
* Calculate the running maximum of the cumulative returns of the USO oil ETF (cum_rets) using np.maximum.accumulate().
* Where the running maximum (running_max) drops below 1, set the running maximum equal to 1.
* Calculate drawdown using the simple formula above with the cum_rets and running_max.
* Review the plot.
```py
# Calculate the running maximum
running_max = np.maximum.accumulate(cum_rets)

# Ensure the value never drops below 1
running_max[running_max<1] = 1

# Calculate the percentage drawdown
drawdown = (cum_rets)/running_max - 1

# Plot the results
drawdown.plot()
plt.show()
```
## Historical value at risk
* Calculate VaR(95), the worst 5% of USO returns (StockReturns_perc), and assign it to var_95.
* Sort StockReturns_perc and assign it to sorted_rets.
* Plot the histogram of sorted returns (sorted_rets).
```py
# Calculate historical VaR(95)
var_95 = np.percentile(StockReturns_perc, 5)
print(var_95)

# Sort the returns for plotting
sorted_rets = StockReturns_perc.sort_values()

# Plot the probability of each sorted return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 quantile
plt.axvline(x=var_95, color='r', linestyle='-', label="VaR 95: {0:.2f}%".format(var_95))
plt.show()
```
## Historical expected shortfall
* Calculate the average of returns in StockReturns_perc where StockReturns_perc is less than or equal to var_95 and assign it to cvar_95.
* Plot the histogram of sorted returns (sorted_rets) using the plt.hist() function.
```py
# Historical CVaR 95
cvar_95 = StockReturns_perc[StockReturns_perc<=var_95].mean()
print(cvar_95)

# Sort the returns for plotting
sorted_rets = sorted(StockReturns_perc)

# Plot the probability of each return quantile
plt.hist(sorted_rets, normed=True)

# Denote the VaR 95 and CVaR 95 quantiles
plt.axvline(x=var_95, color="r", linestyle="-", label='VaR 95: {0:.2f}%'.format(var_95))
plt.axvline(x=cvar_95, color='b', linestyle='-', label='CVaR 95: {0:.2f}%'.format(cvar_95))
plt.show()
```
## Changing VaR and CVaR quantiles
* Calculate the VaR(90) for StockReturns_perc and save the result in var_90.
* Calculate the CVaR(90) for StockReturns_perc and save the result in cvar_90.
```py
# Historical VaR(90) quantiles
var_90 = np.percentile(StockReturns_perc, 10)
print(var_90)

# Historical CVaR(90) quantiles
cvar_90 = StockReturns_perc[StockReturns_perc<=var_90].mean()
print(cvar_90)

# Plot to compare
plot_hist()
```
## Parametric VaR
* Import norm from scipy.stats.
* Calculate the mean and volatility of StockReturns and assign them to mu and vol, respectively.
* Set the confidence_level for VaR(95).
* Calculate VaR(95) using the norm.ppf() function, passing in the confidence level as the first parameter, with mu and vol as the second and third parameters.
```py
# Import norm from scipy.stats
from scipy.stats import norm

# Estimate the average daily return
mu = np.mean(StockReturns)

# Estimate the daily volatility
vol = np.std(StockReturns)

# Set the VaR confidence level
confidence_level = 0.05 

# Calculate Parametric VaR
var_95 = norm.ppf(confidence_level,mu,vol)
print('Mean: ', str(mu), '\nVolatility: ', str(vol), '\nVaR(95): ', str(var_95))
```
## Scaling risk estimates
* Loop from 0 to 100 (not including 100) using the range() function.
* Set the second column of forecasted_values at each index equal to the forecasted VaR, multiplying var_95 by the square root of i + 1 using the np.sqrt() function.
```py
# Aggregate forecasted VaR
forecasted_values = np.empty([100, 2])

# Loop through each forecast period
for i in range(100):
    # Save the time horizon i
    forecasted_values[i, 0] = i
    # Save the forecasted VaR 95
    forecasted_values[i, 1] = var_95*np.sqrt(i+1)
    
# Plot the results
plot_var_scale()
```
## A random walk simulation
* Set the number of simulated days (T) equal to 252, and the initial stock price (S0) equal to 10.
* Calculate T random normal values using np.random.normal(), passing in mu and vol, and T as parameters, then adding 1 to the values and assign it to rand_rets.
* Calculate the random walk by multiplying rand_rets.cumprod() by the initial stock price and assign it to forecasted_values.
```py
# Set the simulation parameters
mu = np.mean(StockReturns)
vol = np.std(StockReturns)
T = 252
S0 = 10

# Add one to the random returns
rand_rets = np.random.normal(mu,vol,T) + 1

# Forecasted random walk
forecasted_values = S0*rand_rets.cumprod()

# Plot the random walk
plt.plot(range(0, T), forecasted_values)
plt.show()
```
## Monte Carlo simulations
* Loop from 0 to 100 (not including 100) using the range() function.
* Call the plotting function for each iteration using the plt.plot() function, passing the range of values T (range(T)) as the first argument and the forecasted_values as the second argument.
```py
# Loop through 100 simulations
for i in range(100):

    # Generate the random returns
    rand_rets = np.random.normal(mu, vol, T) + 1
    
    # Create the Monte carlo path
    forecasted_values = S0*(rand_rets).cumprod()
    
    # Plot the Monte Carlo path
    plt.plot(range(T), forecasted_values)

# Show the simulations
plt.show()
```
## Monte Carlo VaR
* Use the .append() method to append the rand_rets to sim_returns list in each iteration.
* Calculate the parametric VaR(99) using the np.percentile() function on sim_returns.
```py
# Aggregate the returns
sim_returns = []

# Loop through 100 simulations
for i in range(100):

    # Generate the Random Walk
    rand_rets = np.random.normal(mu, vol, T)
    
    # Save the results
    sim_returns.append(rand_rets)

# Calculate the VaR(99)
var_99 = np.percentile(sim_returns,1)
print("Parametric VaR(99): ", round(100*var_99, 2),"%")
```
*Finished by 2021/08/26*