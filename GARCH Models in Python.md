# 1. GARCH Model Fundamentals
## Observe volatility clustering
* Calculate daily returns as percentage price changes and save it to the DataFrame sp_price in a new column called Return.
* View the data by printing out the last 10 rows.
* Plot the Return column and observe signs of volatility clustering.
```py
# Calculate daily returns as percentage price changes
sp_price['Return'] = 100 * (sp_price['Close'].pct_change())

# View the data
print(sp_price.tail(10))

# plot the data
plt.plot(sp_price['Return'], color = 'tomato', label = 'Daily Returns')
plt.legend(loc='upper right')
plt.show()
```
## Calculate volatility
* Plot the 'Return' column in sp_data.
* Calculate the standard deviation of 'Return' data.
```py
# Plot the price returns
plt.plot(sp_data['Return'], color = 'orange')
plt.show()

# Calculate daily std of returns
std_daily = sp_data['Return'].std()
print('Daily volatility: ', '{:.2f}%'.format(std_daily))
```
* Calculate monthly volatility from daily volatility.
* Calculate annual volatility from daily volatility.
```py
# Plot the price returns
plt.plot(sp_data['Return'], color = 'orange')
plt.show()

# Calculate daily std of returns
std_daily = sp_data['Return'].std()
print('Daily volatility: ', '{:.2f}%'.format(std_daily))

# Convert daily volatility to monthly volatility
std_monthly = math.sqrt(21) * std_daily
print ('Monthly volatility: ', '{:.2f}%'.format(std_monthly))

# Convert daily volatility to annaul volatility
std_annual = math.sqrt(252) * std_daily
print ('Annual volatility: ', '{:.2f}%'.format(std_annual))
```
## Simulate ARCH and GARCH series
* Simulate an ARCH(1) process with omega = 0,1. alpha = 0.7.
* Simulate a GARCH(1,1) process with omega = 0,1. alpha = 0.7, and beta = 0.1.
* Plot the simulated ARCH variances and GARCH variances respectively.
```py
# Simulate a ARCH(1) series
arch_resid, arch_variance = simulate_GARCH(n= 200, 
 omega = 0.1, alpha = 0.7)
# Simulate a GARCH(1,1) series
garch_resid, garch_variance = simulate_GARCH(n= 200,  omega = .1, alpha = .7, beta =.1)
# Plot the ARCH variance
plt.plot(arch_variance, color = 'red', label = 'ARCH Variance')
# Plot the GARCH variance
plt.plot(garch_variance, color = 'orange', label = 'GARCH Variance')
plt.legend()
plt.show()
```
## Observe the impact of model parameters
* Generate a GARCH(1,1) process with 200 simulations, omega = 0.1, alpha = 0.3, and beta = 0.2 as input.
```py
# First simulated GARCH
sim_resid, sim_variance = simulate_GARCH(n = 200,  omega = .1, alpha = .3, beta = .2)
plt.plot(sim_variance, color = 'orange', label = 'Variance')
plt.plot(sim_resid, color = 'green', label = 'Residuals')
plt.legend(loc='upper right')
plt.show()
```
* Generate a GARCH(1,1) process with 200 simulations,omega = 0.1, alpha = 0.3, and beta = 0.6 as input.
```py
# Second simulated GARCH
sim_resid, sim_variance = simulate_GARCH(n = 200, omega = .1, alpha = .3, beta =.6)
plt.plot(sim_variance, color = 'red', label = 'Variance')
plt.plot(sim_resid, color = 'deepskyblue', label = 'Residuals')
plt.legend(loc='upper right')
plt.show()
```
## Implement a basic GARCH model
* Define a GARCH(1,1) model basic_gm with 'constant' mean and 'normal' distribution of the residuals.
* Fit the model basic_gm.
* Print a summary of the fitted GARCH model.
* Plot the model estimated result.
```py
# Specify GARCH model assumptions
basic_gm = arch_model(sp_data['Return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit(update_freq = 4)

# Display model fitting summary
print(gm_result.summary())

# Plot fitted results
gm_result.plot()
plt.show()
```
## Make forecast with GARCH models
* Define a basic GARCH(1,1) model basic_gm.
Fit the model.
* Use the fitted model gm_result to make 5-period ahead variance forecast.
Print out the variance forecast result.
```py
# Specify a GARCH(1,1) model
basic_gm = arch_model(sp_data['Return'], p = 1, q = 1, 
                      mean = 'constant', vol = 'GARCH', dist = 'normal')
# Fit the model
gm_result = basic_gm.fit()

# Make 5-period ahead forecast
gm_forecast = gm_result.forecast(horizon = 5)

# Print the forecast variance
print(gm_forecast.variance[-1:])
```
# 2. GARCH Model Configuration
## Plot distribution of standardized residuals
* Obtain model estimated residuals and save it in gm_resid.
* Obtain model estimated volatility and save it in gm_std.
* Calculate the standardized residuals gm_std_resid.
* Plot a histogram of gm_std_resid.
```py
# Obtain model estimated residuals and volatility
gm_resid = gm_result.resid
gm_std = gm_result.conditional_volatility

# Calculate the standardized residuals
gm_std_resid = gm_resid /gm_std

# Plot the histogram of the standardized residuals
plt.hist(gm_std_resid, bins = 50, 
         facecolor = 'orange', label = 'Standardized residuals')
plt.hist(normal_resid, bins = 50, 
         facecolor = 'tomato', label = 'Normal residuals')
plt.legend(loc = 'upper left')
plt.show()
```
## Fit a GARCH with skewed t-distribution
* Define a GARCH model skewt_gm with a skewed Student's t-distribution assumption.
* Fit the model and save the result in skewt_result
* Save the model estimated conditional volatility in skewt_vol.
* Plot skewt_vol together with the normal GARCH estimations and the actual return data.
```py
# Specify GARCH model assumptions
skewt_gm = arch_model(sp_data['Return'], p = 1, q = 1, mean = 'constant', vol = 'GARCH', dist = 'skewt')
# Fit the model
skewt_result = skewt_gm.fit()

# Get model estimated volatility
skewt_vol = skewt_result.conditional_volatility

# Plot model fitting results
plt.plot(skewt_vol, color = 'gold', label = 'Skewed-t Volatility')
plt.plot(normal_vol, color = 'red', label = 'Normal Volatility')
plt.plot(sp_data['Return'], color = 'grey', 
         label = 'Daily Returns', alpha = 0.4)
plt.legend(loc = 'upper right')
plt.show()
```
## Effect of mean model on volatility predictions
* Print out and review model fitting summaries of cmean_result and armean_result.
* Plot the volatility estimation cmean_vol and armean_vol from both models.
* Use .corrcoef() function from numpy package to calculate the correlation coefficient.
```py
# Print model summary of GARCH with constant mean
print(cmean_result.summary())
# Print model summary of GARCH with AR mean
print(armean_result.summary())

# Plot model volatility 
plt.plot(cmean_vol, color = 'blue', label = 'Constant Mean Volatility')
plt.plot(armean_vol, color = 'red', label = 'AR Mean Volatility')
plt.legend(loc = 'upper right')
plt.show()

# Check correlation of volatility estimations
print(np.corrcoef(cmean_vol, armean_vol)[0,1])
```
## Fit GARCH models to cryptocurrency
* Define a GJR-GARCH model as gjr_gm.
* Print and review the model fitting summary of gjrgm_result.
```py
# Specify model assumptions
gjr_gm = arch_model(bitcoin_data['Return'], p = 1, q = 1, o = 1, vol = 'GARCH', dist = 't')

# Fit the model
gjrgm_result = gjr_gm.fit(disp = 'off')

# Print model fitting summary
print(gjrgm_result.summary())
```
* Define an EGARCH model as egarch_gm.
* Print and review the model fitting summary of
```py
# Specify model assumptions
egarch_gm = arch_model(bitcoin_data['Return'], p = 1, q = 1, o = 1, vol = 'EGARCH', dist = 't')

# Fit the model
egarch_result = egarch_gm.fit(disp = 'off')

# Print model fitting summary
print(egarch_result.summary())
```
## Compare GJR-GARCH with EGARCH
* Plot the actual Bitcoin returns.
* Plot GJR-GARCH estimated volatility.
* Plot EGARCH estimated volatility.
```py
# Plot the actual Bitcoin returns
plt.plot(bitcoin_data['Return'], color = 'grey', alpha = 0.4, label = 'Price Returns')

# Plot GJR-GARCH estimated volatility
plt.plot(gjrgm_vol, color = 'gold', label = 'GJR-GARCH Volatility')

# Plot EGARCH  estimated volatility
plt.plot(egarch_vol, color = 'red', label = 'EGARCH Volatility')

plt.legend(loc = 'upper right')
plt.show()
```
## GARCH rolling window forecast
* Fixed rolling window forecast
* Perform GARCH forecast and save the result of each iteration.
* Plot the variance forecast results.
```py
for i in range(30):
    # Specify fixed rolling window size for model fitting
    gm_result = basic_gm.fit(first_obs = i + start_loc, 
                             last_obs = i + end_loc, update_freq = 5)
    # Conduct 1-period variance forecast and save the result
    temp_result = gm_result.forecast(horizon = 1).variance
    fcast = temp_result.iloc[i + end_loc]
    forecasts[fcast.name] = fcast
# Save all forecast to a dataframe    
forecast_var = pd.DataFrame(forecasts).T

# Plot the forecast variance
plt.plot(forecast_var, color = 'red')
plt.plot(sp_data.Return['2019-4-1':'2019-5-10'], color = 'green')
plt.show()
```
## Compare forecast results
* Print out the first 5 rows of variance forecast stored in variance_expandwin and variance_fixedwin respectively.
* Calculate volatility from variance forecast with an expanding window and a fixed rolling window approach respectively. Use a pre-defined function from numpy (imported as np).
* Plot both volatility forecast calculated from the previous step in one chart.
```py
# Print header of variance forecasts with expanding and fixed window
print(variance_expandwin.head())
print(variance_fixedwin.head())

# Calculate volatility from variance forecast with an expanding window
vol_expandwin = np.sqrt(variance_expandwin)
# Calculate volatility from variance forecast with a fixed rolling window
vol_fixedwin = np.sqrt(variance_fixedwin)

# Plot volatility forecast with an expanding window
plt.plot(vol_expandwin, color = 'blue')
# Plot volatility forecast with a fixed rolling window
plt.plot(vol_fixedwin, color = 'red')
plt.plot(bitcoin_data.Return['2019-4-1':'2019-9-15'], color = 'chocolate')
plt.show()
```
# 3.Model Performance Evaluation
## Simplify the model with p-values
* Print the model fitting summary.
* Get the model parameters and p-values, and save them in a DataFrame para_summary.
* Print and review para_summary.
```py
# Print model fitting summary
print(gm_result.summary())

# Get parameter stats from model summary
para_summary = pd.DataFrame({'parameter':gm_result.params,
                             'p-value': gm_result.pvalues})

# Print out parameter stats
print(para_summary)
```
* According to the p-values, which parameter below is not statistically significant given a confidence level of 5%?
gamma

## Simplify the model with t-statistics
* Get the model parameters, standard errors and t-statistic, and save them in the DataFrame para_summary.
* Compute t-statistics manually using parameter values and their standard errors, and save the calculation result in calculated_t.
* Print and review calculated_t.
* Print and review para_summary.
```py
# Get parameter stats from model summary
para_summary = pd.DataFrame({'parameter':gm_result.params,
                             'std-err': gm_result.std_err, 
                             't-value': gm_result.tvalues})

# Verify t-statistic by manual calculation
calculated_t = para_summary['parameter']/para_summary['std-err']

# Print calculated t-value
print(calculated_t)

# Print parameter stats
print(para_summary)
```
## ACF plot
* Import the module needed for ACF plots from the statsmodels package.
* Plot the GARCH model standardized residuals saved in std_resid.
* Generate an ACF plot of the standardized residuals, and set the confidence level to 0.05.
```py
# Import the Python module
from statsmodels.graphics.tsaplots import plot_acf

# Plot the standardized residuals
plt.plot(std_resid)
plt.title('Standardized Residuals')
plt.show()

# Generate ACF plot of the standardized residuals
plot_acf(std_resid, alpha = .05)
plt.show()
```
## Ljung-Box test
* Import the module needed for Ljung-Box tests from the statsmodels package.
* Perform a Ljung-Box test up to lag 10, and save the result in lb_test.
* Print and review p-values from the Ljung-Box test result.
```py
# Import the Python module
from statsmodels.stats.diagnostic import acorr_ljungbox

# Perform the Ljung-Box test
lb_test = acorr_ljungbox(std_resid , lags=10)

# Print the p-values
print('P-values are: ', lb_test[1])
```
## Pick a winner based on log-likelihood
* Print and review model fitting summaries in normal_result and skewt_result respectively.
* Print the log-likelihood in normal_result and skewt_result respectively.
```py
# Print normal GARCH model summary
print(normal_result.summary())
# Print skewed GARCH model summary
print(skewt_result.summary())

# Print the log-likelihodd of normal GARCH
print('Log-likelihood of normal GARCH :', normal_result.loglikelihood)
# Print the log-likelihodd of skewt GARCH
print('Log-likelihood of skewt GARCH :', skewt_result.loglikelihood)
```
* Which model is better according to their log-likelihood values?
* Skewt GARCH

## Pick a winner based on AIC/BIC
* Print the AIC in gjrgm_result and egarch_result respectively.
* Print the BIC in gjrgm_result and egarch_result respectively.
```py
# Print the AIC GJR-GARCH
print('AIC of GJR-GARCH model :', gjrgm_result.aic)
# Print the AIC of EGARCH
print('AIC of EGARCH model :', egarch_result.aic)

# Print the BIC GJR-GARCH
print('BIC of GJR-GARCH model :', gjrgm_result.bic)
# Print the BIC of EGARCH
print('BIC of EGARCH model :', egarch_result.bic)
```
* Which model is better according to their AIC/BIC values?
* EGARCH model (the aic/bic lower, the better)

## Backtesting with MAE, MSE
* In evaluate(), perform the MAE calculation by calling the corresponding function from sklean.metrics.
* In evaluate(), perform the MSE calculation by calling the corresponding function from sklean.metrics.
* Pass variables to evaluate() in order to perform the backtest.
```py
def evaluate(observation, forecast): 
    # Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print('Mean Absolute Error (MAE): {:.3g}'.format(mae))
    # Call sklearn function to calculate MSE
    mse = mean_squared_error(observation, forecast)
    print('Mean Squared Error (MSE): {:.3g}'.format(mse))
    return mae, mse

# Backtest model with MAE, MSE
evaluate(actual_var,forecast_var)
```
# 4. GARCH in Action
## Compute parametric VaR
* Compute 0.05 quantile from the assumed Student's t-distribution.
* Calculate VaR using mean_forecast, variance_forecast from the GARCH model and the quantile from the previous step.
```py
# Obtain the parametric quantile
q_parametric = basic_gm.distribution.ppf(0.05, nu)
print('5% parametric quantile: ', q_parametric)
    
# Calculate the VaR
VaR_parametric = mean_forecast.values + np.sqrt(variance_forecast).values * q_parametric
# Save VaR in a DataFrame
VaR_parametric = pd.DataFrame(VaR_parametric, columns = ['5%'], index = variance_forecast.index)

# Plot the VaR
plt.plot(VaR_parametric, color = 'red', label = '5% Parametric VaR')
plt.scatter(variance_forecast.index,bitcoin_data.Return['2019-1-1':], color = 'orange', label = 'Bitcoin Daily Returns' )
plt.legend(loc = 'upper right')
plt.show()
```
## Compute empirical VaR
* Compute 0.05 quantile from the GARCH standardized residuals std_resid.
* Calculate VaR using mean_forecast, variance_forecast from the GARCH model and the quantile from the previous step.
```py
# Obtain the empirical quantile
q_empirical = std_resid.quantile(0.05)
print('5% empirical quantile: ', q_empirical)

# Calculate the VaR
VaR_empirical = mean_forecast.values + np.sqrt(variance_forecast).values * q_empirical
# Save VaR in a DataFrame
VaR_empirical = pd.DataFrame(VaR_empirical, columns = ['5%'], index = variance_forecast.index)

# Plot the VaRs
plt.plot(VaR_empirical, color = 'brown', label = '5% Empirical VaR')
plt.plot(VaR_parametric, color = 'red', label = '5% Parametric VaR')
plt.scatter(variance_forecast.index,bitcoin_data.Return['2019-1-1':], color = 'orange', label = 'Bitcoin Daily Returns' )
plt.legend(loc = 'upper right')
plt.show()
```
## Compute GARCH covariance
* Calculate correlation between GARCH standardized residuals resid_eur and resid_cad.
* Calculate covariance with GARCH volatility vol_eur, vol_cad and the correlation computed in the previous step.
* Plot the calculated covariance.
```py
# Calculate correlation
corr = np.corrcoef(resid_eur, resid_cad)[0,1]
print('Correlation: ', corr)

# Calculate GARCH covariance
covariance =  corr * vol_cad * vol_eur

# Plot the data
plt.plot(covariance, color = 'gold')
plt.title('GARCH Covariance')
plt.show()
```
## Compute dynamic portfolio variance
* Set the EUR/USD weight Wa1 in portfolio a to 0.9, and Wb1 in portfolio b to 0.5.
* Calculate the variance portvar_a for portfolio a with variance_eur, variance_cad and covariance; do the same to compute portvar_b for portfolio b.
```py
# Define weights
Wa1 = 0.9
Wa2 = 1 - Wa1
Wb1 = 0.5
Wb2 = 1 - Wb1

# Calculate portfolio variance
portvar_a = Wa1**2 * variance_eur + Wa2**2 * variance_cad + 2*Wa1*Wa2 * covariance
portvar_b = Wb1**2 * variance_eur + Wb2**2 * variance_cad + 2*Wb1*Wb2*covariance

# Plot the data
plt.plot(portvar_a, color = 'green', label = 'Portfolio a')
plt.plot(portvar_b, color = 'deepskyblue', label = 'Portfolio b')
plt.legend(loc = 'upper right')
plt.show()
```
## Compute dynamic stock Beta
* Compute the correlation coefficient between Tesla and S&P 500 using standardized residuals from fitted GARCH models (teslaGarch_resid, spGarch_resid).
* Compute Tesla stock Beta using Tesla volatility (teslaGarch_vol), S&P 500 volatility (spGarch_vol) and correlation computed from the previous step.
```py
# Compute correlation between SP500 and Tesla
correlation = np.corrcoef(teslaGarch_resid, spGarch_resid)[0, 1]

# Compute the Beta for Tesla
stock_beta = correlation * (teslaGarch_vol / spGarch_vol)

# Plot the Beta
plt.title('Tesla Stock Beta')
plt.plot(stock_beta)
plt.show()
```
*Finished by 2021/08/31*