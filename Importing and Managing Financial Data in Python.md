# 1. Importing stock listing data from Excel
## Import stock listing info from the NASDAQ
* Load pandas as pd.
* Use pd.read_csv() to load the file nasdaq-listings.csv into the variable nasdaq.
* Use .head() to display the first 10 rows of the data. Which data type would you expect pandas to assign to each column? What symbol is used to represent a missing value?
* Use .info() to identify dtype mismatches in the DataFrame summary. Specifically, are there any columns that should have a more appropriate type?
```py
# Import pandas library
import pandas as pd

# Import the data
nasdaq = pd.read_csv('nasdaq-listings.csv')

# Display first 10 rows
print(nasdaq.head(10))

# Inspect nasdaq
nasdaq.info()
```
## Read data using .read_csv() with adequate parsing arguments
* Read the file nasdaq-listings.csv into nasdaq with pd.read_csv(), adding the arguments na_values and parse_dates equal to the appropriate values. You should use 'NAN' for missing values, and parse dates in the Last Update column.
* Display and inspect the result using .head() and .info() to verify that the data has been imported correctly.
```py
# Import the data
nasdaq = pd.read_csv('nasdaq-listings.csv', na_values='NAN', parse_dates=['Last Update'])

# Display the head of the data
print(nasdaq.head())

# Inspect the data
nasdaq.info()
```
## Load listing info from a single sheet
* Read only the 'nyse' worksheet of 'listings.xlsx' where the symbol 'n/a' represents missing values. Assign the result to nyse.
* Display and inspect nyse with .head() and .info().
```py
# Import the data
nyse = pd.read_excel('listings.xlsx', na_values='n/a', sheetname='nyse')

# Display the head of the data
print(nyse.head())

# Inspect the data
nyse.info()
```
## Load listing data from two sheets
* Create a pd.ExcelFile() object using the file 'listings.xlsx' and assign to xls.
* Save the sheet_names attribute of xls as exchanges.
* Using exchanges to specify sheet names and n/a to specify missing values in pd.read_excel(), read the data from all sheets in xls, and assign to a dictionary listings.
* Inspect only the 'nasdaq' data in this new dictionary with .info().
```py
# Create pd.ExcelFile() object
xls = pd.ExcelFile('listings.xlsx')

# Extract sheet names and store in exchanges
exchanges = xls.sheet_names

# Create listings dictionary with all sheet data
listings = pd.read_excel(xls, sheet_name=exchanges, na_values='n/a')

# Inspect NASDAQ listings
listings['nasdaq'].info()
```
## Load all listing data and iterate over key-value dictionary pairs
* Import data in listings.xlsx from sheets 'nyse' and 'nasdaq' into the variables nyse and nasdaq, respectively. Read 'n/a' to represent missing values.
* Inspect the contents of both DataFrames with .info() to find out how many companies are reported.
With broadcasting, create a new reference column called 'Exchange' holding the values 'NYSE' or 'NASDAQ' for each DataFrame.
* Use pd.concat() to concatenate the nyse and nasdaq DataFrames, in that order, and assign to combined_listings.
```py
# Import the NYSE and NASDAQ listings
nyse = pd.read_excel('listings.xlsx', sheet_name='nyse', na_values='n/a')
nasdaq = pd.read_excel('listings.xlsx', sheet_name='nasdaq', na_values='n/a')

# Inspect nyse and nasdaq
nyse.info()
nasdaq.info()

# Add Exchange reference columns
nyse['Exchange'] = 'NYSE'
nasdaq['Exchange'] = 'NASDAQ'

# Concatenate DataFrames  
combined_listings = pd.concat([nyse, nasdaq]) 
```
## Automate the loading and combining of data from multiple Excel worksheets
* Create the pd.ExcelFile() object using the file listings.xlsx and assign to the variable xls.
Retrieve the sheet names from the .sheet_names attribute of xls and assign to exchanges.
* Create an empty list and assign to the variable listings.
Iterate over exchanges using a for loop with exchange as iterator variable. In each iteration:
* Use pd.read_excel() with xls as the the data source, exchange as the sheetname argument, and 'n/a' as na_values to address missing values. Assign the result to listing.
* Create a new column in listing called 'Exchange' with the value exchange (the iterator variable).
* Append the resulting listing DataFrame to listings.
Use pd.concat() to concatenate the contents of listings and assign to listing_data.
* Inspect the contents of listing_data using .info().
```py
# Create the pd.ExcelFile() object
xls = pd.ExcelFile('listings.xlsx')

# Extract the sheet names from xls
exchanges = xls.sheet_names

# Create an empty list: listings
listings=[]

# Import the data
for exchange in exchanges:
    listing = pd.read_excel(xls, sheetname=exchange, na_values='n/a')
    listing['Exchange'] = exchange
    listings.append(listing)

# Concatenate the listings: listing_data
listing_data = pd.concat(listings)

# Inspect the results
listing_data.info()
```
# 2. Importing financial data from the web
## Get stock data for a single company
* Import the DataReader from pandas_datareader.data and date from the datetime library.
* Using date(), set the start date to January 1, 2016 and end date to December 31, 2016.
* Set ticker to Apple's stock ticker 'AAPL' and data_source to 'iex'.
* Create a DataReader() object to import the stock prices and assign to a variable stock_prices.
* Use .head() and .info() to display and inspect the result
```py
# Import DataReader
from pandas_datareader.data import DataReader

# Import date
from datetime import date

# Set start and end dates
start = date(2016,1,1)
end = date(2016,12,31)

# Set the ticker
ticker = 'AAPL'

# Set the data source
data_source = 'iex'

# Import the stock prices
stock_prices = DataReader(ticker,data_source, start, end)

# Display and inspect the result
print(stock_prices.head())
stock_prices.info()
```
## Visualize a stock price trend
* Import matplotlib.pyplot as plt.
* Using date(), set the start and end dates to January 1, 2016 and December 31, 2016, respectively.
* Set ticker to Facebook's stock ticker 'FB' and data_source to 'iex'.
* Create a DataReader() object to import the stock prices and assign to stock_prices.
* Plot the 'close' data in stock_prices, set ticker as the title, and show the result.
```py
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Set start and end dates
start = date(2016,1,1)
end = date(2016,12,31)

# Set the ticker and data_source
ticker = 'FB'
data_source = 'iex'

# Import the data using DataReader
stock_prices = DataReader(ticker,data_source,start,end)

# Plot close
stock_prices['close'].plot(title=ticker)

# Show the plot
plt.show()
```
## Visualize the long-term gold price trend
* Use date() to set start to January 1, 1968, and set series to series code 'GOLDAMGBD228NLBM'.
* Pass series as the data,'fred' as the data source, and start as the start date to DataReader(). Assign to gold_price.
* Inspect gold_price using .info().
* Plot and show the gold_price series with title 'Gold Price'.
```py
# Set start date
start = date(1968,1,1)

# Set series code
series = 'GOLDAMGBD228NLBM'

# Import the data
gold_price = DataReader(series, 'fred', start=start)

# Inspect the price of gold
gold_price.info()

# Plot the price of gold
gold_price.plot()

# Show the plot
plt.show()
```
## Compare labor market participation and unemployment rates
* Using date(), set start to January 1, 1950.
* Create series as a list containing the series codes 'UNRATE' and 'CIVPART', in that order.
* Pass series, the data source 'fred', and the start date to DataReader(), and assign the result to econ_data.
* Use the .columns attribute to assign 'Unemployment Rate' and 'Participation Rate' as the new column labels.
* Plot and show econ_data using the subplots=True argument, and title it 'Labor Market'.
```py
# Set the start date
start = date(1950,1,1)

# Define the series codes
series = ['UNRATE', 'CIVPART']

# Import the data
econ_data = DataReader(series,'fred',start)

# Assign new column labels
econ_data.columns = ['Unemployment Rate','Participation Rate']

# Plot econ_data
econ_data.plot(subplots=True,title='Labor Market')

# Show the plot
plt.show()
```
## Compare bond and stock performance
* Using date(), set the start date to January 1, 2008.
* Set the series codes as a list containing 'BAMLHYH0A0HYM2TRIV' and 'SP500'.
* Use DataReader() to import both series from 'fred' and assign to data.
* Plot and show data with subplots, titled 'Performance Comparison'.
```py
# Set the start date
start = date(2008,1,1)

# Set the series codes
series = ['BAMLHYH0A0HYM2TRIV', 'SP500']

# Import the data
data = DataReader(series,'fred',start)

# Plot the results
data.plot(subplots=True,title='Performance Comparison')

# Show the plot
plt.show()
```
## Select the top 5 listed consumer companies
* Without using .loc[], filter listings based on the condition that the 'Sector' is equal to 'Consumer Services' and assign to consumer_services.
* Sort consumer_services by 'Market Capitalization' in descending order and assign it to consumer_services2.
* Using .head(), display the first 5 rows of the 'Company Name', 'Exchange', and 'Market Capitalization' columns.
```py
# Select companies in Consumer Services
consumer_services = listings[listings.Sector == 'Consumer Services']

# Sort consumer_services by market cap
consumer_services2 = consumer_services.sort_values('Market Capitalization', ascending=False)

# Display first 5 rows of designated columns
print(consumer_services2[['Company Name', 'Exchange', 'Market Capitalization']].head())
```
## Get the ticker of the largest consumer services company
* Use .set_index() to set the 'Stock Symbol' column as the index for listings, assigning it to listings_ss.
* Use .loc[] to filter rows where 'Sector' is equal to 'Consumer Services', select the column 'Market Capitalization', and apply .idxmax() to assign the ticker of the largest Consumer Services company to ticker.
* Using date(), set start to January 1, 2015.
* Use DataReader() to extract the stock data for the ticker from 'iex' since start and store in data.
* Plot the 'close' and 'volume' values in data, with arguments secondary_y='volume' and title=ticker
```py
# Set the index of listings to Stock Symbol
listings_ss = listings.set_index('Stock Symbol')

# Get ticker of the largest Consumer Services company
ticker = listings_ss.loc[listings_ss.Sector=='Consumer Services', 'Market Capitalization'].idxmax()

# Set the start date
start = date(2015,1,1)

# Import the stock data
data = DataReader(ticker,'iex',start)

# Plot close and volume
data[['close', 'volume']].plot(secondary_y='volume', title=ticker)

# Show the plot
plt.show()
```
## Get the largest consumer company listed after 1998
* Set 'Stock Symbol' as the index for listings.
*Use .loc[] to filter rows where 'Sector' is 'Consumer Services' and IPO Year starting 1998, and also select the 'Market Capitalization' column. Apply .idxmax() and assign the result to ticker.
* Set the start date to January 1, 2015.
* Use the DataReader to get the stock data for the ticker from 'iex' since start.
* Plot the 'close' and 'volume' prices of this company, using 'volume' for secondary_y and ticker as the title
```py
# Set Stock Symbol as the index
listings = listings.set_index('Stock Symbol')

# Get ticker of the largest consumer services company listed after 1997
ticker = listings.loc[(listings.Sector == 'Consumer Services') & (listings['IPO Year'] > 1998), 'Market Capitalization'].idxmax()

# Set the start date
start = date(2015,1,1)

# Import the stock data
data =DataReader(ticker,'iex',start)

# Plot close and volume
data[['close', 'volume']].plot(secondary_y='volume',title=ticker)

# Show the plot
plt.show()
```
## Get data for the 3 largest financial companies
* Set 'Stock Symbol' as the index for listings, assigning it to listings_ss.
* Use .loc[] to filter rows where the company sector is 'Finance'and extract the 'Market Capitalization' column. Apply .nlargest() to assign the 3 largest companies by market cap to top_3_companies.
* Convert the index of the result to a list and assign it to top_3_tickers.
* Use date() to set start to January 1, 2015.
* Use date() to set end to April 1, 2020.
* Use the DataReader() to get the stock data for the top_3_tickers from 'iex' since start until end and assign it to result.
* We are then creating a DataFrame by iterating over the ticker-data pairs and create a MultiIndex by appending 'ticker' to 'date' in the Index.
* Select 'close' from data, apply .unstack(), and inspect the resulting DataFrame, now in wide format, with .info().
```py
# Set Stock Symbol as the index
listings_ss = listings.set_index('Stock Symbol')

# Get ticker of 3 largest finance companies
top_3_companies = listings_ss.loc[listings_ss['Sector']=='Finance','Market Capitalization'].nlargest(n=3)

# Convert index to list
top_3_tickers = top_3_companies.index.tolist()

# Set start date
start = date(2015,1,1)

# Set end date
end = date(2020,4,1)

# Import stock data
result = DataReader(top_3_tickers,'iex',start,end)
result = result[~result.index.duplicated()]
data = pd.DataFrame()
for ticker in result.columns.levels[1]:
    index = pd.MultiIndex.from_arrays([
            [ticker] * len(result),
            result.index.values
            ], names=['ticker', 'date'])
    ticker_df = pd.DataFrame(index=index)
    for col in result.columns.levels[0]:
        ticker_df[col] = result[col][ticker].values
    data = pd.concat([data, ticker_df])

# Unstack and inspect result
data['close'].unstack().info()
```
# 3. Summarizing your data and visualizing the result
## List the poorest and richest countries worldwide
* Load the 'per_capita_income.csv' file into income. No additional arguments other than the file name are needed. (Note that this is a csv file.)
* Inspect the column names and data types with .info().
* Using .sort_values(), sort (in descending order) the income DataFrame by the column which contains the income information.
* Display the first five rows of income using .head() and the last five rows using .tail().
```py
# Import the data
income = pd.read_csv('per_capita_income.csv')

# Inspect the result
income.info()

# Sort the data by income
income = income.sort_values('Income per Capita', ascending=False)

# Display the first and last five rows
print(income.head())
print(income.tail())
```
## Global incomes: Central tendency
* Use the appropriate function to calculate the global mean of 'Income per Capita'.
* Use the appropriate function to calculate the global median of 'Income per Capita'.
* Using broadcasting, create a new column 'Income per Capita (,000)' equal to income['Income per Capita'] // 1000. Then use the appropriate function to calculate the mode for this new column.
```py
# Calculate the mean
print(income['Income per Capita'].mean())

# Calculate the median
print(income['Income per Capita'].median())

# Create the new column
income['Income per Capita (,000)'] = income['Income per Capita']//1000

# Calculate the mode of the new column
income['Income per Capita (,000)'].mode()
```
## Global incomes: Dispersion
* Using the appropriate functions, calculate the mean of income per capita as mean and the standard deviation as std.
* Without using .quantile(), calculate and print the upper and lower bounds of an interval of one standard deviation around the mean in a list bounds:
   * subtract std from mean as the first element
    * add std to mean as the second element
* Using .quantile() and a list of two appropriate decimal values, calculate and print the first and the third quartile of 'Income per Capita' as quantiles. Do the values match?
* Calculate and print the IQR, iqr, using the simple subtraction expression you learned in the video.
```py
# Calculate mean
mean = income['Income per Capita'].mean()

# Calculate standard deviation
std = income['Income per Capita'].std()

# Calculate and print lower and upper bounds
bounds = [mean-std, mean+std]
print(bounds)

# Calculate and print first and third quartiles
quantiles = income['Income per Capita'].quantile([0.25, 0.75])
print(quantiles)

# Calculate and print IQR
iqr = quantiles[.75] - quantiles[.25]
print(iqr)
```
## Deciles of the global income distribution
* Generate the percentages from 10% to 90% with increments of 10% using np.arange(), assign the result to quantiles, and print it.
* Using quantiles and .quantile(), calculate the deciles for the income per capita as deciles, and print the result.
* Plot and show the result as a bar chart with plt.tight_layout(). Title it 'Global Income per Capita - Deciles'.
```py
# Generate range of deciles
quantiles = np.arange(0.1,0.91,0.1)

# Print them
print(quantiles)

# Calculate deciles for 'Income per Capita'
deciles = income['Income per Capita'].quantile(quantiles)

# Print them
print(deciles)

# Plot deciles as a bar chart
deciles.plot(kind='bar', title='Global Income per Capita - Deciles')

# Make sure to use the tight layout!
plt.tight_layout()

# Show the plot
plt.show()
```
## Visualizing international income distribution
* Import seaborn as sns and matplotlib.pyplot as plt.
* Print the summary statistics provided by .describe().
* Plot and show a basic histogram of the 'Income per Capita' column with .distplot().
* Create and show a rugplot of the same data by setting the additional arguments bins equal to 50, kde to False, and rug to True.
```py
# Import seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

# Print the summary statistics for income
print(income.describe())

# Plot a basic histogram of income per capita
sns.distplot(income['Income per Capita'])

# Show the plot
plt.show()

# Plot a rugplot
sns.distplot(income['Income per Capita'],bins=50,kde=False,rug=True)

# Show the plot
plt.show()
```
## Growth rates in Brazil, China, and the US
* Load the file 'income_growth.csv' into the variable growth. Parse the 'DATE' column into dtype datetime64 and set it as the index.
* Inspect the summary statistics for these three growth rates using the appropriate function.
* Iterate over the growth.columns attribute in a for loop to access their labels. Most of the code has been outlined for you.
* In each iteration of distplot(), pass in the iteration variable column to select the respective column, set the keyword hist to False, and set label to column.
* Show the result.
```py
# Load the file into growth
growth = pd.read_csv('income_growth.csv', parse_dates=['DATE']).set_index('DATE')

# Inspect the summary statistics for the growth rates
growth.describe()

# Iterate over the three columns
for column in growth.columns:
    sns.distplot(growth[column], hist=False, label=column)
    
# Show the plot
plt.show()
```
## Highlighting values in the distribution
* Assign the column 'Income per Capita' to inc_per_capita.
* Filter to keep only the rows in inc_per_capita that are lower than the 95th percentile. Reassign to the same variable.
* Plot a default histogram for the filtered version of inc_per_capita and assign it to ax.
* Use ax.axvline() with color='b' to highlight the mean of inc_per_capita in blue,
* Use ax.axvline() with color='g' to highlight the median in green. Show the result!
```py
# Create inc_per_capita
inc_per_capita = income['Income per Capita']

# Filter out incomes above the 95th percentile
inc_per_capita = inc_per_capita[inc_per_capita < inc_per_capita.quantile(0.95)]

# Plot histogram and assign to ax
ax = sns.distplot(inc_per_capita)

# Highlight mean
ax.axvline(inc_per_capita.mean(), color='b')

# Highlight median
ax.axvline(inc_per_capita.median(), color='g')

# Show the plot
plt.show()
```
## Companies by sector on all exchanges
* Create a list exchanges containing the exact strings of the names of the exchanges in the order listed above.
* Use a for loop to iterate over exchanges with an iterator variable exchange that contains the name of each exchange. In each iteration:
 * Apply .value_counts() to 'Sector' and assign the result to sectors.
 * Sort sectors in descending order and plot them in a bar plot.
 * Show the result.
```
# Create the list exchanges
exchanges = ['amex', 'nasdaq', 'nyse']

# Iterate over exchanges then plot and show result
for exchange in exchanges:
    sectors = listings[exchange]['Sector'].value_counts()
    # Sort in descending order and plot
    sectors.sort_values(ascending=False).plot(kind='bar')
    # Show the plot
    plt.show()
```
## Technology IPOs by year on all exchanges
* Use a for loop with iterator variable exchange that contains the name of each exchange.
* In each iteration, append the DataFrame corresponding to the key exchange in listings to all_listings.
* After the loop completes, use pd.concat() to combine the three DataFrames in all_listings and assign the result to listing_data.
* Filter listing_data for 'Technology' companies and assign the result to tech_companies.
* Assign the 'IPO Year' column from tech_companies to ipo years.
* For this data, use .dropna() to remove missing values and .astype() to convert to int.
* Apply .value_counts() to ipo_years, sort the years in ascending order, and create a bar plot titled 'Tech IPOs by Year'.
* Rotate xticks by 45 degrees and show the result.
```py
# Create lists
exchanges = ['amex', 'nasdaq', 'nyse']
all_listings = []

# Use for loop to create listing_data
for exchange in exchanges:
    all_listings.append(listings[exchange])
    
# Combine DataFrames
listing_data = pd.concat(all_listings)

# Select tech companies
tech_companies = listing_data[listing_data['Sector'] == 'Technology']

# Create ipo_years
ipo_years = tech_companies['IPO Year']

# Drop missing values and convert to int
ipo_years = ipo_years.dropna().astype(int)

# Count values, sort ascending by year, and create a bar plot
ipo_years.value_counts(ascending=True).plot(kind='bar', title='Tech IPOs by Year')

# Rotate xticks and show result
plt.xticks(rotation=45)

# Show the plot
plt.show()
```
# 4. Aggregating and describing your data by category
## Median market capitalization by sector
* Inspect nyse using .info().
* With broadcasting and .div(), create a new column market_cap_m that contains the market capitalization in million USD.
* Omit the column 'Market Capitalization' with .drop().
* Apply the .groupby() method to nyse, using 'Sector' as the column to group your data by.
* Calculate the median of the market_cap_m column as median_mcap_by_sector.
* Plot the result as a horizontal bar chart with the title 'NYSE - Median Market Capitalization'. Use plt.xlabel() with 'USD mn' to add a label.
* Show the result.
```py
# Inspect NYSE data
nyse.info()

# Create market_cap_m
nyse['market_cap_m'] = nyse['Market Capitalization'].div(1e6)

# Drop market cap column
nyse = nyse.drop('Market Capitalization', axis=1)

# Group nyse by sector
mcap_by_sector = nyse.groupby('Sector')

# Calculate median
median_mcap_by_sector = mcap_by_sector['market_cap_m'].median()

# Plot and show as horizontal bar chart
median_mcap_by_sector.plot(kind='bar', title='NYSE - Median Market Capitalization')

# Add the label
plt.xlabel('USD mn')

# Show the plot
plt.show()
```
## Median market capitalization by IPO year
* Inspect and display listings using .info() and .head().
* Using broadcasting, create a new column market_cap_m for listings that contains the market cap in millions of USD.
* Select all companies with an 'IPO Year' after 1985.
* Drop all missing values in the 'IPO Year' column, and convert the remaining values to dtype integer.
* Group listings by 'IPO Year', select the market_cap_m column and calculate the median, sort with .sort_index(), and assign the result to ipo_by_year.
* Plot and show the results as a bar chart.
```py
# Inspect listings
listings.info()

# Show listings head
print(listings.head())

# Create market_cap_m
listings['market_cap_m'] = listings['Market Capitalization'].div(1e6)

# Select companies with IPO after 1985
listings = listings[listings['IPO Year'] > 1985]

# Drop missing values and convert to integers
listings['IPO Year'] = listings['IPO Year'].dropna().astype(int)

# Calculate the median market cap by IPO Year and sort the index

ipo_by_year = listings.groupby('IPO Year').market_cap_m.median().sort_index()

# Plot results as a bar chart
ipo_by_year.plot(kind='bar')

# Show the plot
plt.show()
```
## All summary statistics by sector
* Inspect the nasdaq data using .info().
* Create a new column market_cap_m that contains the market cap in millions of USD. On the next line, drop the column 'Market Capitalization'.
* Group your nasdaq data by 'Sector' and assign to nasdaq_by_sector.
* Call the method .describe() on nasdaq_by_sector, assign to summary, and print the result.
* This works, but result is in long format and uses a pd.MultiIndex() that you saw earlier. Convert summary to wide format by calling .unstack().
```py
# Inspect NASDAQ data
nasdaq.info()

# Create market_cap_m
nasdaq['market_cap_m'] = nasdaq['Market Capitalization'].div(1e6)

# Drop the Market Capitalization column
nasdaq.drop('Market Capitalization', axis=1, inplace=True)

# Group nasdaq by Sector
nasdaq_by_sector = nasdaq.groupby('Sector')

# Create summary statistics by sector
summary = nasdaq_by_sector.describe()

# Print the summary
print(summary)

# Unstack 
summary = summary.unstack()

# Print the summary again
print(summary)
```
## Company value by exchange and sector
* Group your data by both 'Sector' and 'Exchange', assigning the result to by_sector_exchange.
* Calculate the median market capitalization for by_sector_exchange and assign to mcap_by_sector_exchange.
* Display the first 5 rows of the result with .head().
* Call .unstack() on mcap_by_sector_exchange to move the Exchange labels to the columns, and assign to mcap_unstacked.
* Plot the result as a bar chart with the title 'Median Market Capitalization by Exchange' and xlabel set to 'USD mn',
* Show the result.
```py
# Group listings by Sector and Exchange
by_sector_exchange = listings.groupby(['Sector', 'Exchange'])

# Calculate the median market cap
mcap_by_sector_exchange = by_sector_exchange.market_cap_m.median()

# Display the head of the result
print(mcap_by_sector_exchange.head())

# Unstack mcap_by_sector_exchange
mcap_unstacked = mcap_by_sector_exchange.unstack()

# Plot as a bar chart
mcap_unstacked.plot(kind='bar', title='Median Market Capitalization by Exchange')

# Set the x label
plt.xlabel('USD mn')

# Show the plot
plt.show()
```
## Calculate several metrics by sector and exchange
* With broadcasting and .div(), create a new column 'market_cap_m' that contains the market capitalization data in millions of USD.
* Group your data by both 'Sector' and 'Exchange', assigning the result to by_sector_exchange.
* Assign the market_cap_m column of by_sector_exchange to a variable bse_mcm.
* Use .agg() and a dictionary argument to calculate the mean, median, and standard deviation for market_cap_m storing the results in 'Average', 'Median', and 'Standard Deviation', respectively, and assign to summary.
* Print the result to your console.
```py
## 
# Create market_cap_m
listings['market_cap_m'] = listings['Market Capitalization'].div(1e6)

# Group listing by both Sector and Exchange
by_sector_exchange = listings.groupby(['Sector', 'Exchange'])

# Subset market_cap_m of by_sector_exchange
bse_mcm = by_sector_exchange['market_cap_m']

# Calculate mean, median, and std in summary
summary = bse_mcm.agg({'Average': 'mean', 'Median': 'median', 'Standard Deviation': 'std'})

# Print the summary
print(summary)
```
## Plot IPO timeline for all exchanges using countplot()
* Filter listings to only include IPO years after the year 2000.
* Convert the data in the column 'IPO Year' to integers.
* Plot a sns.countplot() of listings using 'IPO Year' as the x variable and 'Exchange' for hue.
* Rotate the xticks() by 45 degrees and show the result.
```py
# Select IPOs after 2000
listings = listings[listings['IPO Year'] > 2000]

# Convert IPO Year to integer
listings['IPO Year'] = listings['IPO Year'].astype(int)

# Create a countplot
sns.countplot(x='IPO Year', hue='Exchange', data=listings)

# Rotate xticks and show result
plt.xticks(rotation=45)

# Show the plot
plt.show()
```
## Global median per capita income over time
* Inspect income_trend using .info().
* Create a sns.barplot() using the column 'Year' for x and 'Income per Capita' for y, and show the result after rotating the xticks by 45 degrees.
* Use plt.close() after the initial plt.show() to be able to show a second plot.
* Create a second sns.barplot() with the same x and y settings, using estimator=np.median to calculate the median, and show the result.
```py
# Inspect the data
income_trend.info()

# Create barplot
sns.barplot(x='Year', y='Income per Capita', data=income_trend)

# Rotate xticks
plt.xticks(rotation=45)

# Show the plot
plt.show()

# Close the plot
plt.close()

# Create second barplot
sns.barplot(x='Year', y='Income per Capita', data=income_trend, estimator=np.median)

# Rotate xticks
plt.xticks(rotation=45)

# Show the plot
plt.show()
```
## Calculate several metrics by sector and IPO year
* Import seaborn as sns.
* Filter listings to have companies with IPOs after 2000 from all exchanges except the 'amex'.
* Convert the data in column 'IPO Year' to integers.
* Create the column market_cap_m to express market cap in USD million.
* Filter market_cap_m to exclude values above the 95th percentile.
* Create a pointplot of listings using the column 'IPO Year' for x, 'market_cap_m' for y, and 'Exchange' for hue. Show the result after rotating the xticks by 45 degrees.
```py
# Import the seaborn library as sns
import seaborn as sns

# Exclude IPOs before 2000 and from the 'amex'
listings = listings[(listings['IPO Year'] > 2000) & (listings.Exchange != 'amex')]

# Convert IPO Year to integer
listings['IPO Year'] = listings['IPO Year'].astype(int)

# Create market_cap_m
listings['market_cap_m'] = listings['Market Capitalization'].div(1e6)

# Exclude outliers
listings = listings[listings.market_cap_m < listings.market_cap_m.quantile(.95)]

# Create the pointplot
sns.pointplot(x='IPO Year', y='market_cap_m', hue='Exchange', data=listings)

# Rotate xticks
plt.xticks(rotation=45)

# Show the plot
plt.show()
```
## Inflation trends in China, India, and the US
* Inspect inflation using .info().
* Group inflation by 'Country' and assign to inflation_by_country.
* In a for loop, iterate over country, data pairs returned by inflation_by_country. In each iteration, use .plot() on data with title set to country to show the historical time series.
```py
# Inspect the inflation data
inflation.info()

# Create inflation_by_country
inflation_by_country = inflation.groupby('Country')

# Iterate over inflation_by_country and plot the inflation time series per country
for country, data in inflation_by_country:
    # Plot the data
    data.plot(title=country)
    # Show the plot
    plt.show()
```
## Distribution of inflation rates in China, India, and the US
* Create and show a boxplot of the inflation data with 'Country' for x and 'Inflation' for y.
* Create and show sns.swarmplot() with the same arguments.
```py
# Create boxplot
sns.boxplot(x='Country', y='Inflation', data=inflation)

# Show the plot
plt.show()

# Close the plot
plt.close()

# Create swarmplot
sns.swarmplot(x='Country', y='Inflation', data=inflation)

# Show the plot
plt.show()
```
*Finished by 2021/09/07*




