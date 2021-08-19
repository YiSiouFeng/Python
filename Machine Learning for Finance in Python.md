# 1. Preparing data and a linear model

## Explore the data with some EDA
* Print out the first 5 lines of the two DataFrame (lng_df and spy_df) and examine their contents.
* Use the pandas library to plot raw time series data for 'SPY' and 'LNG' with the adjusted close price ('Adj_Close') -- set legend=True in .plot().
* Use plt.show() to show the raw time series plot (matplotlib.pyplot has been imported as plt).
* Use pandas and matplotlib to make a histogram of the adjusted close 1-day percent difference (use .pct_change()) for SPY and LNG.
```py
print(lng_df.head())  # examine the DataFrames
print(spy_df.head())  # examine the SPY DataFrame

# Plot the Adj_Close columns for SPY and LNG
spy_df['Adj_Close'].plot(label='SPY', legend=True)
lng_df['Adj_Close'].plot(label='LNG', legend=True, secondary_y=True)
plt.show()  # show the plot
plt.clf()  # clear the plot space

# Histogram of the daily price change percent of Adj_Close for LNG
lng_df['Adj_Close'].pct_change().plot.hist(bins=50)
plt.xlabel('adjusted close 1-day percent change')
plt.show()
```
## Correlations
* Create the 5-day future price (as 5d_future_close) with pandas' .shift(-5).
* Use pct_change(5) on 5d_future_close and Adj_Close to create the future 5-day % price change (5d_close_future_pct), and the current 5-day % price change (5d_close_pct).
* Examine correlations between the two 5-day percent price change columns with .corr() on lng_df.
* Using plt.scatter(), make a scatterplot of 5d_close_pct vs 5d_close_future_pct.
```py
# Create 5-day % changes of Adj_Close for the current day, and 5 days in the future
lng_df['5d_future_close'] = lng_df['Adj_Close'].shift(-5)
lng_df['5d_close_future_pct'] = lng_df['5d_future_close'].pct_change(5)
lng_df['5d_close_pct'] = lng_df['Adj_Close'].pct_change(5)

# Calculate the correlation matrix between the 5d close pecentage changes (current and future)
corr = lng_df[['5d_close_pct', '5d_close_future_pct']].corr()
print(corr)

# Scatter the current 5-day percent change vs the future 5-day percent change
plt.scatter(lng_df['5d_close_pct'], lng_df['5d_close_future_pct'])
plt.show()
```
## Create moving average and RSI features
* Create a list of feature names (start with a list containing only '5d_close_pct').
* Use timeperiods of 14, 30, 50, and 200 to calculate moving averages with talib.SMA() from adjusted close prices (lng_df['Adj_Close']).
* Normalize the moving averages with the adjusted close by dividing by Adj_Close.
* Within the loop, calculate RSI with talib.RSI() from Adj_Close and using n for the timeperiod.
```py
feature_names = ['5d_close_pct']  # a list of the feature names for later

# Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
for n in [14,30,50,200]:

    # Create the moving average indicator and divide by Adj_Close
    lng_df['ma' + str(n)] = talib.SMA(lng_df['Adj_Close'].values, timeperiod=n) / (lng_df['Adj_Close'])
    # Create the RSI indicator
    lng_df['rsi' + str(n)] = talib.RSI(lng_df['Adj_Close'].values, timeperiod=n)
    
    # Add rsi and moving average to the feature name list
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

print(feature_names)
```
## Create features and targets
* Drop the missing values from lng_df with .dropna() from pandas.
* Create a variable containing our targets, which are the '5d_close_future_pct' values.
* Create a DataFrame containing both targets (5d_close_future_pct) and features (contained in the existing list feature_names) so we can check the correlations.
```py
# Drop all na values
lng_df = lng_df.dropna()

# Create features and targets
# use feature_names for features; '5d_close_future_pct' for targets
features = lng_df[feature_names]
targets = lng_df['5d_close_future_pct']

# Create DataFrame from target column and feature columns
feature_and_target_cols = ['5d_close_future_pct'] + feature_names
feat_targ_df = lng_df[feature_and_target_cols]

# Calculate correlation matrix
corr = feat_targ_df.corr()
print(corr)
```
## Check the correlations
* Plot a heatmap of the correlation matrix (corr) we calculated in the last exercise (seaborn has been imported as sns for you).
* Turn annotations on using the sns.heatmap() option annot=True. The font-size has already been set for you using annot_kws = {"size": 14}.
* Show the plot with plt.show().`
* Inspect the heatmap that you generated in the previous step. Which feature/variable exhibits the highest correlation with the target (5d_close_future_pct)?

  * Note: If you are having trouble reading the heatmap, you can click the two arrows next to 'Plots' in the upper left-hand window of your heatmap to expand it.
* Clear the plot area with plt.clf() to prepare for our second plot.
* Create a scatter plot of the most correlated feature/variable with the target (5d_close_future_pct) from the lng_df DataFrame.
```py
# Plot heatmap of correlation matrix
sns.heatmap(corr, annot=True, annot_kws = {"size": 14})
plt.yticks(rotation=0, size = 14); plt.xticks(rotation=90, size = 14)  # fix ticklabel directions and size
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot
plt.clf()  # clear the plot area

# Create a scatter plot of the most highly correlated variable with the target
plt.scatter(lng_df['ma200'], lng_df['5d_close_future_pct'])
plt.show()
```
## Create train and test features
* Import the statsmodels.api library with the alias sm.
* Add a constant to the features variable using statsmodels' .add_constant() function.
* Set train_size as 85% of the total number of datapoints (number of rows) using the .shape[0] property of features or targets.
* Break up linear_features and targets into train and test sets using train_size and Python indexing (e.g. [start:stop]).
```py
# Import the statsmodels.api library with the alias sm
import statsmodels.api as sm

# Add a constant to the features
linear_features = sm.add_constant(features)

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * features.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
print(linear_features.shape, train_features.shape, test_features.shape)
```

## Fit a linear model
* Fit the linear model (using the .fit() method) and save the results in the results variable.
* Print out the results summary with the .summary() function.
* Print out the p-values from the results (the .pvalues property of results).
* Make predictions from the train_features and test_features using the .predict() function of our results object.
```py
# Create the linear model and complete the least squares fit
model = sm.OLS(train_targets, train_features)
results = model.fit()  # fit the model
print(results.summary())

# examine pvalues
# Features with p <= 0.05 are typically considered significantly different from 0
print(results.pvalues)

# Make predictions from our model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)
```
## Evaluate our results
* Show test_predictions vs test_targets in a scatterplot, with 20% opacity for the points (use the alpha parameter to set opacity).
* Plot the perfect prediction line using np.arange() and the minimum and maximum values from the xaxis (xmin, xmax).
* Display the legend on the plot with plt.legend().
```py
# Scatter the predictions vs the targets with 20% opacity
plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, test_targets, alpha=0.2, color='r', label='test')

# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

# Set the axis labels and show the plot
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  # show the legend
plt.show()
```
# 2.Machine learning tree methods
## Feature engineering from volume
* Create a 1-day percent change in volume (use pct_change() from pandas), and assign it to the Adj_Volume_1d_change column in lng_df.
* Create a 5-day moving average of the 1-day percent change in Volume, and assign it to the Adj_Volume_1d_change_SMA column in lng_df.
* Plot histograms of these two new features we created using the new_features list.
```py
# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
new_features = ['Adj_Volume_1d_change', 'Adj_Volume_1d_change_SMA']
feature_names.extend(new_features)
lng_df['Adj_Volume_1d_change'] = lng_df['Adj_Volume'].pct_change()
lng_df['Adj_Volume_1d_change_SMA'] = talib.SMA(lng_df['Adj_Volume_1d_change'].values,
                        timeperiod=5)

# Plot histogram of volume % change data
lng_df[new_features].plot(kind='hist', sharex=False, bins=50)
plt.show()
```
## Create day-of-week features
* Use the dayofweek property from the lng_df index to get the days of the week.
* Use the get_dummies function on the days of the week variable, giving it a prefix of 'weekday'.
* Set the index of the days_of_week variable to be the same as the lng_df index so we can merge the two.
* Concatenate the lng_df and days_of_week DataFrames into one DataFrame.
```py
# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(lng_df.index.dayofweek,
                              prefix='weekday',
                              drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = lng_df.index

# Join the dataframe with the days of week dataframe
lng_df = pd.concat([lng_df, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend(['weekday_' + str(i) for i in range(1, 5)])
lng_df.dropna(inplace=True)  # drop missing values in-place
print(lng_df.head())
```
## Examine correlations of the new features
* Extend our new_features variable to contain the weekdays' column names, such as weekday_1, by concatenating the weekday number with the 'weekday_' string.
* Use Seaborn's heatmap to plot the correlations of new_features and the target,
5d_close_future_pct.
```py
# Add the weekday labels to the new_features list
new_features.extend(['weekday_' + str(i) for i in range(1, 5)])

# Plot the correlations between the new features and the targets
sns.heatmap(lng_df[new_features + ['5d_close_future_pct']].corr(), annot=True)
plt.yticks(rotation=0)  # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90)  # ensure x-axis ticklabels are vertical
plt.tight_layout()
plt.show()
```
## Fit a decision tree
* Use the imported class DecisionTreeRegressor with default arguments (i.e. no arguments) to create a decision tree model called decision_tree.
* Fit the model using train_features and train_targets which we've created earlier (and now contain day-of-week and volume features).
* Print the score on the training features and targets, as well as test_features and test_targets.
```py
from sklearn.tree import DecisionTreeRegressor

# Create a decision tree regression model with default arguments
decision_tree = DecisionTreeRegressor()

# Fit the model to the training features and targets
decision_tree.fit(train_features,train_targets)

# Check the score on train and test
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(test_features, test_targets))
```
## Try different max depths
* Loop through the values 3, 5, and 10 for use as the max_depth parameter in our decision tree model.
* Set the max_depth parameter in our DecisionTreeRegressor to be equal to d in each loop iteration.
* Print the model's score on the train_features and train_targets.
```py
# Loop through a few different max depths and check the performance
for d in [3,5,10]:
    # Create the tree and fit it
    decision_tree = DecisionTreeRegressor(max_depth=d)
    decision_tree.fit(train_features, train_targets)

    # Print out the scores on train and test
    print('max_depth=', str(d))
    print(decision_tree.score(train_features,train_targets))
    print(decision_tree.score(test_features, test_targets), '\n')
```
## Check our results
* Create a DecisionTreeRegressor model called decision_tree using 3 for the max_depth hyperparameter.
* Make predictions on the train and test sets (train_features and test_features) with our decision tree model.
* Scatter the train and test predictions vs the actual target values with plt.scatter(), and set the label argument equal to test for the test set.
```py
# Use the best max_depth of 3 from last exercise to fit a decision tree
decision_tree = DecisionTreeRegressor(max_depth=3)
decision_tree.fit(train_features, train_targets)

# Predict values for train and test
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

# Scatter the predictions vs actual values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.show()
```
## Fit a random forest
* Create the random forest model with the imported RandomForestRegressor class.
* Fit (train) the random forest using train_features and train_targets.
* Print out the R score on the train and test sets.
```py
from sklearn.ensemble import RandomForestRegressor

# Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)

# Look at the R^2 scores on train and test
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))
```
## Tune random forest hyperparameters
* Set the n_estimators hyperparameter to be a list with one value (200) in the grid dictionary.
* Set the max_features hyperparameter to be a list containing 4 and 8 in the grid dictionary.
* Fit the random forest regressor model (rfr, already created for you) to the train_features and train_targets with each combination of hyperparameters, g, in the loop.
* Calculate R by using rfr.score() on test_features and append the result to the test_scores list.
```py
from sklearn.model_selection import ParameterGrid

# Create a dictionary of hyperparameters to search
grid = {'n_estimators':[200], 'max_depth': [3], 'max_features': [4,8], 'random_state': [42]}
test_scores = []

# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])
```
## Evaluate performance
* Use the best number for max_features in our RandomForestRegressor (rfr) that we found in the previous exercise (it was 4).
* Make predictions using the model with the train_features and test_features.
* Scatter actual targets (train/test_targets) vs the predictions (train/test_predictions), and label the datasets train and test.
```py
# Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=42)
rfr.fit(train_features, train_targets)

# Make predictions with our model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
plt.show()
```
## Random forest feature importances
* Use the feature_importances_ property of our random forest model (rfr) to extract feature importances into the importances variable.
* Use numpy's argsort to get indices of the feature importances from greatest to least, and save the sorted indices in the sorted_index variable.
* Set xtick labels to be feature names in the labels variable, using the sorted_index list. feature_names must be converted to a numpy array so we can index it with the sorted_index list.
```py
# Get feature importances from our random forest model
importances = rfr.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels 
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()
```
## A gradient boosting model
* Create a GradientBoostingRegressor object with the hyperparameters that have already been set for you.
* Fit the gbr model to the train_features and train_targets.
* Print the scores for the training and test features and targets.
```py
from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=42)
gbr.fit(train_features,train_targets)

print(gbr.score(train_features, train_targets))
print(gbr.score(test_features, test_targets))
```
## Gradient boosting feature importances
* Reverse the sorted_index variable to go from greatest to least using python indexing.
* Create the sorted feature labels list as labels by converting feature_names to a numpy array and indexing with sorted_index.
* Create a bar plot of the xticks, and feature_importances indexed with the sorted_index variable, and labels as the xtick labels.
```py
# Extract feature importances from the fitted gradient boosting model
feature_importances = gbr.feature_importances_

# Get the indices of the largest to smallest feature importances
sorted_index = np.argsort(feature_importances)[::-1]
x = range(features.shape[1])

# Create tick labels 
labels = np.array(feature_names)[sorted_index]

plt.bar(x, feature_importances[sorted_index], tick_label=labels)

# Set the tick lables to be the feature names, according to the sorted feature_idx
plt.xticks(rotation=90)
plt.show()
```

# 3.Neural networks and KNN
## Standardizing data
* Remove day of week features from train/test features using .iloc (day of week are the last 4 features).
* Standardize train_features and test_features using sklearn's scale(); store scaled features as scaled_train_features and scaled_test_features.
* Plot a histogram of the 14-day RSI moving average (indexed at [:, 2]) from unscaled train_features on the first subplot (ax[0]]).
* Plot a histogram of the standardized 14-day RSI moving average on the second subplot (ax[1]).
```python
from sklearn.preprocessing import scale

# Remove unimportant features (weekdays)
train_features = train_features.iloc[:, :-4]
test_features = test_features.iloc[:,:-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()
```
## Optimize n_neighbors
* Loop through values of 2 to 12 for n and set this as n_neighbors in the knn model.
* Fit the model to the training data (scaled_train_features and train_targets).
* Print out the R values using the .score() method of the knn model for the train and test sets, and take note of the best score on the test set.
```python
from sklearn.neighbors import KNeighborsRegressor

for n in range(2,13):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)
    
    # Fit the model to the training data
    knn.fit(scaled_train_features, train_targets)
    
    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features, test_targets))
    print()  # prints a blank line
```
## Evaluate KNN performance
* Set n_neighbors in the KNeighborsRegressor to the best-performing value of 5 (found in the previous exercise).
* Obtain predictions using the knn model from the scaled_train_features and scaled_test_features.
* Create a scatter plot of the test_targets versus the test_predictions and label it test.
```py
# Create the model with the best-performing n_neighbors of 5
knn = KNeighborsRegressor(n_neighbors=5)

# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.legend()
plt.show()
```
## Build and fit a simple neural net
* Create a dense layer with 20 nodes and the ReLU ('relu') activation as the 2 layer in the neural network.
* Create the last dense layer with 1 node and a linear activation (activation='linear').
* Fit the model to the scaled_train_features and train_targets.
```py
Create a dense layer with 20 nodes and the ReLU ('relu') activation as the 2 layer in the neural network.
Create the last dense layer with 1 node and a linear activation (activation='linear').
Fit the model to the scaled_train_features and train_targets.
```
## Plot losses
* Plot the losses ('loss') from history.history.
* Set the title of the plot as the last loss from history.history, and round it to 6 digits.
```py
# Plot the losses from the fit
plt.plot(history.history['loss'])

# Use the last loss as the title
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()
```
## Measure performance
* Obtain predictions from model_1 on the scaled test set data (scaled_test_features and test_targets).
* Print the R score on the test set (test_targets and test_preds).
* Plot the test_preds versus test_targets in a scatter plot with plt.scatter().
```py
from sklearn.metrics import r2_score

# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds,test_targets,label='test')
plt.legend()
plt.show()
```
## Custom loss function
* Set the arguments of the sign_penalty() function to be y_true and y_pred.
* Multiply the squared error (tf.square(y_true - y_pred)) by penalty when the signs of y_true and y_pred are different.
* Return the average of the loss variable from the function -- this is the mean squared error (with our penalty for opposite signs of actual vs predictions).
```py
import keras.losses
import tensorflow as tf

# Create loss function
def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true * y_pred, 0), \
                     penalty* tf.square(y_true - y_pred), \
                     tf.square(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)

keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
print(keras.losses.sign_penalty)
```
## Fit neural net with custom loss function
* Set the input_dim of the first neural network layer to be the number of columns of scaled_train_features with the .shape[1] property.
* Use the custom sign_penalty loss function to .compile() our model_2.
* Plot the loss from the history of the fit. The loss is under history.history['loss'].
```py
# Create the model
model_2 = Sequential()
model_2.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_2.add(Dense(20, activation='relu'))
model_2.add(Dense(1, activation='linear'))

# Fit the model with our custom 'sign_penalty' loss function
model_2.compile(optimizer='adam', loss=sign_penalty)
history = model_2.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()
```
## Visualize the results
* Create predictions on the test set with .predict(), model_2, and scaled_test_features.
* Evaluate the R score on the test set predictions using test_preds and test_targets.
* Plot the test set targets vs actual values with plt.scatter(), and label it 'test'.
```py
# Evaluate R^2 scores
train_preds = model_2.predict(scaled_train_features)
test_preds = model_2.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets,label='test')  # plot test set
plt.legend(); plt.show()
```
## Combatting overfitting with dropout
* Add a dropout layer (Dropout()) after the first Dense layer in the model, and use 20% (0.2) as the dropout rate.
* Use the adam optimizer and the mse loss function when compiling the model in .compile().
* Fit the model to the scaled_train_features and train_targets using 25 epochs.
```py
from keras.layers import Dropout

# Create model with dropout
model_3 = Sequential()
model_3.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(20, activation='relu'))
model_3.add(Dense(1, activation='linear'))

# Fit model with mean squared error loss function
model_3.compile(optimizer='adam', loss='mse')
history = model_3.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()
```
## Ensembling models
* Create predictions on the scaled_train_features and scaled_test_features for the 3 models we fit (model_1, model_2, model_3) using the .predict() method.
* Horizontally stack (np.hstack() the predictions into a matrix, and take the row-wise averages to get average predictions for the train and test sets.
```py
# Make predictions from the 3 neural net models
train_pred1 = model_1.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2 = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3 = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds = np.mean(np.hstack((test_pred1,test_pred2,test_pred3)), axis=1)
print(test_preds[-5:])
```
## See how the ensemble performed
* Evaluate the R scores on the train and test sets. Use the sklearn r2_score() function (already imported for you) with train_targets and train_preds from the previous exercise.
* Plot the train and test predictions versus the actual values with plt.scatter().
```py
from sklearn.metrics import r2_score

# Evaluate the R^2 scores
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend(); plt.show()
```
# 4. Machine learning with modern portfolio theory
## Join stock DataFrames and calculate returns
* Join together lng_df, spy_df, and smlv_df using pd.concat() into the full_df DataFrame.
* Resample the full_df to Business Month Start ('BMS') frequency.
* Get the daily percent change of full_df with .pct_change().
```py
# Join 3 stock dataframes together
full_df = pd.concat([lng_df,spy_df,smlv_df], axis=1).dropna()

# Resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()

# Calculate daily returns of stocks
returns_daily = full_df.pct_change()

# Calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()
print(returns_monthly.tail())
```
## Calculate covariances for volatility
* Loop through the index of returns_monthly.
* Create a mask for returns_daily which uses the current month and year from returns_monthly, and matches this to the current month and year from i in the loop.
* Use the mask on returns_daily and calculate covariances using .cov().
```py
# Daily covariance of stocks (for each monthly period)
covariances = {}
rtd_idx = returns_daily.index
for i in returns_monthly.index:    
    # Mask daily returns for each month and year, and calculate covariance
    mask = (rtd_idx.month == i.month) & (rtd_idx.year == i.year)
    
    # Use the mask to get daily returns for the current month and year of monthy returns index
    covariances[i] = returns_daily[mask].cov()

print(covariances[i])
```
## Calculate portfolios
* Generate 3 random numbers for the weights using np.random.random().
* Calculate returns by taking the dot product (np.dot(); multiplies element-by-element and sums up two arrays) of weights with the monthly returns for the current date in the loop.
* Use the .setdefault() method to add an empty list ([]) to the portfolio_weights dictionary for the current date, then append weights to the list.
```py
portfolio_returns, portfolio_volatility, portfolio_weights = {}, {}, {}

# Get portfolio performances at each month
for date in sorted(covariances.keys()):
    cov = covariances[date]
    for portfolio in range(10):
        weights = np.random.random(3)
        weights /= np.sum(weights) # /= divides weights by their sum to normalize
        returns = np.dot(weights, returns_monthly.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        portfolio_returns.setdefault(date, []).append(returns)
        portfolio_volatility.setdefault(date, []).append(volatility)
        portfolio_weights.setdefault(date, []).append(weights)
        
print(portfolio_weights[date][0])
```
## Plot efficient frontier
* Get the latest date from the covariances dictionary -- remember the dates are the keys.
* Plot the volatility vs returns (portfolio_returns) for the latest date in a scatter plot, and set the alpha value for transparency to be 0.1.
```py
# Get latest date of available data
date = sorted(covariances.keys())[-1]  

# Plot efficient frontier
# warning: this can take at least 10s for the plot to execute...
plt.scatter(x=portfolio_volatility[date], y=portfolio_returns[date],  alpha=0.1)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()
```
## Get best Sharpe ratios
* Using enumerate(), enumerate the portfolio_returns for each date in the loop.
* For the current date in the loop, append to the sharpe_ratio dictionary entry with the return (ret) divided by portfolio_volatility for the current date and current i in the loops.
* Set the value for the current date's max_sharpe_idxs to be the index of the maximum Sharpe ratio using np.argmax().
```py
# Empty dictionaries for sharpe ratios and best sharpe indexes by date
sharpe_ratio, max_sharpe_idxs = {}, {}

# Loop through dates and get sharpe ratio for each portfolio
for date in portfolio_returns.keys():
    for i, ret in enumerate(portfolio_returns[date]):
    
        # Divide returns by the volatility for the date and index, i
        sharpe_ratio.setdefault(date, []).append(ret / portfolio_volatility[date][i])

    # Get the index of the best sharpe ratio for each date
    max_sharpe_idxs[date] = np.argmax(sharpe_ratio[date])

print(portfolio_returns[date][max_sharpe_idxs[date]])
```
## Calculate EWMAs
* Use a span of 30 to calculate the daily exponentially-weighted moving average (ewma_daily).
* Resample the daily ewma to the month by using the Business Monthly Start frequency (BMS) and the first day of the month (.first()).
* Shift ewma_monthly by one month forward, so we can use the previous month's EWMA as a feature to predict the next month's ideal portfolio.
```py
# Calculate exponentially-weighted moving average of daily returns
ewma_daily = returns_daily.ewm(span=30).mean()

# Resample daily returns to first business day of the month with the first day for that month
ewma_monthly = ewma_daily.resample('BMS').first()

# Shift ewma for the month by 1 month forward so we can use it as a feature for future predictions 
ewma_monthly = ewma_monthly.shift(1).dropna()

print(ewma_monthly.iloc[-1])
```
## Make features and targets
* Use the .iterrows() method with ewma_monthly to iterate through the index, value in the loop.
* Use the date in the loop and best_idx to index portfolio_weights to get the ideal portfolio weights based on the best Sharpe ratio.
* Append the ewma to the features.
```py
targets, features = [], []

# Create features from price history and targets as ideal portfolio
for date, ewma in ewma_monthly.iterrows():

    # Get the index of the best sharpe ratio
    best_idx = max_sharpe_idxs[date]
    targets.append(portfolio_weights[date][best_idx])
    features.append(ewma)  # add ewma to features

targets = np.array(targets)
features = np.array(features)
print(targets[-5:])
```
## Plot efficient frontier with best Sharpe ratio
* Set cur_volatility to be the portfolio volatilities for the latest date.
* Construct the "efficient frontier" plot by plotting volatility on the x-axis and returns on the y-axis.
* Get the best portfolio index for the latest date from max_sharpe_idxs.
```py
# Get most recent (current) returns and volatility
date = sorted(covariances.keys())[-1]
cur_returns = portfolio_returns[date]
cur_volatility = portfolio_volatility[date]

# Plot efficient frontier with sharpe as point
plt.scatter(x=cur_volatility, y=cur_returns, alpha=0.1, color='blue')
best_idx = max_sharpe_idxs[date]

# Place an orange "X" on the point with the best Sharpe ratio
plt.scatter(x=cur_volatility[best_idx], y=cur_returns[best_idx], marker='x', color='orange')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()
```
## Make predictions with a random forest
* Set the train_size to be 85% of the full training set data using the .shape property of features.
* Create train and test targets from targets using Python indexing.
* Fit the random forest model to the train_features and train_targets.
```py
# Make train and test features
train_size = int(0.85 * features.shape[0])
train_features = features[:train_size]
test_features = features[train_size:]
train_targets = targets[:train_size]
test_targets = targets[train_size:]

# Fit the model and check scores on train and test
rfr = RandomForestRegressor(n_estimators=300, random_state=42)
rfr.fit(train_features, train_targets)
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))
```
## Get predictions and first evaluation
* Use the rfr random forest model's .predict() method to make predictions on train_features and test_features.
* Multiply the test set portion of returns_monthly by test_predictions to get the returns of our test set predictions.
* Plot the test set returns_monthly for 'SPY' (everything from train_size to the end of the data).
```py
# Get predictions from model on train and test
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Calculate and plot returns from our RF predictions and the SPY returns
test_returns = np.sum(returns_monthly.iloc[train_size:] * test_predictions, axis=1)
plt.plot(test_returns, label='algo')
plt.plot(returns_monthly['SPY'].iloc[train_size:], label='SPY')
plt.legend()
plt.show()
```
## Evaluate returns
* Set the first list entries of both algo_cash and spy_cash to the same amount (cash).
* Multiply the cash in our test_returns loop by 1 + r in order to apply the returns to our cash.
* As with the test_returns loop, in the SPY performance loop, append cash to spy_cash after multiplying by 1 + r to add the returns to cash.
```py 
# Calculate the effect of our portfolio selection on a hypothetical $1k investment
cash = 1000
algo_cash, spy_cash = [cash], [cash]  # set equal starting cash amounts
for r in test_returns:
    cash *= 1 + r
    algo_cash.append(cash)

# Calculate performance for SPY
cash = 1000  # reset cash amount
for r in returns_monthly['SPY'].iloc[train_size:]:
    cash *= 1+r
    spy_cash.append(cash)

print('algo returns:', (algo_cash[-1] - algo_cash[0]) / algo_cash[0])
print('SPY returns:', (spy_cash[-1] - spy_cash[0]) / spy_cash[0])
```
## Plot returns
* Use plt.plot() to plot the algo_cash (with label 'algo') and spy_cash (with label 'SPY').
* Use plt.legend() to display the legend.
```py
# Plot the algo_cash and spy_cash to compare overall returns
plt.plot(algo_cash, label='algo')
plt.plot(spy_cash, label='SPY')
plt.legend()  # show the legend
plt.show()
```
*Finished by 2021/08/18*