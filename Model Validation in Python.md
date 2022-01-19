# Model Validation in Python

# 1. Basic Modeling in scikit-learn
Before we can validate models, we need an understanding of how to create and work with them. This chapter provides an introduction to running regression and classification models in scikit-learn. We will use this model building foundation throughout the remaining chapters.
## Seen vs. unseen data
* Using X_train and X_test as input data, create arrays of predictions using model.predict().
* Calculate model accuracy on both data the model has seen and data the model has not seen before.
* Use the print statements to print the seen and unseen data.
```py
# The model is fit using X_train and y_train
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))
```
## Set parameters and fit a model
* Add a parameter to rfr so that the number of trees built is 100 and the maximum depth of these trees is 6.
* Make sure the model is reproducible by adding a random state of 1111.
* Use the .fit() method to train the random forest regression model with X_train as the input data and y_train as the response.
```py
# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state =1111

# Fit the model
rfr.fit(X_train,y_train)
```
## Feature importances
* Loop through the feature importance output of rfr.
* Print the column names of X_train and the importance score for that column.
```py
# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
      # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))
```
## Classification predictions
* Create two arrays of predictions. One for the classification values and one for the predicted probabilities.
* Use the .value_counts() method for a pandas Series to print the number of observations that were assigned to each class.
* Print the first observation of probability_predictions to see how the probabilities are structured.
```py
# Fit the rfc model. 
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(probability_predictions[0]))
```
## Reusing model parameters
* Print out the characteristics of the model rfc by simply printing the model.
* Print just the random state of the model.
* Print the dictionary of model parameters.
```py
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))
```
## Random forest classifier
* Create rfc using the scikit-learn implementation of random forest classifiers and set a random state of 1111.
* Fit rfc using X_train for the training data and y_train for the responses.
* Predict the class values for X_test.
* Use the method .score() to print an accuracy metric for X_test given the actual values y_test
```py
from sklearn.ensemble import RandomForestClassifier

# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)

# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])

# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))
```


# 2. Validation Basics
This chapter focuses on the basics of model validation. From splitting data into training, validation, and testing datasets, to creating an understanding of the bias-variance tradeoff, we build the foundation for the techniques of K-Fold and Leave-One-Out validation practiced in chapter three.

## Create one holdout set
* Create the X dataset by creating dummy variables for all of the categorical columns.
* Split X and y into train (X_train, y_train) and test (X_test, y_test) datasets.
* Split the datasets using 10% for testing
```py
# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:,0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=.1, random_state=1111)
```

## Create two holdout sets
* Create temporary datasets and testing datasets (X_test, y_test). Use 20% of the overall data for the testing datasets.
* Using the temporary datasets (X_temp, y_temp), create training (X_train, y_train) and validation (X_val, y_val) datasets.
* Use 25% of the temporary data for the validation datasets.

```py
# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test  =\
    train_test_split(X, y, test_size=0.2, random_state=1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val =\
    train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)
```

## Mean absolute error
* Manually calculate the MAE using n as the number of observations predicted.
* Calculate the MAE using sklearn.
* Print off both accuracy values using the print statements.
```py
from sklearn.metrics import mean_absolute_error

# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-lean, the error is {}'.format(mae_two))
```
## Mean squared error
* Manually calculate the MSE.
* Calculate the MSE using sklearn.
```py
from sklearn.metrics import mean_squared_error

n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test,predictions)
print('Using scikit-lean, the error is {}'.format(mse_two))
```

## Performance on data subsets
* Create an array east_teams that can be used to filter labels to East conference teams.
* Create the arrays true_east and preds_east by filtering the arrays y_test and predictions.
* Use the print statements to print the MAE (using scikit-learn) for the East conference. The mean_absolute_error function has been loaded as mae.
* The variable west_error contains the MAE for the West teams. Use the print statement to print out the Western conference MAE.
```py
# Find the East conference teams
east_teams = labels == "E"

# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Print the accuracy metrics
print('The MAE for East teams is {}'.format(
    mae(true_east, preds_east)))

# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))
```

## Confusion matrices
* Use the confusion matrix to calculate overall accuracy.
* Use the confusion matrix to calculate precision and recall.
* Use the three print statements to print each accuracy value.
```py
# Calculate and print the accuracy
accuracy = (324 + 491) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (15 + 491)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (123 + 491)
print("The recall is {0: 0.2f}".format(recall))
```

## Confusion matrices, again
* Import sklearn's function for creating confusion matrices.
* Using the model rfc, create category predictions on the test set X_test.
* Create a confusion matrix using sklearn.
* Print the value from cm that represents the actual 1s that were predicted as 1s (true positives).
```py
from sklearn.metrics import confusion_matrix

# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1,1]))
```

## Precision vs. recall
* Import the precision or the recall metric for sklearn. Only one method is correct for the given context.
* Calculate the precision or recall using y_test for the true values and test_predictions for the predictions.
* Print the final score based on your selected metric.
```py
from sklearn.metrics import precision_score

test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))
```

## Error due to under/over-fitting
* Create a random forest model with 25 trees, a random state of 1111, and max_features of 2. Read the print statements.

```py
# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=2)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))
```
 <script.py> output:
    The training error is 3.88
    The testing error is 9.15
    
    
　　
* Set max_features to 11 (the number of columns in the dataset). Read the print statements.
```py
# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=11)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))
 ```
 <script.py> output:
    The training error is 3.57
    The testing error is 10.05
    
    
    
　　
* Set max_features equal to 4. Read the print statements.
```py
# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=4)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies 
print('The training error is {0:.2f}'.format(
  mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(
  mae(y_test, rfr.predict(X_test))))
 ```
<script.py> output:
    The training error is 3.60
    The testing error is 8.79
    
    
## Am I underfitting?
* For each loop, predict values for both the X_train and X_test datasets.
* For each loop, append the accuracy_score() of the y_train dataset and the corresponding predictions to train_scores.
* For each loop, append the accuracy_score() of the y_test dataset and the corresponding predictions to test_scores.
* Print the training and testing scores using the print statements.
```py
from sklearn.metrics import accuracy_score

test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test,test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))
```

# 3. Cross Validation
Holdout sets are a great start to model validation. However, using a single train and test set if often not enough. Cross-validation is considered the gold standard when it comes to validating model performance and is almost always used when tuning model hyper-parameters. This chapter focuses on performing cross-validation to validate model performance.

## Two samples
* Create samples sample1 and sample2 with 200 observations that could act as possible testing datasets.
* Use the list comprehension statement to find out how many observations these samples have in common.
* Use the Series.value_counts() method to print the values in both samples for column Class.
```py
# Create two different samples of 200 observations 
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations 
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples 
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())
```
## scikit-learn's KFold()
* Call the KFold() method to split data using five splits, shuffling, and a random state of 1111.
* Use the split() method of KFold on X.
* Print the number of indices in both the train and validation indices lists.
```py
from sklearn.model_selection import KFold

# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))
```
## Using KFold indices
* Use train_index and val_index to call the correct indices of X and y when creating training and validation data.
* Fit rfc using the training dataset
* Use rfc to create predictions for validation dataset and print the validation accuracy
```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))
```
## scikit-learn's methods
* Load the method for calculating the scores of cross-validation.
* Load the random forest regression method.
* Load the mean square error metric.
* Load the method for creating a scorer to use with cross-validation.
```py
# Instruction 1: Load the cross-validation method
from sklearn.model_selection import cross_val_score

# Instruction 2: Load the random forest regression model
from sklearn.ensemble import RandomForestRegressor

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer
from sklearn.metrics import mean_squared_error, make_scorer
```

## Implement cross_val_score()
* Fill in cross_val_score().
* Use X_train for the training data, and y_train for the response.
* Use rfc as the model, 10-fold cross-validation, and mse for the scoring function.
* Print the mean of the cv results.
```py
rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc,
                     X=X_train,
                     y=y_train,
                     cv=10,
                     scoring=mse)

# Print the mean error
print(cv.mean())
```
## Leave-one-out-cross-validation
* Create a scorer using mean_absolute_error for cross_val_score() to use.
* Fill out cross_val_score() so that the model rfr, the newly defined mae_scorer, and LOOCV are used.
* Print the mean and the standard deviation of scores using numpy (loaded as np).
```py
from sklearn.metrics import mean_absolute_error, make_scorer

# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=85, scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))
```

# 4. Selecting the best model with Hyperparameter tuning.
The first three chapters focused on model validation techniques. In chapter 4 we apply these techniques, specifically cross-validation, while learning about hyperparameter tuning. After all, model validation makes tuning possible and helps us select the overall best model.

## Creating Hyperparameters
* Print.get_params() in the console to review the possible parameters of the model that you can tune.
* Create a maximum depth list, [4, 8, 12] and a minimum samples list [2, 5, 10] that specify possible values for each hyperparameter.
* Create one final list to use for the maximum features
```py
# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features 
max_features = [4,6,8,10]
```
## Running a model using ranges
* Randomly select a max_depth, min_samples_split, and max_features using your range variables.
* Print out all of the parameters for rfr to see which values were randomly selected.
```py
from sklearn.ensemble import RandomForestRegressor

# Fill in rfr using your variables
rfr = RandomForestRegressor(
    n_estimators=100,
    max_depth=random.choice(max_depth),
    min_samples_split=random.choice(min_samples_split),
    max_features=random.choice(max_features))
    # random.choice => random choose one elements from list

# Print out the parameters
print(rfr.get_params())
```
## Preparing for RandomizedSearch
* Finalize the parameter dictionary by adding a list for the max_depth parameter with options 2, 4, 6, and 8.
* Create a random forest regression model with ten trees and a random_state of 1111.
* Create a mean squared error scorer to use.
```py
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error

# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2,4,6,8],
              "max_features": [2, 4, 6, 8, 10],
              "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)
```
## Implementing RandomizedSearchCV
* Load the method for conducting a random search in sklearn.
* Complete a random search by filling in the parameters: estimator, param_distributions, and scoring.
* Use 5-fold cross validation for this random search.
```py
# Import the method for random search
from sklearn.model_selection import RandomizedSearchCV

# Build a random search using param_dist, rfr, and scorer
random_search =\
    RandomizedSearchCV(
        estimator=rfr,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring=scorer)
```
## Selecting the best precision model
* Create a precision scorer, precision using make_scorer(<scoring_function>).
* Complete the random search method by using rfc and param_dist.
* Use rs.cv_results_ to print the mean test scores.
* Print the best overall score.
```py
from sklearn.metrics import precision_score, make_scorer

# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(
  estimator=rfc, param_distributions=param_dist,
  scoring = precision,
  cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))
```



















