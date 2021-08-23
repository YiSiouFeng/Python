# 1. Exploring and Preparing Loan Data
## Explore the credit data
* Print the structure of the cr_loan data.
* Look at the first five rows of the data.
```py
# Check the structure of the data
print(cr_loan.dtypes)

# Check the first five rows of the data
print(cr_loan.head())
```
* Plot a histogram of loan_amnt within the data.
```py
# Look at the distribution of loan amounts with a histogram
n, bins, patches = plt.hist(x=cr_loan['loan_amnt'], bins='auto', color='blue',alpha=0.7, rwidth=0.85)
plt.xlabel("Loan Amount")
plt.show()
```
* Create a scatter plot of a person's income and age. In this case, income is the independent variable and age is the dependent variable.
```py
print("There are 32 000 rows of data so the scatter plot may take a little while to plot.")

# Plot a scatter plot of income against age
plt.scatter(cr_loan['person_income'], cr_loan['person_age'],c='blue', alpha=0.5)
plt.xlabel('Personal Income')
plt.ylabel('Persone Age')
plt.show()
```
## Crosstab and pivot tables
* Create a cross table of loan_intent and loan_status.
```py
# Create a cross table of the loan intent and loan status
print(pd.crosstab(cr_loan['loan_intent'], cr_loan['loan_status'], margins = True))
```
* Create a cross table of home ownership grouped by loan_status and loan_grade.
```py
# Create a cross table of home ownership, loan status, and grade
print(pd.crosstab(cr_loan['person_home_ownership'],[cr_loan['loan_status'],cr_loan['loan_grade']]))
```
* Create a cross table of home ownership, loan status, and average loan_percent_income.
```py
# Create a cross table of home ownership, loan status, and average percent income
print(pd.crosstab(cr_loan['person_home_ownership'], cr_loan['loan_status'],
              values=cr_loan['loan_percent_income'], aggfunc='mean'))
```
* Create a boxplot of the loan's percent of the person's income grouped by loan_status.
```py
# Create a box plot of percentage income by loan status
cr_loan.boxplot(column = ['loan_percent_income'], by = 'loan_status')
plt.title('Average Percent Income by Loan Status')
plt.suptitle('')
plt.show()
```
## Finding outliers with cross tables
* Print the cross table of loan_status and person_home_ownership with the max person_emp_length
* Create and array of indices for records with an employment length greater than 60. Store it as indices.
* Drop the records from the data using the array indices and create a new dataframe called cr_loan_new.
* Print the cross table from earlier, but instead use both min and max
```py
# Create the cross table for loan status, home ownership, and the max employment length
print(pd.crosstab(cr_loan['loan_status'],cr_loan['person_home_ownership'],
                  values=cr_loan['person_emp_length'], aggfunc='max'))

# Create an array of indices where employment length is greater than 60
indices = cr_loan[cr_loan['person_emp_length'] > 60].index

# Drop the records from the data based on the indices and create a new dataframe
cr_loan_new = cr_loan.drop(indices)

# Create the cross table from earlier and include minimum employment length
print(pd.crosstab(cr_loan_new['loan_status'],cr_loan_new['person_home_ownership'],
            values=cr_loan_new['person_emp_length'], aggfunc=['min','max']))
```
## Visualizing credit outliers
* Create a scatter plot of person age on the x-axis and loan_amnt on the y-axis.
```py
# Create the scatter plot for age and amount
plt.scatter(cr_loan['person_age'], cr_loan['loan_amnt'], c='blue', alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()
```
* Use the .drop() method from Pandas to remove the outliers and create cr_loan_new.
* Create a scatter plot of age on the x-axis and loan interest rate on the y-axis with a label for loan_status.
```py
# Use Pandas to drop the record from the data frame and create a new one
cr_loan_new = cr_loan.drop(cr_loan[cr_loan['person_age'] > 100].index)

# Create a scatter plot of age and interest rate
colors = ["blue","red"]
plt.scatter(cr_loan_new['person_age'], cr_loan_new['loan_int_rate'],
            c = cr_loan_new['loan_status'],
            cmap = matplotlib.colors.ListedColormap(colors),
            alpha=0.5)
plt.xlabel("Person Age")
plt.ylabel("Loan Interest Rate")
plt.show()
```
## Replacing missing credit data
* Print an array of column names that contain missing data using .isnull().
* Print the top five rows of the data set that has missing data for person_emp_length.
* Replace the missing data with the median of all the employment length using .fillna().
* Create a histogram of the person_emp_length column to check the distribution.
```py
# Print a null value column array
print(cr_loan.columns[cr_loan.isnull().any()])

# Print the top five rows with nulls for employment length
print(cr_loan[cr_loan['person_emp_length'].isnull()].head())

# Impute the null values with the median value for all employment lengths
cr_loan['person_emp_length'].fillna((cr_loan['person_emp_length'].median()), inplace=True)

# Create a histogram of employment length
n, bins, patches = plt.hist(cr_loan['person_emp_length'], bins='auto', color='blue')
plt.xlabel("Person Employment Length")
plt.show()
```
## Removing missing data
* Print the number of records that contain missing data for interest rate.
* Create an array of indices for rows that contain missing interest rate called indices.
* Drop the records with missing interest rate data and save the results to cr_loan_clean.
```py
# Print the number of nulls
print(cr_loan['loan_int_rate'].isnull().sum())

# Store the array on indices
indices = cr_loan[cr_loan['loan_int_rate'].isnull()].index

# Save the new data without missing data
cr_loan_clean = cr_loan.drop(indices)
```
## Missing data intuition
* Replace the data with the value Other.

# 2. Logistic Regression for Defaults
## Logistic regression basics
* Create the X and y sets using the loan_int_rate and loan_status columns.
Create and fit a logistic regression model to the training data and call it clf_logistic_single.
Print the parameters of the model with .get_params().
Check the intercept of the model with the .intercept_ attribute.
```py
# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate']]
y = cr_loan_clean[['loan_status']]

# Create and fit a logistic regression model
clf_logistic_single = LogisticRegression()
clf_logistic_single.fit(X, np.ravel(y))

# Print the parameters of the model
print(clf_logistic_single.get_params())

# Print the intercept of the model
print(clf_logistic_single.intercept_)
```
## Multivariate logistic regression
* Create a new X data set with loan_int_rate and person_emp_length. Store it as X_multi.
* Create a y data set with just loan_status.
* Create and .fit() a LogisticRegression() model on the new X data. Store it as clf_logistic_multi.
* Print the .intercept_ value of the model
```py
# Create X data for the model
X_multi = cr_loan_clean[['loan_int_rate','person_emp_length']]


# Create a set of y data for training
y = cr_loan_clean[['loan_status']]

# Create and train a new logistic regression
clf_logistic_multi = LogisticRegression(solver='lbfgs').fit(X_multi, np.ravel(y))

# Print the intercept of the model
print(clf_logistic_multi.intercept_)
```
## Creating training and test sets
* Create the data set X using interest rate, employment length, and income. Create the y set using loan status.
* Use train_test_split() to create the training and test sets from X and y.
* Create and train a LogisticRegression() model and store it as clf_logistic.
* Print the coefficients of the model using .coef_.
```py
# Create the X and y data sets
X = cr_loan_clean[['loan_int_rate','person_emp_length','person_income']]
y = cr_loan_clean[['loan_status']]

# Use test_train_split to create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=123)

# Create and fit the logistic regression model
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Print the models coefficients
print(clf_logistic.coef_)
```
## Changing coefficients
* Check the first five rows of both X training sets.
* Train a logistic regression model, called clf_logistic1, with the X1 training set.
* Train a logistic regression model, called clf_logistic2, with the X2 training set.
* Print the coefficients for both logistic regression models.
```py
# Print the first five rows of each training set
print(X1_train.head())
print(X2_train.head())

# Create and train a model on the first training data
clf_logistic1 = LogisticRegression(solver='lbfgs').fit(X1_train, np.ravel(y_train))

# Create and train a model on the second training data
clf_logistic2 = LogisticRegression(solver='lbfgs').fit(X2_train, np.ravel(y_train))


# Print the coefficients of each model
print(clf_logistic1.coef_)
print(clf_logistic2.coef_)
```
## One-hot encoding credit data
* Create a data set for all the numeric columns called cred_num and one for the non-numeric columns called cred_str.
* Use one-hot encoding on cred_str to create a new data set called cred_str_onehot.
* Union cred_num with the new one-hot encoded data and store the results as cr_loan_prep.
* Print the columns of the new data set.
```py
# Create two data sets for numeric and non-numeric data
cred_num = cr_loan_clean.select_dtypes(exclude=['object'])
cred_str = cr_loan_clean.select_dtypes(include=['object'])

# One-hot encode the non-numeric columns
cred_str_onehot = pd.get_dummies(cred_str)

# Union the one-hot encoded columns to the numeric ones
cr_loan_prep = pd.concat([cred_num, cred_str_onehot], axis=1)

# Print the columns in the new data set
print(cr_loan_prep.columns)
```
## Predicting probability of default
* Train a logistic regression model on the training data and store it as clf_logistic.
* Use predict_proba() on the test data to create the predictions and store them in preds.
* Create two data frames, preds_df and true_df, to store the first five predictions and true loan_status values.
* Print the true_df and preds_df as one set using .concat().
```py
# Train the logistic regression model on the training data
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(y_train))

# Create predictions of probability for loan status using test data
preds = clf_logistic.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))
```
## Default classification reporting
* Create a data frame of just the probabilities of default from preds called preds_df.
* Reassign loan_status values based on a threshold of 0.50 for probability of default in preds_df.
* Print the value counts of the number of rows for each loan_status.
* Print the classification report using y_test and preds_df
```py
# Create a dataframe for the probabilities of default
preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])

# Reassign loan status based on the threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the row counts for each loan status
print(preds_df['loan_status'].value_counts())

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
```
## Selecting report metrics
* Print the classification report for y_test and predicted loan status.
```py
# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df['loan_status'], target_names=target_names))
```
* Print all the non-average values using precision_recall_fscore_support().
```py
# Print all the non-average values from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status']))
```
* Print all the non-average values using precision_recall_fscore_support().
```py
# Print the first two numbers from the report
print(precision_recall_fscore_support(y_test,preds_df['loan_status'])[2])
```
## Visually scoring credit models
* Create a set of predictions for probability of default and store them in preds.
* Print the accuracy score the model on the X and y test sets.
* Use roc_curve() on the test data and probabilities of default to create fallout and sensitivity Then, create a ROC curve plot with fallout on the x-axis.
* Compute the AUC of the model using test data and probabilities of default and store it in auc.
```py
# Create predictions and store them in a variable
preds = clf_logistic.predict_proba(X_test)

# Print the accuracy score the model
print(clf_logistic.score(X_test,y_test))

# Plot the ROC curve of the probabilities of default
prob_default = preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.show()

# Compute the AUC and store it in a variable
auc = roc_auc_score(y_test, prob_default)
```
## Thresholds and confusion matrices
* Reassign values of loan_status using a threshold of 0.5 for probability of default within preds_df.
* Print the confusion matrix of the y_test data and the new loan status values.
```py
# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))
```
* Reassign values of loan_status using a threshold of 0.5 for probability of default within preds_df.
* Print the confusion matrix of the y_test data and the new loan status values.
```py
# Set the threshold for defaults to 0.5
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.5 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))
```
* Reassign values of loan_status using a threshold of 0.4 for probability of default within preds_df.
* Print the confusion matrix of the y_test data and the new loan status values.
```py
# Set the threshold for defaults to 0.4
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Print the confusion matrix
print(confusion_matrix(y_test,preds_df['loan_status']))
```
* Q: Based on the confusion matrices you just created, calculate the default recall for each. Using these values, answer the following: which threshold gives us the highest value for default recall?
* A: 0.4

## How thresholds affect performance
* Reassign the loan_status values using the threshold 0.4.
* Store the number of defaults in preds_df by selecting the second value from the value counts and store it as num_defaults.
* Get the default recall rate from the classification matrix and store it as default_recall
* Estimate the unexpected loss from the new default recall by multiplying 1 - default_recall by the average loan amount and number of default loans.
```py
# Reassign the values of loan status based on the new threshold
preds_df['loan_status'] = preds_df['prob_default'].apply(lambda x: 1 if x > 0.4 else 0)

# Store the number of loan defaults from the prediction data
num_defaults = preds_df['loan_status'].value_counts()[1]

# Store the default recall from the classification report
default_recall = precision_recall_fscore_support(y_test,preds_df['loan_status'])[1][1]

# Calculate the estimated impact of the new default recall rate
print( avg_loan_amnt* num_defaults * (1 - default_recall))
```
## Threshold selection
* Plot the graph of thresh for the x-axis then def_recalls, non-default recall values, and accuracy scores on each y-axis.
```py
plt.plot(thresh,def_recalls)
plt.plot(thresh,nondef_recalls)
plt.plot(thresh,accs)
plt.xlabel("Probability Threshold")
plt.xticks(ticks)
plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
plt.show()
```
* Question
Have a closer look at this plot. In fact, expand the window to get a really good look. Think about the threshold values from thresh and how they affect each of these three metrics. Approximately what starting threshold value would maximize these scores evenly?
* A: 0.275
* 

# 3. Gradient Boosted Trees Using XGBoost
## Trees for defaults
* Create and train a gradient boosted tree using XGBClassifier() and name it clf_gbt.
Predict probabilities of default on the test data and store the results in gbt_preds.
* Create two data frames, preds_df and true_df, to store the first five predictions and true loan_status values.
* Concatenate and print the data frames true_df and preds_df in order, and check the model's results.
```py
# Train a model
import xgboost as xgb
clf_gbt = xgb.XGBClassifier().fit(X_train, np.ravel(y_train))

# Predict with a model
gbt_preds = clf_gbt.predict_proba(X_test)

# Create dataframes of first five predictions, and first five true labels
preds_df = pd.DataFrame(gbt_preds[:,1][0:5], columns = ['prob_default'])
true_df = y_test.head()

# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), preds_df], axis = 1))
```
## Gradient boosted portfolio performance
* Print the first five rows of portfolio.
* Create the expected_loss column for the gbt and lr model named gbt_expected_loss and lr_expected_loss.
* Print the sum of lr_expected_loss for the entire portfolio.
* Print the sum of gbt_expected_loss for the entire portfolio.
```py
# Print the first five rows of the portfolio data frame
print(portfolio.head())

# Create expected loss columns for each model using the formula
portfolio['gbt_expected_loss'] = portfolio['gbt_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']
portfolio['lr_expected_loss'] = portfolio['lr_prob_default'] * portfolio['lgd'] * portfolio['loan_amnt']

# Print the sum of the expected loss for lr
print('LR expected loss: ', np.sum(portfolio['lr_expected_loss']))

# Print the sum of the expected loss for gbt
print('GBT expected loss: ', np.sum(portfolio['gbt_expected_loss']))
```
## Assessing gradient boosted trees
* Predict the loan_status values for the X test data and store them in gbt_preds.
* Check the contents of gbt_preds to see predicted loan_status values not probabilities of default.
* Print a classification_report() of the model's performance against y_test.
```py
# Predict the labels for loan status
gbt_preds = clf_gbt.predict(X_test)

# Check the values created by the predict method
print(gbt_preds)

# Print the classification report of the model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))
```
## Column importance and default prediction
* Create and train a XGBClassifier() model on the X_train and y_train training sets and store it as clf_gbt.
* Print the column importances for the columns in clf_gbt by using .get_booster() and .get_score().
```py
# Create and train the model on the training data
clf_gbt = xgb.XGBClassifier().fit(X_train,np.ravel(y_train))

# Print the column importances from the model
print(clf_gbt.get_booster().get_score(importance_type = 'weight'))
```
## Visualizing column importance
* Create and train a XGBClassifier() model on X2_train and call it clf_gbt2.
* Plot the column importances for the columns that clf_gbt2 trained on.
```py
# Train a model on the X data with 2 columns
clf_gbt2 = xgb.XGBClassifier().fit(X2_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt2, importance_type = 'weight')
plt.show()
```
* Create and train another XGBClassifier() model on X3_train and call it clf_gbt3.
* Plot the column importances for the columns that clf_gbt3 trained on.
```py
# Train a model on the X data with 3 columns
clf_gbt3 = xgb.XGBClassifier().fit(X3_train,np.ravel(y_train))

# Plot the column importance for this model
xgb.plot_importance(clf_gbt3, importance_type = 'weight')
plt.show()
```
## Column selection and model performance
* Use both gbt and gbt2 to predict loan_status and store the values in gbt_preds and gbt2_preds.
* Print the classification_report() of the first model.
* Print the classification_report() of the second model.
```py
# Predict the loan_status using each model
gbt_preds = gbt.predict(X_test)
gbt2_preds = gbt2.predict(X2_test)

# Print the classification report of the first model
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))

# Print the classification report of the second model
print(classification_report(y_test,gbt2_preds, target_names=target_names))
```
* Have a look at the classification_report() for both models. What is the highest F-1 score for predicting defaults?
* 0.72

## Cross validating credit models
* Set the number of folds to 5 and the stopping to 10. Store them as n_folds and early_stopping.
* Create the matrix object DTrain using the training data.
* Use cv() on the parameters, folds, and early stopping objects. Store the results as cv_df.
* Print the contents of cv_df.
```py
# Set the values for number of folds and stopping iterations
n_folds = 5
early_stopping = 10

# Create the DTrain matrix for XGBoost
DTrain = xgb.DMatrix(X_train, label = y_train)

# Create the data frame of cross validations
cv_df = xgb.cv(params, DTrain, num_boost_round = 5, nfold=n_folds,
            early_stopping_rounds=early_stopping)

# Print the cross validations data frame
print(cv_df)
```
## Limits to cross-validation testing
* Print the first five rows of the CV results data frame.
* Print the average of the test set AUC from the CV results data frame rounded to two places.
* Plot a line plot of the test set AUC over the course of each iteration.
```py
# Print the first five rows of the CV results data frame
print(cv_results_big.head())

# Calculate the mean of the test AUC scores
print(np.mean(cv_results_big['test-auc-mean']).round(2))

# Plot the test AUC scores for each iteration
plt.plot(cv_results_big['test-auc-mean'])
plt.title('Test AUC Score Over 600 Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Test AUC Score')
plt.show()
```
## Cross-validation scoring
* Create a gradient boosted tree with a learning rate of 0.1 and a max depth of 7. Store the model as gbt.
* Calculate the cross validation scores against the X_train and y_train data sets with 4 folds. Store the results as cv_scores.
* Print the cross validation scores.
* Print the average accuracy score and standard deviation with formatting.
```py
# Create a gradient boosted tree model using two hyperparameters
gbt = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 7)

# Calculate the cross validation scores for 4 folds
cv_scores = cross_val_score(gbt, X_train, np.ravel(y_train), cv = 4)

# Print the cross validation scores
print(cv_scores)

# Print the average accuracy and standard deviation of the scores
print("Average accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(),
                                              cv_scores.std() * 2))
```
## Undersampling training data
* Create data sets of non-defaults and defaults stored as nondefaults and defaults.
* Sample the nondefaults to the same number as count_default and store it as nondefaults_under.
* Concatenate nondefaults and defaults using .concat() and store it as X_y_train_under.
* Print the .value_counts() of loan status for the new data set.
```py
# Create data sets for defaults and non-defaults
nondefaults = X_y_train[X_y_train['loan_status'] == 0]
defaults = X_y_train[X_y_train['loan_status'] == 1]

# Undersample the non-defaults
nondefaults_under = nondefaults.sample(count_default)

# Concatenate the undersampled nondefaults with defaults
X_y_train_under = pd.concat([nondefaults_under.reset_index(drop = True),defaults.reset_index(drop = True)], axis = 0)

# Print the value counts for loan status
print(X_y_train_under['loan_status'].value_counts())
```
## Undersampled tree performance
* Print the classification_report() for both the old model and new model.
```py
# Check the classification reports
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, gbt_preds, target_names=target_names))
print(classification_report(y_test, gbt2_preds, target_names=target_names))
```
* Print a confusion_matrix() of the old and new model predictions.
```py
# Print the confusion matrix for both old and new models
print(confusion_matrix(y_test,gbt_preds))
print(confusion_matrix(y_test,gbt2_preds))
```
* Print the roc_auc_score of the new model and old model.
```py
# Print and compare the AUC scores of the old and new models
print(roc_auc_score(y_test, gbt_preds))
print(roc_auc_score(y_test, gbt2_preds))
```
## Undersampling intuition
* Intuition check again! Now you've seen the effects of undersampling the training set to improve default prediction. You undersampled the training data set X_train, and it had a positive impact on the new model's AUC score and recall for defaults. The training data had class imbalance which is normal for most credit loan data.
You did not undersample the test data X_test. Why not undersample the test set as well?
* You should not undersample the test set because it will make the test set unrealistic.

# 4. Model Evaluation and Implementation
## Comparing model reports
* Print the classification_report() for the logistic regression predictions.
* Print the classification_report() for the gradient boosted tree predictions.
* Print the macro average of the F-1 Score for the logistic regression using precision_recall_fscore_support().
* Print the macro average of the F-1 Score for the gradient boosted tree using precision_recall_fscore_support().
```py
# Print the logistic regression classification report
target_names = ['Non-Default', 'Default']
print(classification_report(y_test, preds_df_lr['loan_status'], target_names=target_names))

# Print the gradient boosted tree classification report
print(classification_report(y_test, preds_df_gbt['loan_status'], target_names=target_names))

# Print the default F-1 scores for the logistic regression
print(precision_recall_fscore_support(y_test,preds_df_lr['loan_status'], average = 'macro')[2])

# Print the default F-1 scores for the gradient boosted tree
print(precision_recall_fscore_support(y_test,preds_df_gbt['loan_status'], average = 'macro')[2])
```
## Comparing with ROCs
* Calculate the fallout, sensitivity, and thresholds for the logistic regression and gradient boosted tree.
* Plot the ROC chart for the lr then gbt using the fallout on the x-axis and sensitivity on the y-axis for each model.
```py
# ROC chart components
fallout_lr, sensitivity_lr, thresholds_lr = roc_curve(y_test, clf_logistic_preds)
fallout_gbt, sensitivity_gbt, thresholds_gbt = roc_curve(y_test,clf_gbt_preds)

# ROC Chart with both
plt.plot(fallout_lr, sensitivity_lr, color = 'blue', label='%s' % 'Logistic Regression')
plt.plot(fallout_gbt,sensitivity_gbt, color = 'green', label='%s' % 'GBT')
plt.plot([0, 1], [0, 1], linestyle='--', label='%s' % 'Random Prediction')
plt.title("ROC Chart for LR and GBT on the Probability of Default")
plt.xlabel('Fall-out')
plt.ylabel('Sensitivity')
plt.legend()
plt.show()
```
* Print the AUC for the logistic regression.
* Print the AUC for the gradient boosted tree.
```py
# Print the logistic regression AUC with formatting
print("Logistic Regression AUC Score: %0.2f" % roc_auc_score(y_test, clf_logistic_preds))

# Print the gradient boosted tree AUC with formatting
print("Gradient Boosted Tree AUC Score: %0.2f" % roc_auc_score(y_test,clf_gbt_preds))
```
## Calibration curves
* Create a calibration curve plot() by starting with the perfect calibration guideline and label it 'Perfectly calibrated'. Then add the labels for the y-axis and x-axis in order.
```py
# Create the calibration curve plot with the guideline
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.xlabel('Fraction of positives')
plt.ylabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()
```
* Add a plot of the mean predicted values on the x-axis and fraction of positives on the y-axis for the logistic regression model to the plot of the guideline. Label this 'Logistic Regression'.
```py
# Add the calibration curve for the logistic regression to the plot
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_lr,frac_of_pos_lr,'s-', label='%s' % 'Logistic Regression')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()
```
* Finally, add a plot of the mean predicted values on the x-axis and fraction of positives on the y-axis for the gradient boosted tree to the plot. Label this 'Gradient Boosted tree'.
```py
# Add the calibration curve for the gradient boosted tree
plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')    
plt.plot(mean_pred_val_lr, frac_of_pos_lr,'s-', label='%s' % 'Logistic Regression')
plt.plot(mean_pred_val_gbt, frac_of_pos_gbt,'s-', label='%s' % 'Gradient Boosted tree')
plt.ylabel('Fraction of positives')
plt.xlabel('Average Predicted Probability')
plt.legend()
plt.title('Calibration Curve')
plt.show()
```
## Credit acceptance rates
* Print the summary statistics of prob_default within the data frame of predictions using .describe().
* Calculate the threshold for a 85% acceptance rate using quantile() and store it as threshold_85.
* Create a new column called pred_loan_status based on threshold_85.
* Print the value counts of the new values in pred_loan_status.
```py
# Check the statistics of the probabilities of default
print(test_pred_df['prob_default'].describe())

# Calculate the threshold for a 85% acceptance rate
threshold_85 = np.quantile(test_pred_df['prob_default'], 0.85)

# Apply acceptance rate threshold
test_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > threshold_85 else 0)

# Print the counts of loan status after the threshold
print(test_pred_df['pred_loan_status'].value_counts())
```
## Visualizing quantiles of acceptance
* Create a histogram of the predicted probabilities clf_gbt_preds.
* Calculate the threshold for an acceptance rate of 85% using quantile(). Store this value as threshold.
* Plot the histogram again, except this time add a reference line using .axvline().
```py
# Plot the predicted probabilities of default
plt.hist(clf_gbt_preds, color = 'blue', bins = 40)

# Calculate the threshold with quantile
threshold = np.quantile(clf_gbt_preds, 0.85)

# Add a reference line to the plot for the threshold
plt.axvline(x = threshold, color = 'red')
plt.show()
```
## Bad rates
* Print the first five rows of the predictions data frame.
* Create a subset called accepted_loans which only contains loans where the predicted loan status is 0.
* Calculate the bad rate based on true_loan_status of the subset using sum() and .count().
```py
# Print the top 5 rows of the new data frame
print(test_pred_df.head())

# Create a subset of only accepted loans
accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]

# Calculate the bad rate
print(np.sum(accepted_loans['true_loan_status']) / accepted_loans['true_loan_status'].count())
```
## Acceptance rate impact
* Print the summary statistics of the loan_amnt column using .describe().
* Calculate the average value of loan_amnt and store it as avg_loan.
* Set the formatting for pandas to '${:,.2f}'
* Print the cross table of the true loan status and predicted loan status multiplying each by avg_loan.
```py
# Print the statistics of the loan amount column
print(test_pred_df['loan_amnt'].describe())

# Store the average loan amount
avg_loan = np.mean(test_pred_df['loan_amnt'])

# Set the formatting for currency, and print the cross tab
pd.options.display.float_format = '${:,.2f}'.format

print(pd.crosstab(test_pred_df['true_loan_status'],test_pred_df['pred_loan_status_15']).apply(lambda x: x * avg_loan, axis = 0))
```
## Making the strategy table
* Print the contents of accept_rates.
```py
# Print accept rates
print(accept_rates)
```
* Populate the arrays thresholds and bad_rates using a for loop. Calculate the threshold thresh, and store it in thresholds. Then reassign the loan_status values using thresh. After that, Create accepted_loans where loan_status is 0.
```py
# Populate the arrays for the strategy table with a for loop
for rate in accept_rates:
    # Calculate the threshold for the acceptance rate
    thresh = np.quantile(preds_df_gbt['prob_default'], rate).round(3)
    # Add the threshold value to the list of thresholds
    thresholds.append(np.quantile(preds_df_gbt['prob_default'], rate).round(3))
    # Reassign the loan_status value using the threshold
    test_pred_df['pred_loan_status'] = test_pred_df['prob_default'].apply(lambda x: 1 if x > thresh else 0)
    # Create a set of accepted loans using this acceptance rate
    accepted_loans = test_pred_df[test_pred_df['pred_loan_status'] == 0]
    # Calculate and append the bad rate using the acceptance rate
    bad_rates.append(np.sum((accepted_loans['true_loan_status']) / len(accepted_loans['true_loan_status'])).round(3))
```
* Create the strategy table as a data frame and call it strat_df.
* Print the contents of strat_df.
```py
# Create a data frame of the strategy table
strat_df = pd.DataFrame(zip(accept_rates, thresholds, bad_rates),
                        columns = ['Acceptance Rate','Threshold','Bad Rate'])

# Print the entire table
print(strat_df)
```
## Visualizing the strategy
* Create a simple boxplot of the values within strat_df using the pandas boxplot method.
```py
# Visualize the distributions in the strategy table with a boxplot
strat_df.boxplot()
plt.show()
```
* Create a line plot of the acceptance rates on the x-axis and bad rates on the y-axis with a title(), xlabel(), and ylabel().
```py
# Plot the strategy curve
plt.plot(strat_df['Acceptance Rate'], strat_df['Bad Rate'])
plt.title('Acceptance Rate')
plt.ylabel('Bad Rate')
plt.xlabel('Acceptance and Bad Rates')
plt.axes().yaxis.grid()
plt.axes().xaxis.grid()
plt.show()
```
## Estimated value profiling
* Check the contents of the new strat_df by printing the entire data frame.
```py
# Print the contents of the strategy df
print(strat_df)
```
* Create a line plot of the acceptance rate on the x-axis and estimated value from strat_df on the y-axis with axis labels for both x and y.
```py
# Create a line plot of estimated value
plt.plot(strat_df['Acceptance Rate'],strat_df['Estimated Value'])
plt.title('Estimated Value by Acceptance Rate')
plt.xlabel('Acceptance Rate')
plt.ylabel('Estimated Value')
plt.axes().yaxis.grid()
plt.show()
```
* Print the row with the highest 'Estimated Value' from strat_df.
```py
# Print the row with the max estimated value
print(strat_df.loc[strat_df['Estimated Value'] == np.max(strat_df['Estimated Value'])])
```
## Total expected loss
* Print the top five rows of test_pred_df.
* Create a new column expected_loss for each loan by using the formula above.
* Calculate the total expected loss of the entire portfolio, rounded to two decimal places, and store it as tot_exp_loss.
* Print the total expected loss.
```py
# Print the first five rows of the data frame
print(test_pred_df.head())

# Calculate the bank's expected loss and assign it to a new column
test_pred_df['expected_loss'] = test_pred_df['prob_default'] * test_pred_df['loss_given_default'] * test_pred_df['loan_amnt']

# Calculate the total expected loss to two decimal places
tot_exp_loss= round(np.sum(test_pred_df['expected_loss']),2)

# Print the total expected loss
print('Total expected loss: ', '${:,.2f}'.format(tot_exp_loss))
```
*Finished by 2021/08/24*