# 1. Data Pre-processing and Visualization
## The hunt for missing values
* Print out the features of loan_data along with the number of missing values.
```py
# Import modules
import numpy as np
import pandas as pd

# Print missing values
print(loan_data.isna().sum())
```
* Drop the rows with missing values and print the percentage of rows remaining.
```py
# Drop rows with missing values
dropNArows = loan_data.dropna(axis=0)

# Print percentage of rows remaining
print(dropNArows.shape[0]/loan_data.shape[0] * 100)
```
* Drop the columns with missing values and print the percentage of columns remaining.
```py
# Drop columns with missing values
dropNAcols = loan_data.dropna(axis=1)

# Print percentage of columns remaining
print(dropNAcols.shape[1]/loan_data.shape[1] * 100)
```
* Impute loan_data's missing values with 0 into loan_data_filled
* Compare 'Credit Score' using .describe() before imputation using loan_data and after using loan_data_filled.
```py
# Fill missing values with zero
loan_data_filled = loan_data.fillna(0)

# Examine 'Credit Score' before
print(loan_data['Credit Score'].describe())

# Examine 'Credit Score' after
print(loan_data_filled['Credit Score'].describe())
```
## Simple imputation
* Subset loan_data's numeric columns and assign them to numeric_cols.
* Instantiate a simple imputation object with a mean imputation strategy.
* Fit and transform the data.
* Convert the returned array back to a DataFrame.
* Print the imputed DataFrame's information using the .info() function to check for missing values.
```py
# Import imputer module
from sklearn.impute import SimpleImputer

# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Impute with mean
imp_mean = SimpleImputer(strategy='mean')
loans_imp_mean = imp_mean.fit_transform(numeric_cols)

# Convert returned array to DataFrame
loans_imp_meanDF = pd.DataFrame(loans_imp_mean, columns=numeric_cols.columns)

# Check the DataFrame's info
print(loans_imp_meanDF.info())
```
## Iterative imputation
* Subset loan_data's numeric columns and assign them to numeric_cols.
* Instantiate an iterative imputation object with 5 iterations and posterior sampling enabled.
* Fit and transform the data.
* Convert return array object back to DataFrame.
* Print the imputed DataFrame's information using the .info() function to check for missing values.
```py
# Subset numeric features: numeric_cols
numeric_cols = loan_data.select_dtypes(include=[np.number])

# Iteratively impute
imp_iter = IterativeImputer(max_iter=5, sample_posterior=True, random_state=123)
loans_imp_iter = imp_iter.fit_transform(numeric_cols)

# Convert returned array to DataFrame
loans_imp_iterDF = pd.DataFrame(loans_imp_iter, columns=numeric_cols.columns)

# Check the DataFrame's info
print(loans_imp_iterDF.info())
```
## Training vs test set distributions and transformations
* If the distribution of test data (not yet seen by the model) is significantly different than the distribution of the training data, what problems can occur? What transformations can be applied to data before passing them to an ML model and why should these transformations be performed?
```py
If the data used to train a model is only slightly different than test or future data, it won't matter and you can go ahead with model tuning.
press
```
## Train/test distributions
* Subset loan_data to only the Credit Score and Annual Income features, and the target variable Loan Status in that order.
* Create an 80/20 split of loan_data and assign it to loan_data_subset.
* Create pairplots of trainingSet and testSet (in that order) setting the hue argument to the target variable Loan Status.
```py
# Create `loan_data` subset: loan_data_subset
loan_data_subset = loan_data[['Credit Score','Annual Income','Loan Status']]

# Create train and test sets
trainingSet, testSet = train_test_split(loan_data_subset, test_size=0.2, random_state=123)

# Examine pairplots
plt.figure()
sns.pairplot(trainingSet, hue='Loan Status', palette='RdBu')
plt.show()

plt.figure()
sns.pairplot(testSet, hue='Loan Status', palette='RdBu')
plt.show()
```
## Log and power transformations
* Subset loan_data for 'Years of Credit History' and plot its distribution and kernel density estimation (kde) using distplot().
```py
# Subset loan_data
cr_yrs = loan_data['Years of Credit History']

# Histogram and kernel density estimate
plt.figure()
sns.distplot(cr_yrs)
plt.show()
```
* Apply a log transformation using the Box-Cox transformation to cr_yrs and plot its distribution and kde.
```py
# Subset loan_data
cr_yrs = loan_data['Years of Credit History']

# Box-Cox transformation
cr_yrs_log = boxcox(cr_yrs, lmbda=0)

# Histogram and kernel density estimate
plt.figure()
sns.distplot(cr_yrs_log)
plt.show()
```
* Transform 'Years of Credit History' using the Box-Cox square-root argument and plot its distribution and kde.
```py
# Subset loan_data
cr_yrs = loan_data['Years of Credit History']

# Box-Cox transformation
cr_yrs_log = boxcox(cr_yrs, lmbda=0)

# Histogram and kernel density estimate
plt.figure()
sns.distplot(cr_yrs_log)
plt.show()
```
## Outlier detection
* Create a univariate boxplot using the feature Annual Income from loan_data.
* Create a multivariate boxplot conditioned on Loan Status using the feature Annual Income from loan_data.
```py
# Import modules
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate and multivariate boxplots
fig, ax =plt.subplots(1,2)
sns.boxplot(y=loan_data['Annual Income'], ax=ax[0])
sns.boxplot(x='Loan Status', y='Annual Income', data=loan_data, ax=ax[1])
plt.show()
```
* Create a univariate boxplot using the feature Monthly Debt from loan_data.
* Create a multivariate boxplot conditioned on Loan Status using the feature Monthly Debt from loan_data.
```py
# Import modules
import matplotlib.pyplot as plt
import seaborn as sns

# Multivariate boxplot
fig, ax =plt.subplots(1,2)
sns.boxplot(y=loan_data['Monthly Debt'], ax=ax[0])
sns.boxplot(x='Loan Status', y='Monthly Debt', data=loan_data, ax=ax[1])
plt.show()
```
* Create a univariate boxplot using the feature Years of Credit History from loan_data.
* Create a multivariate boxplot conditioned on Loan Status using the feature Years of Credit History from loan_data.
```py
# Import modules
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate and multivariate boxplots
fig, ax =plt.subplots(1,2)
sns.boxplot(x=loan_data['Years of Credit History'], ax=ax[0])
sns.boxplot(x='Loan Status', y='Years of Credit History', data=loan_data, ax=ax[1])
plt.show()
```
## Handling outliers
* Create an index of rows to keep for absolute z-scores less than 3 on the numeric columns and use it to index and concatenate subsets.
```py
# Print: before dropping
print(numeric_cols.mean())
print(numeric_cols.median())
print(numeric_cols.max())

# Create index of rows to keep
idx = (np.abs(stats.zscore(numeric_cols)) < 3).all(axis=1)

# Concatenate numeric and categoric subsets
ld_out_drop = pd.concat([numeric_cols.loc[idx], categoric_cols.loc[idx]], axis=1)

# Print: after dropping
print(ld_out_drop.mean())
print(ld_out_drop.median())
print(ld_out_drop.max())
```
* Winsorize 'Monthly Debt' with 5% upper and lower limits and print the mean, median and max before and after.
```py
# Print: before winsorize
print((loan_data['Monthly Debt']).mean())
print((loan_data['Monthly Debt']).median())
print((loan_data['Monthly Debt']).max())

# Winsorize numeric columns
debt_win = mstats.winsorize(loan_data['Monthly Debt'], limits=[0.05, 0.05])

# Convert to DataFrame, reassign column name
debt_out = pd.DataFrame(debt_win, columns=['Monthly Debt'])

# Print: after winsorize
print(debt_out.mean())
print(debt_out.median())
print(debt_out.max())
```
* Find the median of the values of Monthly Debt that are lower than 2120 and replace outliers with it.
```py
# Print: before replace with median
print((loan_data['Monthly Debt']).mean())
print((loan_data['Monthly Debt']).median())
print((loan_data['Monthly Debt']).max())

# Find median
median = loan_data.loc[loan_data['Monthly Debt'] < 2120, 'Monthly Debt'].median()
loan_data['Monthly Debt'] = np.where(loan_data['Monthly Debt'] > 2120, median, loan_data['Monthly Debt'])

print((loan_data['Monthly Debt']).mean())
print((loan_data['Monthly Debt']).median())
print((loan_data['Monthly Debt']).max())
```
## Z-score standardization
* Create a subset of the numeric and categorical columns in loan_data.
* Instantiate a standard scaler object and assign it to scaler.
* Fit and transform the relevant columns with a call to the appropriate method, then convert the returned object back to a DataFrame.
* Concatenate the categorical and scaled numeric columns.
```py
# Subset features
numeric_cols = loan_data.select_dtypes(include=[np.number])
categoric_cols = loan_data.select_dtypes(include=[object])

# Instantiate
scaler = StandardScaler()

# Fit and transform, convert to DF
numeric_cols_scaled = scaler.fit_transform(numeric_cols)
numeric_cols_scaledDF = pd.DataFrame(numeric_cols_scaled, columns=numeric_cols.columns)

# Concatenate categoric columns to scaled numeric columns
final_DF = pd.concat([categoric_cols, numeric_cols_scaledDF],  axis=1)
print(final_DF.head())
```
# 2. Supervised Learning
## Best feature subset
* How do you select the optimal subset of independent variables in a regression model?
As a reminder, filter methods rank features based on their statistical performance while wrapper, embedded and tree-based methods use a machine learning model to evaluate performance.
Select the statement that is true:
* Embedded methods, such as Lasso and Ridge Regression, are regularization methods that extract features that contribute the most during a given iteration and provide the best subset dependent on the penalty parameter.

## Filter and wrapper methods
* Create correlation matrix with diabetes and a heatmap, then subset the features which have greater than 50% correlation.
```py
# Create correlation matrix and print it
cor = diabetes.corr()
print(cor)

# Correlation matrix heatmap
plt.figure()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor["progression"])

# Selecting highly correlated features
best_features = cor_target[cor_target > 0.5]
print(best_features)
```
* Instantiate a linear kernel SVR estimator and a feature selector with 5 cross-validations, fit to features and target.
```py
# Import modules
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV

# Instantiate estimator and feature selector
svr_mod = SVR(kernel="linear")
feat_selector = RFECV(svr_mod, cv=5)

# Fit
feat_selector = feat_selector.fit(X,y )

# Print support and ranking
print(feat_selector.support_)
print(feat_selector.ranking_)
print(X.columns)
* Drop the unimportant column found in step 2 from X and instantiate a LarsCV object and fit it to your data.
```
* Drop the unimportant column found in step 2 from X and instantiate a LarsCV object and fit it to your data.
```py
# Import modules
from sklearn.linear_model import LarsCV

# Drop feature suggested not important in step 2
X = X.drop('sex', axis=1)

# Instantiate
lars_mod = LarsCV(cv=5, normalize=False)

# Fit
feat_selector = lars_mod.fit(X, y)

# Print r-squared score and estimated alpha
print(lars_mod.score(X, y))
print(lars_mod.alpha_)
```
## Feature selection through feature importance
* Import the correct function to instantiate a Random Forest regression model.
* Fit the model and print feature importance.
```py
# Import
from sklearn.ensemble import RandomForestRegressor

# Instantiate
rf_mod = RandomForestRegressor(max_depth=2, random_state=123, 
              n_estimators=100, oob_score=True)

# Fit
rf_mod.fit(X, y)

# Print
print(diabetes.columns)
print(rf_mod.feature_importances_)
```
* Import the correct function to instantiate an Extra Tree regression model.
* Fit the model and print feature importance.
```py
# Import
from sklearn.ensemble import ExtraTreesRegressor

# Instantiate
xt_mod = ExtraTreesRegressor()

# Fit
xt_mod.fit(X,y)

# Print
print(diabetes.columns)
print(xt_mod.feature_importances_)
```
## Avoiding overfitting
* Perform ElasticNet regression which uses l1-ratio regularization which is a combination of L1 and L2.

## Lasso regularization
* Import the functions needed for regular and cross-validated Lasso Regression, as well as mean squared error.
* Split your data into training and testing data with 30% test size.
* Instantiate a cross-validated lasso regression model setting 10-fold cross-validation and 10000 iterations, then fit it to your training data.
* Instantiate a lasso estimator passing the best alpha value from lasso_cv.
* Fit the model and print the mean squared error of your predictions.
```py
# Import modules
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.3)

# Instantiate cross-validated lasso, fit
lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000)
lasso_cv.fit(X_train, y_train)

# Instantiate lasso, fit, predict and print MSE
lasso = Lasso(alpha = lasso_cv.alpha_)
lasso.fit(X_train, y_train)
print(mean_squared_error(y_true=y_test, y_pred=lasso.predict(X_test)))
```
## Ridge regularization
* Import the functions needed for regular and cross-validated Ridge Regression, as well as mean squared error.
* Split your data into training and testing data with 30% test size.
* Instantiate a cross-validated ridge regression model object setting alphas to a list of 13 log scale values from -6 to 6 using np.logspace().
* Fit it to your training data.
* Instantiate a ridge estimator passing the best alpha value from ridge_cv.
* Fit the model and print the mean squared error of your predictions.
```py
# Import modules
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.3)

# Instantiate cross-validated ridge, fit
ridge_cv = RidgeCV(alphas=np.logspace(-6, 6, 13))
ridge_cv.fit(X_train, y_train)

# Instantiate ridge, fit, predict and print MSE
ridge = Ridge(alpha = ridge_cv.alpha_)
ridge.fit(X_train,y_train)
print(mean_squared_error(y_true=y_test, y_pred=ridge.predict(X_test)))
```
## Classification model features
* You want to build a classification model that predicts whether a bank loan application should be either approved or denied. What features should be used to train a simple classifier?
Select the answer that is true:
* Additional features should be created with feature engineering and they should be used along with the original features that have the highest correlation with the target variable.

## Logistic regression baseline classifier
* Fit and predict a Logistic Regression on loan_data with the target variable Loan Status as y and evaluate the trained model's accuracy score.
```py
# Create X matrix and y array
X = loan_data.drop("Loan Status", axis=1)
y = loan_data["Loan Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Instantiate
logistic = LogisticRegression()

# Fit
logistic.fit(X_train,y_train)

# Predict and print accuracy
print(accuracy_score(y_true=y_test, y_pred=logistic.predict(X_test)))
```
* Convert Annual Income to monthly, and derive the ratio of Monthly Debt to monthly_income and store it in dti_ratio
* Convert the target variable to numerical values and replace categorical features with dummy values.
```py
# Create dti_ratio variable
monthly_income = loan_data["Annual Income"]/12
loan_data["dti_ratio"] = loan_data["Monthly Debt"]/monthly_income * 100
loan_data = loan_data.drop(["Monthly Debt","Annual Income"], axis=1)

# Replace target variable levels
loan_data["Loan Status"] = loan_data["Loan Status"].replace({'Fully Paid': 0, 'Charged Off': 1})

# One-hot encode categorical variables
loan_data = pd.get_dummies(data=loan_data)

# Print
print(loan_data.head())
```
* Fit and predict a Logistic Regression on loans_dti and evaluate the trained model's accuracy score.
```py
# Create X matrix and y array
X = loans_dti.drop("Loan Status", axis=1)
y = loans_dti["Loan Status"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Instantiate
logistic_dti = LogisticRegression()

# Fit
logistic_dti.fit(X_train,y_train)

# Predict and print accuracy
print(accuracy_score(y_true=y_test, y_pred=logistic_dti.predict(X_test)))
```
## Bootstrap aggregation (bagging)
* Instantiate a bagging classifier by calling the appropriate function as introduced in the video exercise and set the appropriate argument for 50 estimators.
* Fit the model object to the data.
* Create predictions using the test set.
* Evaluate the model fit.
```py
# Instantiate bootstrap aggregation model
bagged_model = BaggingClassifier(n_estimators=50, random_state=123)

# Fit
bagged_model.fit(X_train, y_train)

# Predict
bagged_pred = bagged_model.predict(X_test)

# Print accuracy score
print(accuracy_score(y_test, bagged_pred))
```Boosting
* Instantiate an AdaBoost boosting classifier and set the appropriate argument to generate 50 estimators.
* Fit the data.
* Create predictions using the test set.
* Evaluate the model fit.
```py
# Boosting model
boosted_model = AdaBoostClassifier(n_estimators=50, random_state=123)

# Fit
boosted_model_fit = boosted_model.fit(X_train, y_train)

# Predict
boosted_pred = boosted_model_fit.predict(X_test)

# Print model accuracy
print(accuracy_score(y_test, boosted_pred))
```
## XG Boost
* Instantiate an XGBoost boosting classifier and set the appropriate argument to generate 10 estimators.
* Fit the data.
* Create predictions using the test data.
* Evaluate the model fit.
```py
# Instantiate
xgb = XGBClassifier(random_state=123, learning_rate=0.1, n_estimators=10, max_depth=3)

# Fit
xgb = xgb.fit(X_train, y_train)

# Predict
xgb_pred = xgb.predict(X_test)

# Print accuracy score
print('Final prediction score: [%.8f]' % accuracy_score(y_test,xgb_pred))
```
# 3. Unsupervised Learning
## The curse of dimensionality
* How do PCA and SVD algorithms help overcome the curse of dimensionality?
* SVD decomposes the original data matrix into three matrices and returns singular values.
## Principal component analysis
* Import the relevant module to perform PCA.
* Create a feature matrix X and target array y with progression from the diabetes dataset.
* Instantiate a principal component analysis object to perform linear dimensionality reduction that returns 3 components.
* Print the ratio of variance explained.
```py
# Import module
from sklearn.decomposition import PCA

# Feature matrix and target array
X = diabetes.drop('progression', axis=1)
y = diabetes['progression']

# PCA
pca = PCA(n_components=3)

# Fit and transform
principalComponents = pca.fit_transform(X)

# Print ratio of variance explained
print(pca.explained_variance_ratio_)
```
## Singular value decomposition
* Import the relevant module to perform SVD.
* Create a feature matrix X and target array y with progression from the diabetes dataset.
* Fit and transform the feature matrix.
* Print the ratio of variance explained.
```py
# Import module
from sklearn.decomposition import TruncatedSVD

# Feature matrix and target array
X = diabetes.drop('progression', axis=1)
y = diabetes['progression']

# SVD
svd = TruncatedSVD(n_components=3)

# Fit and transform
principalComponents = svd.fit_transform(X)

# Print ratio of variance explained
print(svd.explained_variance_ratio_)
```
## Reducing high-dimensional data
* In the video lesson, you saw how to use PCA and T-SNE to reduce dimensionality and visualize the new dimensions.
* Select the answer that is false:
```py
A dataset with 30 features can be reduced to 5 principal components using either PCA or t-sne and visualized.
```

## Visualization separation of classes with PCA I
* Assign the target variable values to the list targets.
* Pass the lists just created to the zip() function inside the for loop.
* Pass the instances where Loan Status is equal to target to indicesToKeep.
* Pass the appropriate object as the rows argument to loan_data_PCA.loc[] which keeps the data points on the x-axis equal to the target.
* Pass the appropriate object as the column argument to loan_data_PCA.loc[]which keeps the data points on the y-axis equal to PC2.
* Add a legend to the plot to label the different classes.
```py
targets = [0, 1]
colors = ['r', 'b']

# For loop to create plot
for target, color in zip(targets,colors):
    indicesToKeep = loan_data_PCA['Loan Status'] == target
    ax.scatter(loan_data_PCA.loc[indicesToKeep, 'principal component 1']
               , loan_data_PCA.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)

# Legend    
ax.legend(targets)
ax.grid()
plt.show()
```
## Visualization PCs with a scree plot
* Create a data matrix X, removing the target variable.
Instantiate, fit and transform a PCA object that returns 10 PCs.
```py
# Remove target variable
X = loan_data.drop('Loan Status', axis=1)

# Instantiate
pca = PCA(n_components=10)

# Fit and transform
principalComponents = pca.fit_transform(X)
```
* Create a DataFrame mapping Variance Explained to the explained variance ratio.
* Create a scree plot from pca_df setting your PCs on the x-axis and explained variance on the y-axis.
```py 
# List principal components names
principal_components = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']

# Create a DataFrame
pca_df = pd.DataFrame({'Variance Explained': pca.explained_variance_ratio_,
             'PC':principal_components})

# Plot DataFrame
sns.barplot(x='PC',y='Variance Explained', 
           data=pca_df, color="c")
plt.show()
```
* Instantiate, fit and transform a PCA object not setting n_components.
* Print the variance explained ratio.
```py
# Instantiate, fit and transform
pca2 = PCA()
principalComponents2 = pca2.fit_transform(X)

# Assign variance explained
var = pca2.explained_variance_ratio_
```
* Assign the cumulative sum of the explained ratios from the previous step to cumulative_var.
* Plot the results.
```py
# Plot cumulative variance
cumulative_var = np.cumsum(var)*100
plt.plot(cumulative_var,'k-o',markerfacecolor='None',markeredgecolor='k')
plt.title('Principal Component Analysis',fontsize=12)
plt.xlabel("Principal Component",fontsize=12)
plt.ylabel("Cumulative Proportion of Variance Explained",fontsize=12)
plt.show()
```
## Clustering algorithms
* What's the best way to determine which clustering algorithm should be used for a given dataset?
* Select the answer that is false:
```py
Comparing different distance metrics between different clustering algorithms is the best way to select which algorithm to use.
```
## K-means clustering
* Create a feature matrix X by dropping the target variable progression and fit the data to the instantiated k-means object.
```py
# Import module
from sklearn.cluster import KMeans

# Create feature matrix
X = diabetes.drop("progression", axis=1)

# Instantiate
kmeans = KMeans(n_clusters=2, random_state=123)

# Fit
fit = kmeans.fit(X)

# Print inertia
print("Sum of squared distances for 2 clusters is", kmeans.inertia_)
```
* Instantiate a 5 cluster k-means and print its inertia.
```py
# Import module
from sklearn.cluster import KMeans

# Create feature matrix
X = diabetes.drop("progression", axis=1)

# Instantiate
kmeans = KMeans(n_clusters=5, random_state=123)

# Fit
fit = kmeans.fit(X)

# Print inertia
print("Sum of squared distances for 5 clusters is", kmeans.inertia_)
```
* Fit the feature matrix to a 10-cluster k-means and print its inertia.
```py
# Import module
from sklearn.cluster import KMeans

# Create feature matrix
X = diabetes.drop("progression", axis=1)

# Instantiate
kmeans = KMeans(n_clusters=10, random_state=123)

# Fit
fit = kmeans.fit(X)

# Print inertia
print("Sum of squared distances for 10 clusters is", kmeans.inertia_)
```
* Fit the feature matrix to a 20-cluster k-means and print its inertia.
```py
# Import module
from sklearn.cluster import KMeans

# Create feature matrix
X = diabetes.drop("progression", axis=1)

# Instantiate
kmeans = KMeans(n_clusters=20, random_state=123)

# Fit
fit = kmeans.fit(X)

# Print inertia
print("Sum of squared distances for 20 clusters is", kmeans.inertia_)
```
## Hierarchical agglomerative clustering
* Import the relevant packages to create a dendrogram and perform agglomerative hierarchical clustering.
* Create a dendrogram on X using the ward linkage method.
* Instantiate an agglomerative clustering cluster cluster object, then fit it to the data matrix X.
* Print the number of clusters automatically chosen through agglomerative clustering.
```py
# Import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Create dendrogram
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()

# Create clusters and fit
hc = AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward')
hc.fit(X)

# Print number of clusters
print(hc.n_clusters_)
```
## What is the optimal k?
* How do you decide on the optimal number of clusters in a partitive clustering algorithm such as K-Means?
* Once a K-means model has been trained on several values for k, both the elbow method and silhouette methods can be used to decide on the best value for the number of clusters.

## Silhouette method
* Import the necessary modules to instantiate a K-means algorithm and get its silhouette score.
* Create the target matrix X by dropping the target variable progression.
* Instantiate, fit and predict a K-Means object for each number of clusters ranging from 2 to 8.
* Print the silhouette score for each iteration of clustering.
```py
# Import modules
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Feature matrix
X = diabetes.drop("progression", axis=1)

# For loop
for n_clusters in range(2,9):
    kmeans = KMeans(n_clusters=n_clusters)
    # Fit and predict your k-Means object
    preds = kmeans.fit_predict(X)
    score = silhouette_score(X, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
```
## Elbow method
* Create an empty list named sum_of_squared_distances
* Instantiate and fit a K-Means object for each number of clusters ranging from 1 to 14.
* Append the inertia score for each iteration of K-Means to sum_of_squared_distances.
* Create a plot with the number of clusters on the x-axis, and the sum of squared distances on the y-axis.
```py
# Create empty list
sum_of_squared_distances = []

# Create for loop
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    sum_of_squared_distances.append(kmeans.inertia_)

# Plot
plt.plot(range(1,15), sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()
```
# 4. Model Selection and Evaluation
## Validating model performance
* How do you ensure your model will perform well against test (unseen) data?
* Create a cross-validated Decision Tree model, then evaluate its performance on the test set.

## Decision tree
* Import the correct function for a decision tree classifier and split the data into train and test sets.
* Instantiate a decision tree classifier, fit, predict, and print accuracy.
```py
# Import modules
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=123)

# Instantiate, Fit, Predict
loans_clf = DecisionTreeClassifier() 
loans_clf.fit(X_train,y_train)
y_pred = loans_clf.predict(X_test)

# Evaluation metric
print("Decision Tree Accuracy: {}".format(accuracy_score(y_test,y_pred)))
```
* Import the correct function to perform cross-validated grid search.
* Instantiate a decision tree classifier and use it with the parameter grid to perform a cross-validated grid-search.
* Fit and print model evaluation metrics
```py
# Import modules
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
param_grid = {"criterion": ["gini"], "min_samples_split": [2, 10, 20], 
              "max_depth": [None, 2, 5, 10]}

# Instantiate classifier and GridSearchCV, fit
loans_clf = DecisionTreeClassifier()
dtree_cv = GridSearchCV(loans_clf, param_grid, cv=5)
fit = dtree_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Decision Tree Parameter: {}".format(dtree_cv.best_params_))
print("Tuned Decision Tree Accuracy: {}".format(dtree_cv.best_score_))
```
## A forest of decision trees
* Import the correct function for a random forest classifier and split the data into train and test sets.
* Instantiate a random forest classifier, fit, predict, and print accuracy.
```py
# Import modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=123)

# Instantiate, Fit, Predict
loans_rf = RandomForestClassifier() 
loans_rf.fit(X_train,y_train)
y_pred = loans_rf.predict(X_test)

# Evaluation metric
print("Random Forest Accuracy: {}".format(accuracy_score(y_test,y_pred)))
```
* Import the correct function to perform cross-validated grid search.
* Perform the same steps, this time while performing cross-validated grid-search.
```py
# Import modules
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
param_grid = {"criterion": ["gini"], "min_samples_split": [2, 10, 20], 
              "max_depth": [None, 2, 5, 10],"max_features": [10, 20, 30]}

# Instantiate classifier and GridSearchCV, fit
loans_rf = RandomForestClassifier()
rf_cv = GridSearchCV(loans_rf, param_grid, cv=5)
fit = rf_cv.fit(X_train,y_train)

# Print the optimal parameters and best score
print("Tuned Random Forest Parameter: {}".format(rf_cv.best_params_))
print("Tuned Random Forest Accuracy: {}".format(rf_cv.best_score_))
```
## X-ray weapon detection
* When a dataset has imbalanced classes, a low precision score indicates a high number of false positives, so consider trying different classification algorithms and/or resampling techniques to improve precision.
* When a dataset has imbalanced classes, a low precision score indicates a high number of false positives, so consider trying different classification algorithms and/or resampling techniques to improve precision.
## Imbalanced class metrics
* Import the necessary modules to create a logistic regression model as well as confusion matrix, accuracy, precision, recall, and F1-scores.
* Instantiate a logistic regression object, fit and predict.
* Print the confusion matrix and accuracy score.
* Print the precision, recall, and F1-scores.
```py
# Import
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Instantiate, fit, predict
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Print evaluation metrics
print("Confusion matrix:\n {}".format(confusion_matrix(y_test, y_pred)))
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("Precision: {}".format(precision_score(y_test,y_pred)))
print("Recall: {}".format(recall_score(y_test,y_pred)))
print("F1: {}".format(f1_score(y_test,y_pred)))
```
## Resampling techniques
* Create an upsampled minority class the length of the majority class and concatenate (done for you).
Create a downsampled majority class the length of the minority class and concatenate (done for you).
```py
# Upsample minority and combine with majority
loans_upsampled = resample(deny, replace=True, n_samples=len(approve), random_state=123)
upsampled = pd.concat([approve, loans_upsampled])

# Downsample majority and combine with minority
loans_downsampled = resample(approve, replace = False,  n_samples = len(deny), random_state = 123)
downsampled = pd.concat([loans_downsampled, deny])
```
* Create an upsampled feature matrix and target array.
* Instantiate a logistic regression model object, fit, and predict with X_test.
* Print the evaluation metrics.
```py
# Upsampled feature matrix and target array
X_train_up = upsampled.drop('Loan Status', axis=1)
y_train_up = upsampled['Loan Status']

# Instantiate, fit, predict
loan_lr_up = LogisticRegression(solver='liblinear')
loan_lr_up.fit(X_train_up,y_train_up)
upsampled_y_pred = loan_lr_up.predict(X_test)

# Print evaluation metrics
print("Confusion matrix:\n {}".format(confusion_matrix(y_test, upsampled_y_pred)))
print("Accuracy: {}".format(accuracy_score(y_test,upsampled_y_pred)))
print("Precision: {}".format(precision_score(y_test,upsampled_y_pred)))
print("Recall: {}".format(recall_score(y_test,upsampled_y_pred)))
print("F1: {}".format(f1_score(y_test,upsampled_y_pred)))
```
* Create a downsampled feature matrix and target array.
* Instantiate a logistic regression model object, fit, and predict with X_test.
* Print the evaluation metrics.
```py
# Downsampled feature matrix and target array
X_train_down = downsampled.drop('Loan Status', axis=1)
y_train_down = downsampled['Loan Status']

# Instantiate, fit, predict
loan_lr_down = LogisticRegression(solver='liblinear')
loan_lr_down.fit(X_train_down,y_train_down)
downsampled_y_pred = loan_lr_down.predict(X_test)

# Print evaluation metrics
print("Confusion matrix:\n {}".format(confusion_matrix(y_test, downsampled_y_pred)))
print("Accuracy: {}".format(accuracy_score(y_test, downsampled_y_pred)))
print("Precision: {}".format(precision_score(y_test, downsampled_y_pred)))
print("Recall: {}".format(recall_score(y_test, downsampled_y_pred)))
print("F1: {}".format(f1_score(y_test, downsampled_y_pred)))
```
## Addressing multicollinearity
* After careful exploratory data analysis, you realize that your baseline regression model suffers from multicollinearity. How would you check if that is true or not? Without losing any information, can you build a better baseline model?
Select the answer that is false:
* Create a correlation matrix and/or heatmap, then remove the multicollinear independent variables.
press

## Multicollinearity techniques - feature engineering
* Instantiate, fit, and predict a Linear Regression.
Print the model coefficients, MSE, and r-squared.
```py
# Instantiate, fit, predict
lin_mod = LinearRegression()
lin_mod.fit(X_train,y_train)
y_pred = lin_mod.predict(X_test)

# Coefficient estimates
print('Coefficients: \n', lin_mod.coef_)

# Mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test,y_pred))

# Explained variance score
print('R_squared score: %.2f' % r2_score(y_test, y_pred))
```
* Create a correlation matrix, plot it to a heatmap.
* Print the matrix to explore the independent variable relationships.
```py
# Correlation matrix
diab_corr = diabetes.corr()

# Generate correlation heatmap
ax = sns.heatmap(diab_corr, center=0, cmap=sns.diverging_palette(20,220, n=256), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

# Print correlations
print(diab_corr)
```
* Engineer a new feature by combining s1 and s2 from diabetes, then remove them.
* Split your data into training and testing data with 30% test size and print the column names.
```py
# Feature engineering
diabetes['s1_s2'] = diabetes['s1'] * diabetes['s2']
diabetes = diabetes.drop(['s1','s2'], axis=1)

# Print variable names
print(diabetes.columns)

# Train/test split
X2 = diabetes.drop('progression', axis=1)
y2 = diabetes['progression']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=123)
```
* Instantiate, fit, and predict a Linear Regression.
* Print the model coefficients, MSE, and r-squared.
```py
# Instantiate, fit, predict
lin_mod2 = LinearRegression()
lin_mod2.fit(X_train2,y_train2)
y_pred2 = lin_mod2.predict(X_test2)

# Coefficient estimates
print('Coefficients: \n', lin_mod2.coef_)

# Mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test2, y_pred2))

# Explained variance score
print('R_squared score: %.2f' % r2_score(y_test2,y_pred2))
```
## Multicollinearity techniques - PCA
* Import the necessary modules to perform PCA.
* Instantiate and fit.
* Transform train and test separately.
```py
# Import
from sklearn.decomposition import PCA

# Instantiate
pca = PCA()

# Fit on train
pca.fit(X_train)

# Transform train and test
X_trainPCA = pca.transform(X_train)
X_testPCA = pca.transform(X_test)
```
* Instantiate, fit, and predict a Linear Regression to PCA transformed dataset.
* Print the model coefficients, MSE, and r-squared.
```py
# Import
from sklearn.linear_model import LinearRegression

# Instantiate, fit, predict
LinRegr = LinearRegression()
LinRegr.fit(X_trainPCA,y_train)
predictions = LinRegr.predict(X_testPCA)

# The coefficients
print('Coefficients: \n', LinRegr.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test,predictions))
```
* Create a correlation matrix, plot it to a heatmap.
* Print the matrix to explore the independent variable relationships.
```py
# Correlation matrix
X_trainPCA = pd.DataFrame(X_trainPCA)
diab_corrPCA = X_trainPCA.corr()

# Generate correlation heatmap
ax = sns.heatmap(diab_corrPCA, center=0, cmap=sns.diverging_palette(20,220, n=256), square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

# Print correlations
print(diab_corrPCA)
```
## Random forest vs gradient boosting
* What are the main similarities and differences of Random Forest (RF) and Gradient Boosting (GB)algorithms?
Select the answer that is false:
* Random Forest and Gradient Boosting can use any algorithm, not just decision trees.

## Random forest ensemble
* Import the modules to create a Random Forest model and create a confusion matrix, accuracy, precision, recall, and F1-scores.
* Instantiate a RF classifier and set the appropriate argument to generate 50 estimators.
* Fit the data to the instantiated Random Forest Classifier model object.
* Create predictions using the trained model object.
* Evaluate the model fit.
```py
# Import
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Instantiate, fit, predict
rf_model = RandomForestClassifier(n_estimators=50, random_state=123, oob_score = True)
rf_model = rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Print evaluation metrics
print("Random Forest Accuracy: {}".format(accuracy_score(y_test,rf_pred)))
print("Confusion matrix:\n {}".format(confusion_matrix(y_test,rf_pred)))
print("Precision: {}".format(precision_score(y_test,rf_pred)))
print("Recall: {}".format(recall_score(y_test,rf_pred)))
print("F1: {}".format(f1_score(y_test,rf_pred)))
```

## Gradient boosting ensemble
* Import the modules to create a Gradient Boosting model and print out the confusion matrix, accuracy, precision, recall, and F1-scores.
* Instantiate a GB classifier and set the appropriate argument to generate 50 estimators and with a learning rate of 0.01.
* Fit the data and create predictions.
* Evaluate the model fit by printing trained model evaluation metrics.
```py
# Import
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Instantiate, fit, predict
gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.01,random_state=123)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Print evaluation metrics
print("Gradient Boosting Accuracy: {}".format(accuracy_score(y_test,gb_pred)))
print("Confusion matrix:\n {}".format(confusion_matrix(y_test,gb_pred)))
print("Precision: {}".format(precision_score(y_test,gb_pred)))
print("Recall: {}".format(recall_score(y_test,gb_pred)))
print("F1: {}".format(f1_score(y_test,gb_pred)))
```
* Pick the ensemble model that had the best accuracy.
```
Random Forest
Random Forest Accuracy: 0.7138666666666666
Gradient Boosting Accuracy: 0.7162
I DO NOT KNOW why answer is RF.....
```
*Finished by 2021/08/21*