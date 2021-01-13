import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import numpy
from sklearn.ensemble import RandomForestRegressor



# load data
df = pd.read_csv('NA_ignitions_2016.csv')
# print(df.head())

# print(df.sum(axis=0))

#create targets and predictors
target = ['magnitude'] # 'num_tweets'
predictors = ['latitude','longitude','size','perimeter','duration','speed','expansion']
df[predictors] = df[predictors] / df[predictors].max()
print("Target: {}".format(target))
print()


# create train and test sets
X = df[predictors].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)


# SVR
print("SVR Model:")
svr = SVR() # create an empty regressor for the data
parameters = [{'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
              'C':[10,100,500,1000],
              'gamma': [0.01,0.05,0.1,0.15,0.2],
              'epsilon':[0.005, 0.025, 0.05, 0.075, 0.1]},] # parameters for grid search

clf = GridSearchCV(svr, parameters, n_jobs=1)
clf.fit(X_train, y_train.ravel()) # fit the regressor to the data
X_pred = clf.predict(X_test) # run the regressor on the test set
y_test = numpy.array(y_test)

#print_pairs(pred_norm, test_targets_norm) # print pairs of results from test set
# print('Best Parameters: {}'.format(clf.best_params_))
# print('')
print('R2 Score: {}%'.format(round((clf.score(X_test, y_test)) * 100),4))
print('Best Estimator: {}'.format(clf.best_estimator_))
# print('Best Sore: {}'.format(clf.best_score_))
print('')


# Random forests
print('Random forests model:')
regr = RandomForestRegressor()
regr.fit(X_train, y_train.ravel())
score = regr.score(X_test, y_test)
print('R2 Score: {}%'.format(round((score) * 100),4))
print(regr.get_params())

print('')
