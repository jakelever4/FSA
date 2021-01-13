import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
# import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import google_analyse_sentiment
import B_complete_sentiment_data
from sklearn import metrics
import numpy as np

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('NA_ignitions_2016.csv')
# print(df.shape)


# remove fires which have no tweets associated with them
df = df[df.num_tweets != 0]
# print(df.shape)


# data information


# X = complete_sentiment_data.get_queries_from_location('California', 'CA', "Monterey County, CA, USA".split(','))
# print(X)
# sentiment = google_analyse_sentiment.analyze('#CAStateParks urges the public to avoid traveling to impacted park units particularly in Santa Cruz, Sonoma Coast, Monterey  & Bay Area due to wildfires & safety concerns. Air quality remains poor. For the latest updates on park closures, please visit')
# print(sentiment)


num_tweets = df['num_tweets'].sum()
print('num tweets: {}'.format(num_tweets))

print('min number of tweets: {}, max number of tweets: {}, avg number of tweets: {}, st dev {}'.format(df['num_tweets'].min(), df['num_tweets'].max(), df['num_tweets'].mean(), df['num_tweets'].std()))

print('min magnitude: {}, max magnitude: {}, avg magnitude: {}, st dev {}'.format(df['magnitude'].min(), df['magnitude'].max(), df['magnitude'].mean(), df['magnitude'].std()))

print('min sentiment: {}, max sentiment: {}, avg sentiment: {}, st dev {}'.format(df['sentiment_score'].min(), df['sentiment_score'].max(), df['sentiment_score'].mean(), df['sentiment_score'].std()))


locations = df['location'].value_counts()
print('number of locations: {}'.format(locations.size))

print('')


def create_random_forest_physical_regressor(target):
    target = [target]
    predictors = ['sentiment_score', 'magnitude', 'num_tweets']
    df[predictors] = df[predictors] / df[predictors].max()
    print("Target: {}".format(target))

    # create train and test sets
    X = df[predictors].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Random forests
    print('Random forests model:')
    regr = RandomForestRegressor(
        n_estimators=1000,
        criterion='mse',
        oob_score=True
    )

    regr.fit(X_train, y_train.ravel())
    score = regr.score(X_test, y_test)
    print('R2 Score: {}%'.format(round((score) * 100), 4))
    print(regr.get_params())

    # print(regr.oob_prediction_)
    print('OOB Score: {}%'.format(round((regr.oob_score_) * 100), 4))

    y_pred = regr.predict(X_test)

    # visualise predictions
    # for i in range(len(y_pred)):
    #     print('{}, {}'.format(y_pred[i], y_test[i]))

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print('MAE: {}'.format(mae))
    print('MSE: {}'.format(mse))
    print('RMSE: {}'.format(rmse))

    print('')

    return regr


def create_random_forest_sentiment_regressor(target):
    target = [target]
    predictors = ['latitude', 'longitude', 'size', 'perimeter', 'duration', 'speed', 'expansion']
    df[predictors] = df[predictors] / df[predictors].max()
    print("Target: {}".format(target))

    # create train and test sets
    X = df[predictors].values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Random forests
    print('Random forests model:')
    regr = RandomForestRegressor(
        n_estimators=1000,
        criterion='mse',
        oob_score=True
    )

    regr.fit(X_train, y_train.ravel())
    score = regr.score(X_test, y_test)
    print('R2 Score: {}%'.format(round((score) * 100), 4))
    print(regr.get_params())

    # print(regr.oob_prediction_)
    print('OOB Score: {}%'.format(round((regr.oob_score_) * 100), 4))


    y_pred = regr.predict(X_test)

    # visualise predictions
    # for i in range(len(y_pred)):
    #     print('{}, {}'.format(y_pred[i], y_test[i]))

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)

    print('MAE: {}'.format(mae))
    print('MSE: {}'.format(mse))
    print('RMSE: {}'.format(rmse))

    print('')

    return regr


mag_regr = create_random_forest_sentiment_regressor('magnitude')
mag_regr = create_random_forest_sentiment_regressor('sentiment_score')
mag_regr = create_random_forest_sentiment_regressor('num_tweets')

regr = create_random_forest_physical_regressor('latitude')
regr = create_random_forest_physical_regressor('longitude')
regr = create_random_forest_physical_regressor('size')
regr = create_random_forest_physical_regressor('perimeter')
regr = create_random_forest_physical_regressor('duration')
regr = create_random_forest_physical_regressor('speed')
regr = create_random_forest_physical_regressor('expansion')