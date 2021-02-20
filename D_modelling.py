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
from sklearn import metrics
import numpy as np
import math
import pickle



pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('datasets/AUS_Ignitions_2016.csv')
# print(df.shape)
#
#
# # remove fires which have no tweets associated with them
# df = df[df.num_tweets != 0]
# # print(df.shape)


# data information
num_tweets = df['num_tweets'].sum()
print('num tweets: {}'.format(num_tweets))

print('min number of tweets: {}, max number of tweets: {}, avg number of tweets: {}, st dev {}'.format(df['num_tweets'].min(), df['num_tweets'].max(), df['num_tweets'].mean(), df['num_tweets'].std()))

print('min magnitude: {}, max magnitude: {}, avg magnitude: {}, st dev {}'.format(df['magnitude'].min(), df['magnitude'].max(), df['magnitude'].mean(), df['magnitude'].std()))

print('min sentiment: {}, max sentiment: {}, avg sentiment: {}, st dev {}'.format(df['sentiment'].min(), df['sentiment'].max(), df['sentiment'].mean(), df['sentiment'].std()))


locations = df['location'].value_counts()
print('number of locations: {}'.format(locations.size))

print('')


def create_random_forest_regressor_all_vars(target, df):
    predictors = ['latitude', 'longitude', 'size', 'perimeter', 'duration', 'speed', 'expansion', 'sentiment',
                  'magnitude', 'num_tweets']
    predictors.remove(target)
    target = [target]

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


def create_random_forest_physical_regressor(target, df):
    target = [target]
    predictors = ['sentiment', 'magnitude', 'num_tweets']
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
    # print('R2 Score: {}%'.format(round((score) * 100), 4))
    # print(regr.get_params())

    # print(regr.oob_prediction_)
    # print('OOB Score: {}%'.format(round((regr.oob_score_) * 100), 4))

    y_pred = regr.predict(X_test)

    # visualise predictions
    # for i in range(len(y_pred)):
    #     print('{}, {}'.format(y_pred[i], y_test[i]))

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # print('MAE: {}'.format(mae))
    # print('MSE: {}'.format(mse))
    print('RMSE: {}'.format(rmse))

    print('')

    filename = 'models/' + target[0] + '_predictor_AUS.p'
    pickle.dump(regr, open(filename, 'wb'))

    return regr


def create_random_forest_sentiment_regressor(target, df):
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
    # print(regr.get_params())

    # print(regr.oob_prediction_)
    # print('OOB Score: {}%'.format(round((regr.oob_score_) * 100), 4))


    y_pred = regr.predict(X_test)

    # visualise predictions
    # for i in range(len(y_pred)):
    #     print('{}, {}'.format(y_pred[i], y_test[i]))

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)

    # print('MAE: {}'.format(mae))
    # print('MSE: {}'.format(mse))
    print('RMSE: {}'.format(rmse))

    print('')

    filename = 'models/' + target[0] + '_predictor_AUS.p'
    pickle.dump(regr, open(filename, 'wb'))

    return regr


def create_random_forest_location_regressor(df):
    lat_target = 'latitude'
    lon_target = 'longitude'

    predictors = ['sentiment', 'magnitude', 'num_tweets']
    df[predictors] = df[predictors] / df[predictors].max()
    print("Target: Location")

    # create train and test sets
    X = df[predictors].values

    y_lat = df[lat_target].values
    y_lon = df[lon_target].values

    X_lat_train, X_lat_test, y_lat_train, y_lat_test = train_test_split(X, y_lat, test_size=0.25)
    X_lon_train, X_lon_test, y_lon_train, y_lon_test = train_test_split(X, y_lon, test_size=0.25)


    # Random forests
    print('Random forests model:')
    regr_lat = RandomForestRegressor(
        n_estimators=1000,
        criterion='mse',
        oob_score=True
    )
    regr_lon = RandomForestRegressor(
        n_estimators=1000,
        criterion='mse',
        oob_score=True
    )

    print('RESULTS FOR LATITUDE MODEL')
    regr_lat.fit(X_lat_train, y_lat_train.ravel())
    score = regr_lat.score(X_lat_test, y_lat_test)
    print('R2 Score: {}%'.format(round((score) * 100), 4))
    print(regr_lat.get_params())

    # print(regr.oob_prediction_)
    print('OOB Score: {}%'.format(round((regr_lat.oob_score_) * 100), 4))

    print('RESULTS FOR LONGITUDE MODEL')
    regr_lon.fit(X_lon_train, y_lon_train.ravel())
    score = regr_lon.score(X_lon_test, y_lon_test)
    print('R2 Score: {}%'.format(round((score) * 100), 4))
    print(regr_lon.get_params())

    # print(regr.oob_prediction_)
    print('OOB Score: {}%'.format(round((regr_lon.oob_score_) * 100), 4))


    y_lat_pred = regr_lat.predict(X_lat_test)
    y_lon_pred = regr_lon.predict(X_lon_test)

    lon_diffs = []
    lat_diffs = []
    distances_km = []

    for i in range(len(y_lat_pred)):
        y_i_lat_pred = y_lat_pred[i]
        y_i_lat_true = y_lat_test[i]
        y_i_lon_pred = y_lon_pred[i]
        y_i_lon_true = y_lon_test[i]

        lon_diff, lat_diff, distance_km = calculate_distance_from_coordinate_estimate(y_i_lon_true, y_i_lat_true, y_i_lon_pred, y_i_lat_pred)
        lon_diffs.append(lon_diff)
        lat_diffs.append(lat_diff)
        distances_km.append(distance_km)
        print('Error: {} km'.format(distance_km))

    array = np.array([y_lat_test, y_lat_pred, y_lon_test, y_lon_pred, lat_diffs, lon_diffs, distances_km])
    array_t = np.transpose(array)

    loc_df = pd.DataFrame(array_t, columns=['lat_true', 'lat_pred', 'lon_true', 'lon_pred' ,'lat_distance_km', 'lon_distance_km', 'diff_km'])
    print(loc_df)

    print('Max distance error: {}. Min distance error: {}. AVG distance error: {}'.format(loc_df['diff_km'].max(), loc_df['diff_km'].min(), loc_df['diff_km'].mean()))
    avg_distance_err = loc_df['diff_km'].mean()

    mae = abs(avg_distance_err)
    print('MAE: {}'.format(mae))

    #
    # mse = metrics.mean_squared_error(loc_df['lat_distance'], loc_df['lon_distance'])
    # print('MSE: {}'.format(mse))
    # rmse = math.sqrt(mse)
    # print('RMSE: {}'.format(rmse))
    #
    # print()
    #
    # return regr_lat, regr_lon


def calculate_distance_from_coordinate_estimate(lon_true, lat_true, lon_pred, lat_pred):
    lon_diff = (lon_pred - lon_true) * 111
    # print('Lon diff: {} km'.format(lon_diff))
    lon_diff_squared_km = lon_diff ** 2
    lat_diff = (lat_pred - lat_true) * 111
    # print('Lat diff: {} km'.format(lat_diff))
    lat_diff_squared_km = lat_diff ** 2

    diff = math.sqrt(lon_diff_squared_km + lat_diff_squared_km)
    return lon_diff, lat_diff, diff



# # Run random forest regression models on variables of the opposing type
regressor = create_random_forest_sentiment_regressor('magnitude', df)
regressor = create_random_forest_sentiment_regressor('sentiment', df)
regressor = create_random_forest_sentiment_regressor('num_tweets', df)

# latitude_regressor = create_random_forest_location_regressor(df)
regressor = create_random_forest_physical_regressor('longitude', df)
regressor = create_random_forest_physical_regressor('latitude', df)
regressor = create_random_forest_physical_regressor('size', df)
regressor = create_random_forest_physical_regressor('perimeter', df)
regressor = create_random_forest_physical_regressor('duration', df)
regressor = create_random_forest_physical_regressor('speed', df)
regressor = create_random_forest_physical_regressor('expansion', df)

# Run random forest regression on ALL remaining variables
# regressor = create_random_forest_regressor_all_vars('magnitude', df)
# regressor = create_random_forest_regressor_all_vars('sentiment', df)
# regressor = create_random_forest_regressor_all_vars('num_tweets', df)

# regrressor = create_random_forest_regressor_all_vars('latitude', df)
# regrressor = create_random_forest_regressor_all_vars('longitude', df)
# regrressor = create_random_forest_regressor_all_vars('size', df)
# regrressor = create_random_forest_regressor_all_vars('perimeter', df)
# regrressor = create_random_forest_regressor_all_vars('duration', df)
# regrressor = create_random_forest_regressor_all_vars('speed', df)
# regrressor = create_random_forest_regressor_all_vars('expansion', df)