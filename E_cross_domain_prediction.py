import pickle
import pandas as pd
from sklearn import metrics
import numpy as np
# import D_modelling

# load models from disk
duration_predictor_AUS = pickle.load(open('models/duration_predictor_AUS.p', 'rb'))
duration_predictor_NA = pickle.load(open('models/duration_predictor_NA.p', 'rb'))

expansion_predictor_AUS = pickle.load(open('models/expansion_predictor_AUS.p', 'rb'))
expansion_predictor_NA = pickle.load(open('models/expansion_predictor_NA.p', 'rb'))

latitude_predictor_AUS = pickle.load(open('models/latitude_predictor_AUS.p', 'rb'))
latitude_predictor_NA = pickle.load(open('models/latitude_predictor_NA.p', 'rb'))

magnitude_predictor_AUS = pickle.load(open('models/magnitude_predictor_AUS.p', 'rb'))
magnitude_predictor_NA = pickle.load(open('models/magnitude_predictor_NA.p', 'rb'))

num_tweets_predictor_AUS = pickle.load(open('models/num_tweets_predictor_AUS.p', 'rb'))
num_tweets_predictor_NA = pickle.load(open('models/num_tweets_predictor_NA.p', 'rb'))

longitude_predictor_AUS = pickle.load(open('models/longitude_predictor_AUS.p', 'rb'))
longitude_predictor_NA = pickle.load(open('models/longitude_predictor_NA.p', 'rb'))

perimeter_predictor_AUS = pickle.load(open('models/perimeter_predictor_AUS.p', 'rb'))
perimeter_predictor_NA = pickle.load(open('models/perimeter_predictor_NA.p', 'rb'))

sentiment_predictor_AUS = pickle.load(open('models/sentiment_predictor_AUS.p', 'rb'))
sentiment_predictor_NA = pickle.load(open('models/sentiment_predictor_NA.p', 'rb'))

size_predictor_AUS = pickle.load(open('models/size_predictor_AUS.p', 'rb'))
size_predictor_NA = pickle.load(open('models/size_predictor_NA.p', 'rb'))

speed_predictor_AUS = pickle.load(open('models/speed_predictor_AUS.p', 'rb'))
speed_predictor_NA = pickle.load(open('models/speed_predictor_NA.p', 'rb'))

# test
def run_cross_prediction(target, predictors, aus_predictor, na_predictor):
    # load datasets
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    df_aus = pd.read_csv('datasets/AUS_Ignitions_2016.csv')

    df_na = pd.read_csv('datasets/NA_Ignitions_2016.csv')
    df_na = df_na[df_na.num_tweets != 0]

    df_na[predictors] = df_na[predictors] / df_na[predictors].max()
    df_aus[predictors] = df_aus[predictors] / df_aus[predictors].max()

    print('TARGET: {}'.format(target))

    # create input datasets and true values
    X_na = df_na[predictors].values
    y_na = df_na[target].values

    X_aus = df_aus[predictors].values
    y_aus = df_aus[target].values

    # predict unseen data
    y_na_pred = aus_predictor.predict(X_na)
    y_aus_pred = na_predictor.predict(X_aus)

    # visualise predictions
    # for i in range(len(y_na_pred)):
    #     print('NA data, Aus model')
    #     print('pred: {}, true: {}'.format(y_na_pred[i], y_na[i]))
    #
    # for i in range(len(y_aus_pred)):
    #     print('AUS data, NA model')
    #     print('pred: {}, true: {}'.format(y_aus_pred[i], y_aus[i]))

    # calculate metrics
    mae_na = metrics.mean_absolute_error(y_na, y_na_pred)
    mse_na = metrics.mean_squared_error(y_na, y_na_pred)
    rmse_na = np.sqrt(mse_na)

    print('Results for AUS model predicting NA Data')
    print('MAE: {}'.format(mae_na))
    print('MSE: {}'.format(mse_na))
    print('RMSE: {}'.format(rmse_na))

    print('')

    mae_aus = metrics.mean_absolute_error(y_aus, y_aus_pred)
    mse_aus = metrics.mean_squared_error(y_aus, y_aus_pred)
    rmse_aus = np.sqrt(mse_aus)

    print('Results for NA model predicting AUS Data')
    print('MAE: {}'.format(mae_aus))
    print('MSE: {}'.format(mse_aus))
    print('RMSE: {}'.format(rmse_aus))

    print('')


physical_predictors = ['latitude', 'longitude', 'size', 'perimeter', 'duration', 'speed', 'expansion']
sentiment_predictors = ['sentiment', 'magnitude', 'num_tweets']

run_cross_prediction('sentiment', physical_predictors, sentiment_predictor_AUS, sentiment_predictor_NA)
run_cross_prediction('magnitude', physical_predictors, magnitude_predictor_AUS, magnitude_predictor_NA)
run_cross_prediction('num_tweets', physical_predictors, num_tweets_predictor_AUS, num_tweets_predictor_NA)
#
run_cross_prediction('latitude', sentiment_predictors, latitude_predictor_AUS, latitude_predictor_NA)
run_cross_prediction('longitude', sentiment_predictors, longitude_predictor_AUS, longitude_predictor_NA)
run_cross_prediction('size', sentiment_predictors, size_predictor_AUS, size_predictor_NA)
run_cross_prediction('perimeter', sentiment_predictors, perimeter_predictor_AUS, perimeter_predictor_NA)
run_cross_prediction('duration', sentiment_predictors, duration_predictor_AUS, duration_predictor_NA)
run_cross_prediction('speed', sentiment_predictors, speed_predictor_AUS, speed_predictor_NA)
run_cross_prediction('expansion', sentiment_predictors, expansion_predictor_AUS, expansion_predictor_NA)


# load datasets
pd.set_option("display.max_rows", None, "display.max_columns", None)
df_aus = pd.read_csv('datasets/AUS_Ignitions_2016.csv')

df_na = pd.read_csv('datasets/NA_Ignitions_2016.csv')
df_na = df_na[df_na.num_tweets != 0]

df_combined = pd.concat([df_na, df_aus])
df_combined.describe()
df_combined.to_csv('datasets/COMBINED_NA_AUS_Ignitions_2016.csv', header=True, index=False)

# regressor = D_modelling.create_random_forest_sentiment_regressor('sentiment', df_combined)
# regressor = D_modelling.create_random_forest_sentiment_regressor('magnitude', df_combined)
# regressor = D_modelling.create_random_forest_sentiment_regressor('num_tweets', df_combined)
#
# regressor = D_modelling.create_random_forest_physical_regressor('latitude', df_combined)
# regressor = D_modelling.create_random_forest_physical_regressor('longitude', df_combined)
# regressor = D_modelling.create_random_forest_physical_regressor('size', df_combined)
# regressor = D_modelling.create_random_forest_physical_regressor('perimeter', df_combined)
# regressor = D_modelling.create_random_forest_physical_regressor('duration', df_combined)
# regressor = D_modelling.create_random_forest_physical_regressor('speed', df_combined)
# regressor = D_modelling.create_random_forest_physical_regressor('expansion', df_combined)

