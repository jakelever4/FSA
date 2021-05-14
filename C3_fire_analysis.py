import pandas as pd
import SQLite
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics


# CREATE DATABASE CONNECTION
pd.set_option("display.max_rows", None, "display.max_columns", None)
db_file = 'US_V3.db'
conn = SQLite.create_connection(db_file)
fires_df = SQLite.select_all_fires(conn)


# CONVERT CATEGORICAL VARIABLES
fires_df['direction_cat'] = fires_df['direction'].astype('category').cat.codes
fires_df['landcover_cat'] = fires_df['landcover'].astype('category').cat.codes
fires_df['state_cat'] = fires_df['state'].astype('category').cat.codes

fires_df['start_doy'] = fires_df['start_date'].apply(lambda  x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)
fires_df['end_doy'] = fires_df['end_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)


# SET UP GEOMETRY FOR MAPS
crs = {'init': 'epsg:4326'}
geometry = [Point(xy) for xy in zip( fires_df['longitude'], fires_df['latitude'])]
geo_df = gpd.GeoDataFrame(fires_df, crs=crs, geometry=geometry)
na_map = gpd.read_file('USA_Canada_ShapefileMerge/USA_Canada_ShapefileMerge.shp')



fires_df.plot(subplots=True, layout=(3,6))
fires_df.hist(bins=30, figsize=(15, 5))
plt.show()

def plot_all_fires(colour_by):
    # PLOTTED FIRE MAP WITH COLOR BY VARIABLE
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
    na_map.plot(ax=ax, alpha=1.0, color='grey')
    geo_df.plot(column=colour_by, ax =ax, markersize=5, marker='.', label=colour_by, legend=True)
    # ax.set_aspect('auto')
    ax.set_xlim([-180, -55])
    ax.set_ylim([15,80])
    plt.legend(prop={'size': 10}, loc='best')
    plt.title('North American Fires Coloured by {}'.format(colour_by))
    plt.xlabel('Lon')
    plt.ylabel('Lat')

    plt.savefig('V3_graphs/fires_Map_{}.png'.format(colour_by))
    plt.show()


def predict(X, target_var, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, target_var, test_size=0.25)

    # Various hyper-parameters to tune
    xgb1 = xgb.XGBRegressor()
    parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
                  'objective': ['reg:linear'],
                  'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], #so called `eta` value
                  'max_depth': [7, 8, 9, 10],
                  'min_child_weight': [4],
                  'verbosity': [0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.4, 0.6],
                  'n_estimators': [100, 200, 300, 400, 500]}

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv = 4,
                            n_jobs = 4,
                            verbose=True)

    xgb_grid.fit(X_train, y_train)

    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)

    y_pred = xgb_grid.predict(X_test)
    y_test = y_test.to_numpy()
    for i in range(len(y_pred)):
        print('True: {} pred: {}'.format(y_test[i], y_pred[i]))

    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    print('Results for target variable: {}'.format(target_name))

    print('R2 Score: {}'.format(r2))
    print('MAE: {}'.format(mae))
    print('MSE: {}'.format(mse))

    score = xgb_grid.score(X_test, y_test)
    print(score)

    xgb.plot_tree(xgb_grid.best_estimator_,num_trees=0)
    plt.savefig('V3_graphs/tree_{}.png'.format(target_name), dpi=2000, bbox_inches='tight')
    # plt.show()

    # here the f score is how often the variable is split on - i.e. the F(REQUENCY) score
    xgb.plot_importance(xgb_grid.best_estimator_)
    plt.tight_layout()
    plt.savefig('V3_graphs/feature_importance_{}.png'.format(target_name))
    # plt.show()

    return None


predictors = ['latitude', 'longitude', 'size', 'perimeter', 'duration', 'speed', 'expansion', 'pop_density', 'direction_cat', 'landcover_cat', 'state_cat', 'start_doy', 'end_doy']
targets = ['sentiment', 'magnitude', 'num_tweets']
X = fires_df[predictors]


for target in targets:
    predict(X, fires_df[target], target)


for col in fires_df:
    plot_all_fires(col)






# for col in fires_df:
#     fires_df[col].hist(bins=30, figsize=(15,10))
#     plt.show()