import SQLite
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from shapely.geometry import Point
import geopandas as gpd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
import sklearn.metrics
from scipy.interpolate import interp1d


pd.set_option("display.max_rows", None, "display.max_columns", None)
db_file = 'database.db'
conn = SQLite.create_connection(db_file)
crs = {'init': 'epsg:4326'}

# fires_df = SQLite.select_all_fires(conn)


def get_unique_days_for_fire(fire_ID, conn):
    q = """SELECT DISTINCT date FROM tweets WHERE fire_ID = {};""".format(fire_ID)
    dates = pd.Series(SQLite.execute_query(q, conn, cols=['date']).sort_values(['date'], ascending=[1]).squeeze())
    return dates


# Create more vars: calculates avg sentiment, date range,
# also max/min sentiment across all fires, min/max avg sentiment across all fires
# avg_sent_col = []
# date_range_col = []
# duration2_col = []
# for ind in fires_df.index:
#     avg_sentiment = []
#
#     sentiment = fires_df['sentiment'][ind]
#     magnitude = fires_df['magnitude'][ind]
#     pos_sentiment = fires_df['overall_positive_sentiment'][ind]
#     neg_sentiment = fires_df['overall_negative_sentiment'][ind]
#
#     # start_date = datetime.strptime(fires_df['start_date'][ind], '%Y-%m-%d')
#     # end_date = datetime.strptime(fires_df['end_date'][ind], '%Y-%m-%d')
#     # date_range = pd.date_range(start_date - timedelta(days=2), end_date)
#     print(fires_df['fire_ID'][ind])
#     unique_days = list(get_unique_days_for_fire(fires_df['fire_ID'][ind], conn).values)
#     print(unique_days)
#     date_range_col.append(unique_days)
#     duration2_col.append(len(unique_days))
#
#     max_sentiment = 0
#     max_pos_sent = 0
#     max_neg_sent = 0
#     max_avg_sentiment = 0
#     min_avg_sentiment = 0
#     for i in range(len(sentiment)):
#         if magnitude[i] != 0:
#             avg_s = sentiment[i] / magnitude[i]
#             if avg_s > max_avg_sentiment:
#                 max_avg_sentiment = avg_s
#                 max_avg_day_index = i
#                 max_avg_fire = ind
#             elif avg_s < min_avg_sentiment:
#                 min_avg_sentiment = avg_s
#                 min_avg_day_index = i
#                 min_avg_fire = ind
#         else:
#             avg_s = sentiment[i]
#         avg_sentiment.append(avg_s)
#         if sentiment[i] > max_sentiment:
#             max_sentiment = sentiment[i]
#             max_sent_day_index = i
#             fire_row_index = ind
#
#         pos_sent = pos_sentiment[i]
#         if pos_sent > max_pos_sent:
#             max_pos_sent = pos_sent
#             pos_sent_day = i
#             pos_sent_fire = ind
#
#         neg_sent = neg_sentiment[i]
#         if neg_sent < max_neg_sent:
#             max_neg_sent = neg_sent
#             neg_sent_day = i
#             neg_sent_fire = ind
#
#     avg_sent_col.append(avg_sentiment)
#
#
# column = pd.DataFrame(np.array(avg_sent_col), columns=['avg_sentiment'])
# fires_df['avg_sentiment'] = column
#
# date_ranges = pd.DataFrame(np.array(date_range_col), columns=['unique_days'])
# fires_df['unique_days'] = date_ranges
#
# duration2_col = pd.DataFrame(np.array(duration2_col), columns=['s_duration'])
# fires_df['s_duration'] = duration2_col
#
# print(fires_df)
# print('SAVING DF')
# fires_df.to_csv('fires_df.csv', index = False)


# print('Max sent: {}, on day: {} of fire ID {}'.format(max_sentiment,max_sent_day_index,fires_df['fire_ID'][fire_row_index]))
# print('Max pos sent: {}, on day: {} of fire ID {}'.format(max_pos_sent,pos_sent_day,fires_df['fire_ID'][pos_sent_fire]))
# print('Max  neg sent: {}, on day: {} of fire ID {}'.format(max_neg_sent,neg_sent_day,fires_df['fire_ID'][neg_sent_fire]))
# print('Max  avg sent: {}, on day: {} of fire ID {}'.format(max_avg_sentiment,max_avg_day_index,fires_df['fire_ID'][max_avg_fire]))
# print('Min  avg sent: {}, on day: {} of fire ID {}'.format(min_avg_sentiment,min_avg_day_index,fires_df['fire_ID'][min_avg_fire]))



fires_df = pd.read_csv('fires_df.csv')

fires_df['sentiment'] = fires_df['sentiment'].apply(lambda x: json.loads(x))
fires_df['overall_positive_sentiment'] = fires_df['overall_positive_sentiment'].apply(lambda x: json.loads(x))
fires_df['overall_negative_sentiment'] = fires_df['overall_negative_sentiment'].apply(lambda x: json.loads(x))
fires_df['magnitude'] = fires_df['magnitude'].apply(lambda x: json.loads(x))
fires_df['num_tweets'] = fires_df['num_tweets'].apply(lambda x: json.loads(x))
fires_df['avg_sentiment'] = fires_df['avg_sentiment'].apply(lambda x: json.loads(x))

fires_df['start_doy'] = fires_df['start_date'].apply(lambda  x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)
fires_df['end_doy'] = fires_df['end_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)

# avg_sentiment = []
avg_magnitude = []
for ind, row in fires_df.iterrows():
    # s = [x / y for x,y in zip(row['sentiment'],row['num_tweets'])]
    m = [x / y for x,y in zip(row['magnitude'],row['num_tweets'])]
    # avg_sentiment.append(s)
    avg_magnitude.append(m)

# fires_df['s'] = avg_sentiment
fires_df['avg_magnitude'] = avg_magnitude

fires_df['s_mean'] = fires_df['sentiment'].apply(lambda x: statistics.mean(x))
fires_df['s_var'] = fires_df['sentiment'].apply(lambda x: np.var(x))

fires_df['m_mean'] = fires_df['magnitude'].apply(lambda x: statistics.mean(x))
fires_df['m_var'] = fires_df['magnitude'].apply(lambda x: np.var(x))

fires_df.drop(columns=['duration'])


def run_kmeans(kmax, data):
    sils = []
    max_sil = 0
    best_k = 0
    for k in range(2,kmax + 1):
        kmeans = KMeans(n_clusters=k)
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        score = silhouette_score(data, kmeans.labels_, metric='euclidean')
        sils.append(score)
        print('K: {}, Sil Score: {}'.format(k,score))
        if score > max_sil:
            max_sil = score
            best_k = k
            best_est = kmeans

        print(kmeans.cluster_centers_)
        print(kmeans.labels_)

    print('Best K:{} SIL score: {}'.format(best_k, max_sil))
    fig, ax = plt.subplots(1)
    fig.suptitle('Silhouette score as K varies')
    ax.set_xlabel('K')
    ax.set_ylabel('sil score')
    ax.plot(range(2, kmax + 1), sils)
    plt.show()

    return best_est, best_k


def plot_sentiment_for_fire(fire):
    # if fire['duration'] != 17:
    #     return

    dates = get_unique_days_for_fire(fire['fire_ID'], conn)
    short_dates = []
    for date in dates:
        short_dates.append(date[5:])
    dates = short_dates

    try:
        label = fire['label']
    except ValueError:
        label = 'None'

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(14, 8))
    fig.suptitle('Label: {}, fire ID: {}, Location: {}. over burn period ({} to {})'.format(label, fire['fire_ID'], fire['location'], fire['start_date'], fire['end_date']))

    axs[0,0].set_title('SUM(Sentiment)')
    axs[0,0].set(xlabel='Date', ylabel='SUM(Sentiment)')
    axs[0,1].set_title('SUM(Magnitude)')
    axs[0,1].set(ylabel='SUM(Magnitude)')
    axs[1,0].set(ylabel='AVG(Sentiment)')
    axs[1,0].set_title('AVG(Sentiment)')
    axs[1,1].set(ylabel='AVG(Magnitude)')
    axs[1,1].set_title('AVG(Magnitude)')

    magnitude = fire['magnitude']
    sentiment = fire['sentiment']
    num_tweets = fire['num_tweets']
    avg_sentiment = fire['avg_sentiment']
    avg_magnitude = fire['avg_magnitude']
    pos_sentiment = fire['overall_positive_sentiment']
    neg_sentiment = fire['overall_negative_sentiment']

    axs[0,0].plot(dates, sentiment, color='b')
    axs[0,0].bar(dates, pos_sentiment, color='g')
    axs[0,0].bar(dates, neg_sentiment, color='r')
    axs[0,1].plot(dates, magnitude, color='orange')
    axs[1,0].plot(dates, avg_sentiment, color= colors[fire['label']]) #
    axs[1,0].plot(dates, np.zeros(len(magnitude)), color='deepskyblue', ls=':')
    axs[1,1].plot(dates, avg_magnitude, color='purple')

    axs[1,0].set_ylim([-1, 1])

    plt.setp(axs[1,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(axs[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')

    plt.show()
    fig.savefig('graphs/senti_{}.png'.format(fire['fire_ID']))


def perform_PCA(fires_df, n_components, attribute):
    fires_to_reduce = fires_df[fires_df['s_duration'].values >= n_components]
    fires_to_reduce = fires_to_reduce.sort_values(['s_duration', 'fire_ID'], ascending=[1,1])
    dmax = fires_to_reduce['s_duration'].max()
    reduced_fires_df = pd.DataFrame()
    explained_variance = []

    for d in range(n_components + 1, dmax + 1):
        print('reducing fires of d: {}'.format(d))
        fires = fires_to_reduce[fires_to_reduce['s_duration'] == d]

        modelData = list(fires[attribute])
        print('{} fires found'.format(len(modelData)))
        if len(modelData) < n_components:
            print('not enough fires to perform PCA, skipping')
            continue
        modelData = pd.DataFrame(modelData)

        pipeline = make_pipeline(StandardScaler(), PCA(n_components=n_components))
        pipeline.fit(modelData)

        pca = pipeline.steps[1][1]
        reduced_data = pca.fit_transform(modelData)
        reduced_data_list = []
        for row in reduced_data:
            reduced_data_list.append(row)
        fires['PCA_data'] = reduced_data_list

        reduced_fires_df = reduced_fires_df.append(fires)

        # print(reduced_data)
        print(pca.explained_variance_ratio_)
        exp_var = sum(pca.explained_variance_ratio_)
        explained_variance.append(exp_var)
        print('explained variance: {}'.format(exp_var))

    var = statistics.mean(explained_variance)
    print(explained_variance)
    print('average explained varaince across all fire durations: {}'.format(var))
    return reduced_fires_df


# APRIL 2021 - PREDICTING SENTIMENTAL VARIABLES

# categorical variables to convert
print(fires_df['direction'].value_counts())
print(fires_df['landcover'].value_counts())
print(fires_df['state'].value_counts())

fires_df['direction_cat'] = fires_df['direction'].astype('category').cat.codes
fires_df['landcover_cat'] = fires_df['landcover'].astype('category').cat.codes
fires_df['state_cat'] = fires_df['state'].astype('category').cat.codes


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
    plt.savefig('graphs/tree_{}.png'.format(target_name), dpi=2000, bbox_inches='tight')
    # plt.show()

    # here the f score is how often the variable is split on - i.e. the F(REQUENCY) score
    xgb.plot_importance(xgb_grid.best_estimator_)
    plt.tight_layout()
    plt.savefig('graphs/feature_importance_{}.png'.format(target_name))
    # plt.show()

    return None


def interpolate_vector(fire_col, fire_df, length):
    fire_col = fire_df[fire_col]
    interp_data = []
    for y in fire_col:
        if len(y) < 4:
            interp_data.append(None)
            continue
        x = np.linspace(0, 1, num=len(y), endpoint=True)
        xnew = np.linspace(0, 1, num=length, endpoint=True)

        f = interp1d(x, y, kind='cubic')
        ynew = f(xnew)
        interp_data.append(ynew)
        y0 = np.linspace(0, 0, num=length, endpoint=True)

        plt.plot(x, y, 'o', xnew, ynew, '-', xnew, y0, '--')
        plt.show()

    fire_df['interp_data'] = interp_data
    return fire_df


# PART 1 F(X) - PREDICTING SOCIAL SENTIMENT VALUES
predictors = ['latitude', 'longitude', 'size', 'perimeter', 's_duration', 'speed', 'expansion', 'pop_density', 'direction_cat', 'landcover_cat', 'state_cat', 'start_doy', 'end_doy']
targets = ['s_mean', 'm_mean', 'overall_magnitude', 'overall_sentiment', 'total_tweets', 's_var', 'm_var']
X = fires_df[predictors]

for pred in targets:
    print(fires_df[pred].describe())

# s_mean = fires_df['s_mean']
# h = s_mean.hist(bins = 300)
# print(fires_df['s_mean'].describe())

fires_df = interpolate_vector('sentiment', fires_df, 20)

for target in targets:
    predict(X, fires_df[target], target)




# OLD MARCH 21 WORK - PCA OF SENTIMENT VECTORS, KMEANS OF FIRES ETC


fires_df = fires_df.sort_values(['s_duration', 'fire_ID'], ascending=[0,1])


pca_dim = 8
pca_var = 'avg_sentiment'

reduced_data = perform_PCA(fires_df,pca_dim, pca_var)
x = pd.DataFrame.from_records(np.array(reduced_data['PCA_data']))

kmeans, k = run_kmeans(14, x)
labels = kmeans.labels_
reduced_data['label'] = labels

colors = ['red','green','blue','purple', 'cyan', 'orange', 'yellow', 'navy', 'black', 'slateblue', 'indigo', 'plum', 'violet', 'aqua']

fig, axs = plt.subplots(k,2, figsize=(20, 10))

fig.suptitle('Reduced and Original Dataset coloured by Kmeans label')
axs[0,0].set_title('Reduced Data (PCA)')
axs[1,0].set(xlabel='Day', ylabel=pca_var)
axs[0,1].set_title('Original Data')
axs[1,1].set(xlabel='Day', ylabel=pca_var)

reduced_label_means = []
x_label_means = []
for l in range(k):
    fires = reduced_data[reduced_data['label'] == l]

    red_data = pd.DataFrame.from_records(np.array(fires['PCA_data']))
    rd_mean = np.mean(red_data)
    reduced_label_means.append(rd_mean)
    x = pd.DataFrame.from_records(np.array(fires[pca_var]))
    mean_x = np.mean(x)
    x_label_means.append(mean_x)

    for ind, fire in fires.iterrows():
        rd = fire['PCA_data']
        x = fire[pca_var]
        axs[l,0].plot(range(len(rd)), rd, c=colors[l])
        axs[l,1].plot(range(len(x)), x, c=colors[l])

    axs[l,0].plot(range(len(rd_mean)), rd_mean, c='black')
    axs[l,1].plot(range(len(mean_x)), mean_x, c='black')
    axs[l,0].set(ylabel=pca_var)
    axs[l,0].set(ylabel=pca_var)
    # axs[l,0].plot(np.unique(x), np.poly1d(np.polyfit(x, range(len(x)), 1))(np.unique(x)))

plt.show()

def plot_PCA_data(fire):
    fig, ax = plt.subplots(figsize=(20, 10))
    fig.suptitle('Reduced Data Sentiment Curve for fire id {}'.format(fire['fire_ID']))

    reduced_data = fire['PCA_data']
    ax.plot(range(len(reduced_data)), reduced_data)
    ax.set(xlabel='t', ylabel=pca_var)
    ax.set_ylim([-1, 1])
    ax.plot(range(len(reduced_data)), np.zeros(len(reduced_data)), color='deepskyblue', ls=':')
    plt.show()



# DATA MODELLING / ML
# X = PHYSICAL VARIABLES - PREDICTORS
# Y = SOCIAL VARIABLES - TARGETS


X = reduced_data.filter(predictors, axis=1) # 'direction', 'landcover', 'latitude', 'longitude',

Y = reduced_data.filter(['sentiment', 'overall_sentiment', 'overall_positive_sentiment', 'overall_negative_sentiment',
                         'magnitude', 'overall_magnitude', 'num_tweets', 'total_tweets', 'avg_sentiment',
                         'avg_magnitude', 's_mean', 's_var', 'm_mean', 'm_var', 'reduced_data', 'label'], axis=1)

y = Y['label']


def predict_labels(X, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25)

    # param_grid = {
    #     'booster': ['gbtree', 'dart'],
    #     'n_estimators': [100,500],
    #     'eta': [0.1,0.3],
    #     #'gamma': [0, 1],
    #     'max_depth': [2,4],
    #     #'lambda': [1,2],
    #     #'alpha': [0,1],
    #     #'tree_method': ['auto', 'exact', 'approx', 'hist']
    # }

    parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
                  'objective': ['reg:logistic'],
                  'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2], #so called `eta` value
                  'max_depth': [6, 7, 8, 9],
                  'min_child_weight': [4],
                  'verbosity': [0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.4, 0.6],
                  'n_estimators': [100, 200, 300, 400, 500]}

    clf = XGBClassifier()
    gscv = GridSearchCV(estimator=clf, param_grid=parameters, cv = 4, n_jobs = 4, verbose=True)
    gscv.fit(X_train, y_train)

    print(gscv.best_params_)
    # print(gscv)
    # print(gscv.cv_results_.keys())
    # print(gscv.cv_results_)

    y_pred = gscv.predict(X_test)
    y_test = y_test.to_numpy()

    for i in range(len(y_pred)):
        print('True: {} pred: {}'.format(y_test[i], y_pred[i]))

    score = gscv.score(X_test, y_test)
    print(score)



predict_labels(X, y)

# print(fires_df)

reduced_data = reduced_data.iloc[::-1]
for ind, row in reduced_data.iterrows():
    plot_PCA_data(row)
    plot_sentiment_for_fire(row)


# i = 0
# for ind, row in reduced_data['reduced_data'].iteritems():
#     days = list(range(len(row)))
#     label = labels[i]
#     ax[label].plot(days, row, c=colors[label])
#     i += 1
# plt.show()

#
# plot_stack(fires_df, 6, 14)
# plot_stack(fires_df, 10, 14)

# fig, axs = plt.subplots(k,2)
#
# i = 0
# for ind, row in reduced_data.iterrows():
#     red_data = row['reduced_data']
#     avg_s = row[pca_var]
#     rd_x = list(range(len(red_data)))
#     s_x = list(range(len(avg_s)))
#     label = labels[i]
#     axs[0,0].set_title('Reduced Data (PCA)')
#     axs[0,1].set(xlabel='Day', ylabel=pca_var)
#
#     axs[0,1].set_title('Original Data')
#     axs[1,1].set(xlabel='Day', ylabel=pca_var)
#
#     axs[label,0].plot(rd_x, red_data, c=colors[label])
#     axs[label,1].plot(s_x, avg_s, c=colors[label])
#     i += 1
#
# plt.show()

# SHOW CORRELATION MATRIX FOR THE VARIABLES
# f = plt.figure(figsize=(13, 8))
# corr = fires_df.corr()
# corr[abs(corr) < 0.5] = 0
# plt.matshow(corr, fignum=f.number)
# plt.xticks(range(fires_df.select_dtypes(['number']).shape[1]), fires_df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(fires_df.select_dtypes(['number']).shape[1]), fires_df.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
#
# plt.savefig('graphs/V4_Corelation_matrix.png')
# plt.show()


# SCATTER MATRIX
# scatter_matrix(fires_df, alpha=0.2)



# data visualisation
geometry = [Point(xy) for xy in zip( fires_df['longitude'], fires_df['latitude'])]
geo_fires_df = gpd.GeoDataFrame(fires_df, crs=crs, geometry=geometry)

na_map = gpd.read_file('USA_Canada_ShapefileMerge/USA_Canada_ShapefileMerge.shp')


# Plotted fires
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
# na_map.plot(ax=ax, color='grey')
# geo_fires_df.plot(ax =ax, markersize=5, marker='.')
# ax.set_xlim([-180, -55])
# ax.set_ylim([20,80])
# plt.title('North American Fires in GFA/Investigation Scope')
# plt.xlabel('Lon')
# plt.ylabel('Lat')
#
# plt.savefig('graphs/V4_Scope.png')
# plt.show()


