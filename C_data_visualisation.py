import matplotlib.pyplot as plt
# import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns


pd.set_option("display.max_rows", None, "display.max_columns", None)
# df = pd.read_csv('datasets/AUS_ignitions_2016.csv')
# dfi = pd.read_csv('datasets/AUS_ignitions_2016_I.csv')



x = pd.read_csv('datasets/NA_ignitions_2016.csv', error_bad_lines=False)
hist = x.hist(column='duration', bins=50)
# hist.show()
# print(df.shape)


# remove fires which have no tweets associated with them
df = df[df.num_tweets != 0]
# print(df.shape)
crs = {'init': 'epsg:4326'}


# SHOW CORRELATION MATRIX FOR THE VARIABLES
f = plt.figure(figsize=(13, 8))
corr = df.corr()
corr[corr<0.8] = 0
plt.matshow(corr, fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

plt.savefig('graphs/AUS_Corelation_matrix.png')
plt.show()


# SCATTER MATRIX
scatter_matrix(df, alpha=0.2)


# data visualisation
geometry = [Point(xy) for xy in zip( df['longitude'], df['latitude'])]
geometryi = [Point(xy) for xy in zip( dfi['longitude'], dfi['latitude'])]

geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
geo_dfi = gpd.GeoDataFrame(dfi, crs=crs, geometry=geometryi)


aus_map = gpd.read_file('nsaasr9nnd_02211a04es_geo___/aust_cd66states.shp')


# Plotted ALL fires
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
aus_map.plot(ax=ax, color='grey')
geo_dfi.plot(ax =ax, markersize=5, marker='.')
plt.title('All Austrailian Fires in Global Fire Atlas Scope')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('graphs/AUS_Scope.png')
plt.show()


# PLOTTED FIRE MAP WITH SENTIMENT COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
aus_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df.plot(column='sentiment', ax =ax, markersize=5, marker='.', label='Sentiment Score', legend=True)
# ax.set_aspect('auto')
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('Austrailian Fires Coloured by Sentiment Score')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('graphs/AUS_Fires_Map_Sentiment.png')
plt.show()


# PLOTTED FIRE MAP WITH MAGNITUDE COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
aus_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df.plot(column='magnitude', ax =ax, markersize=5, marker='.', label='Magnitude Score', legend=True)
# ax.set_aspect('auto')
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('Austrailian Fires Coloured by Magnitude Score')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('graphs/AUS_Fires_Map_Magnitude.png')
plt.show()


# PLOTTED FIRE MAP WITH NUM_TWEETS COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
aus_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df.plot(column='num_tweets', ax =ax, markersize=5, marker='.', label='Num_tweets', legend=True)
# ax.set_aspect('auto')
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('Austrailian Fires Coloured by Number of Tweets')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('graphs/AUS_Fires_Map_num_tweets.png')
plt.show()


# HISTOGRAM OF LOCATION FREQUENCY
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,6))
locations = df['location'].value_counts()
p = locations[locations > 5].plot(kind='bar', ax=ax)
# ax.set_aspect('auto')
plt.tight_layout()
plt.title('Histogram of Distinct Locations (Frequency > 15) Found From Geocode Analysis')

plt.savefig('graphs/AUS_location_freq_histogram.png')
plt.show()

locations = df['location'].value_counts()
print('number of locations: {}'.format(locations.size))

print('')


# # show all fires
# all_fires = gpd.read_file('Global_fire_atlas_V1_ignitions_2016/Global_fire_atlas_V1_ignitions_2016.shp')
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
# all_fires.plot(ax=ax, marker='.', markersize=1)
# plt.title('Global Fire Atlas 2016 Ignitions (All Fires)')
# plt.xlabel('Lon')
# plt.ylabel('Lat')
# # plt.tight_layout()
#
# plt.savefig('graphs/2016_all_fires.png')
# plt.show()