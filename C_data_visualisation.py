import matplotlib.pyplot as plt
# import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns


pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('fires_df.csv', error_bad_lines=False)
dfi = pd.read_csv('fires_df_aus.csv')

df['s'] = df['overall_sentiment'] / df['total_tweets']
df['m'] = df['overall_magnitude'] / df['total_tweets']

# remove fires which have no tweets associated with them
df = df[df.num_tweets != 0]
# print(df.shape)
crs = {'init': 'epsg:4326'}


# SHOW CORRELATION MATRIX FOR THE VARIABLES
# f = plt.figure(figsize=(13, 8))
# corr = df.corr()
# corr[corr<0.8] = 0
# plt.matshow(corr, fignum=f.number)
# plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
#
# plt.savefig('graphs/V4_Corelation_matrix.png')
# plt.show()
#
#
# # SCATTER MATRIX
# scatter_matrix(df, alpha=0.2)


# data visualisation
geometry = [Point(xy) for xy in zip( df['longitude'], df['latitude'])]
geometryi = [Point(xy) for xy in zip( dfi['longitude'], dfi['latitude'])]

geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
geo_dfi = gpd.GeoDataFrame(dfi, crs=crs, geometry=geometryi)


na_map = gpd.read_file('USA_Canada_ShapefileMerge/USA_Canada_ShapefileMerge.shp')
aus_map = gpd.read_file('AUS_2021_AUST_SHP_GDA2020/AUS_2021_AUST_GDA2020.shp')

# # Plotted ALL fires
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
na_map.plot(ax=ax, color='grey')
geo_df.plot(ax =ax, markersize=5, marker='.')
plt.title('All North American Fires in Global Fire Atlas Scope', fontsize=20)
plt.xlabel('Lon', fontsize=22)
plt.ylabel('Lat', fontsize=22)

ax.set_xlim([-180, -55])
ax.set_ylim([20,80])

plt.tight_layout()

plt.savefig('graphs2/US_Scope.png')
plt.show()


# PLOTTED FIRE MAP WITH SENTIMENT COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
na_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df.plot(column='s', ax =ax, markersize=5, marker='.', label='Avg Sentiment Score', legend=True)
# ax.set_aspect('auto')
ax.set_xlim([-180, -55])
ax.set_ylim([20,80])
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('North American Fires Coloured by Average Sentiment Score', fontsize=19)
plt.xlabel('Lon', fontsize=22)
plt.ylabel('Lat', fontsize=22)
plt.tight_layout()

plt.savefig('graphs2/US_sentiment_map.png')
plt.show()


# PLOTTED FIRE MAP WITH MAGNITUDE COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
na_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df.plot(column='m', ax =ax, markersize=5, marker='.', label='Avg Magnitude Score', legend=True)
# ax.set_aspect('auto')
ax.set_xlim([-180, -55])
ax.set_ylim([20,80])
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('North American Fires Coloured by Average Magnitude Score', fontsize=19)
plt.xlabel('Lon', fontsize=22)
plt.ylabel('Lat', fontsize=22)
plt.tight_layout()

plt.savefig('graphs2/US_magnitude_map.png')
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