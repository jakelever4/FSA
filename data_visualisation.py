import matplotlib.pyplot as plt
# import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd


pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.read_csv('datasets/NA_ignitions_2016.csv')
# print(df.shape)


# remove fires which have no tweets associated with them
df = df[df.num_tweets != 0]
# print(df.shape)
crs  = {'init': 'epsg:4326'}


# show all fires
all_fires = gpd.read_file('Global_fire_atlas_V1_ignitions_2016/Global_fire_atlas_V1_ignitions_2016.shp')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
all_fires.plot(ax=ax, marker='.', markersize=1)
plt.title('Global Fire Atlas 2016 Ignitions (All Fires)')
plt.xlabel('Lon')
plt.ylabel('Lat')
# plt.tight_layout()

plt.savefig('2016_all_fires.png')
plt.show()



# data visualisation
geometry = [Point(xy) for xy in zip( df['longitude'], df['latitude'])]
geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

canada_map = gpd.read_file('USA_Canada_ShapefileMerge/USA_Canada_ShapefileMerge.shp')


# Plotted map with NA states
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
canada_map.plot(ax=ax, column='StateName')
# ax.set_aspect('auto')
ax.set_xlim([-180, -55])
ax.set_ylim([20,80])
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('North American Area Scope')

plt.savefig('NA_Scope.png')
plt.show()


# PLOTTED FIRE MAP WITH SENTIMENT COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
canada_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df[geo_df['sentiment_score'] > -10].plot(column='sentiment_score', ax =ax, markersize=10, marker='.', label='Sentiment Score', legend=True)
# ax.set_aspect('auto')
ax.set_xlim([-180, -55])
ax.set_ylim([20,80])
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('North American (English Speaking) Fires Coloured by Sentiment Score')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('NA_Fires_Map_Sentiment.png')
plt.show()


# PLOTTED FIRE MAP WITH MAGNITUDE COLOR
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
canada_map.plot(ax=ax, alpha=0.8, color='grey')
geo_df[geo_df['magnitude'] > -10].plot(column='magnitude', ax =ax, markersize=10, marker='.', label='Magnitude', legend=True)
# ax.set_aspect('auto')
ax.set_xlim([-180, -55])
ax.set_ylim([20,80])
plt.legend(prop={'size': 10}, loc='lower left')
plt.title('North American (English Speaking) Fires Coloured by Magnitude Score')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('NA_Fires_Map_Magnitude.png')
plt.show()


# HISTOGRAM OF LOCATION FREQUENCY
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,6))
locations = df['location'].value_counts()
p = locations[locations > 15].plot(kind='bar', ax=ax)
# ax.set_aspect('auto')
plt.tight_layout()
plt.title('Histogram of Distinct Locations (Frequency > 15) Found From Geocode Analysis')

plt.savefig('location_freq_histogram.png')
plt.show()