import SQLite
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from shapely.geometry import Point
import geopandas as gpd


pd.set_option("display.max_rows", None, "display.max_columns", None)
db_file = 'database.db'
conn = SQLite.create_connection(db_file)
crs = {'init': 'epsg:4326'}

fires_df = SQLite.select_all_fires(conn)


# SHOW CORRELATION MATRIX FOR THE VARIABLES
f = plt.figure(figsize=(13, 8))
corr = fires_df.corr()
corr[corr < 0.8] = 0
plt.matshow(corr, fignum=f.number)
plt.xticks(range(fires_df.select_dtypes(['number']).shape[1]), fires_df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(fires_df.select_dtypes(['number']).shape[1]), fires_df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

plt.savefig('graphs/V4_Corelation_matrix.png')
plt.show()


# SCATTER MATRIX
scatter_matrix(fires_df, alpha=0.2)


# data visualisation
geometry = [Point(xy) for xy in zip( fires_df['longitude'], fires_df['latitude'])]
geo_fires_df = gpd.GeoDataFrame(fires_df, crs=crs, geometry=geometry)


na_map = gpd.read_file('USA_Canada_ShapefileMerge/USA_Canada_ShapefileMerge.shp')


# Plotted fires
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,8))
na_map.plot(ax=ax, color='grey')
geo_fires_df.plot(ax =ax, markersize=5, marker='.')
plt.title('North American Fires in GFA/Investigation Scope')
plt.xlabel('Lon')
plt.ylabel('Lat')

plt.savefig('graphs/V4_Scope.png')
plt.show()


for row in fires_df['sentiment']:
    print(row)