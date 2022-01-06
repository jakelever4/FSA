import numpy as np
import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
import geopandas as gpd
import matplotlib.pyplot as plt
import h5py


pd.set_option("display.max_rows", None, "display.max_columns", None)


london = [51.51, 0.12]
la = [34.05, -118.24]
ny = [40.71, -74.00]


# print(lookup_density(london))
# print(lookup_density(la))
# print(lookup_density(ny))


def get_weather_for_day_only(timestamp, latitude, longitude):
    url = 'https://api.darksky.net/forecast/' + darksky_key + '/' + str(latitude) + ',' + str(longitude) + ',' + str(
        round(timestamp)) + '?' + 'units=si&' + 'exclude=flags'
    response = requests.request("GET", url)
    if response.status_code == 200:
        parsed = json.loads(response.text)
        return parsed
    else:
        print('Error: response code: ' + str(response.status_code))


def get_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)
    end = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)

    range = pd.date_range(start=start, end=end)
    return range


def get_weather_vector(start_date, end_date, lat, lon):
    date_range = get_date_range(start_date, end_date)

    cloud_cover = []
    humidity = []
    precip_intensity = []
    precip_probability = []
    pressure = []
    temp_max = []
    uv_index = []
    wind_bearing = []
    wind_speed = []

    for date in date_range:
        forecast = get_weather_for_day_only(date.timestamp(), lat, lon)
        forecast = forecast['daily']['data'][0]


        try:
            cloud_cover.append(forecast['cloudCover'])
        except KeyError:
            cloud_cover.append(None)

        try:
            humidity.append(forecast['humidity'])
        except KeyError:
            humidity.append(None)

        try:
            precip_intensity.append(forecast['precipIntensity'])
        except KeyError:
            precip_intensity.append(None)

        try:
            precip_probability.append(forecast['precipProbability'])
        except KeyError:
            precip_probability.append(None)

        try:
            pressure.append(forecast['pressure'])
        except KeyError:
            pressure.append(None)

        try:
            temp_max.append(forecast['temperatureMax'])
        except KeyError:
            temp_max.append(None)

        try:
            uv_index.append(forecast['uvIndex'])
        except KeyError:
            uv_index.append(None)

        try:
            wind_bearing.append(forecast['windBearing'])
        except KeyError:
            wind_bearing.append(None)

        try:
            wind_speed.append(forecast['windSpeed'])
        except KeyError:
            wind_speed.append(None)


        # print(json.dumps(forecast, indent=4, sort_keys=True))

    return [cloud_cover, humidity, precip_intensity, precip_probability, pressure, temp_max, uv_index, wind_bearing, wind_speed]





# [cloud_cover, humidity, precip_intensity, precip_probability, pressure, temp_max, uv_index, wind_bearing, wind_speed] = get_weather_vector('2016-10-28','2016-11-02', 20.8396,-156.418)
# for x in [cloud_cover, humidity, precip_intensity, precip_probability, pressure, temp_max, uv_index, wind_bearing, wind_speed]:
#     print(x)

# ignitions = gpd.read_file('Global_fire_atlas_V1_ignitions_2016/Global_fire_atlas_V1_ignitions_2016.shp')
# print(ignitions.shape)


# # df = pd.read_csv('datasets/V4_Ignitions_2016_I.csv')
# perimeters = gpd.read_file('Global_fire_atlas_V1_perimeter_2016/Global_fire_atlas_V1_perimeter_2016.shp')
# print(perimeters.head(5))
# # print(perimeters.geom_type)
# print(perimeters.shape)
#
# for perimeter in perimeters:
#     fig, ax = plt.subplots(figsize = (10,10))
#     perimeter.plot(ax=ax)
#     plt.show()

# hf = h5py.File('datasets/GFED4.1s_2020_beta.hdf5', 'r')
#
#
# print(hf.keys())
# print(hf.items())

# dataset = hf.get('lat')
# print(group)


# dataset2 = hf.get('emissions/01')
# print(dataset2.items())
# print(dataset2.keys())
#
# lat = hf.get('lat')
# lon = hf.get('lon')
#
# c = dataset2.get('C')
# dm = dataset2.get('DM')
# daily_fraction = dataset2.get('daily_fraction')
#
# for value in c:
#     print(value.shape)

# print(group2)
# print(group2.items())

