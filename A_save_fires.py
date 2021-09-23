import fiona
import csv
import geocode
import location_obj
import pandas as pd


# BOUNDING BOX FOR AUSTRAILIA
countries = ['Australia', 'New Zealand']
lat_search = [-10.6681857235, -46.641235447]
lon_search = [113.338953078, 178.517093541]


df = pd.read_csv('datasets/COMBINED_NA_AUS_Ignitions_2016.csv')
ids = df[['fire_ID', 'location', 'state', 'state_short']]
ids.astype({'fire_ID': 'int32'})

def check_coords_in_bounding_box(lat, lon):
    if lat_search[0] < lat < lat_search[1] or lat_search[1] < lat < lat_search[0]:
        if lon_search[0] < lon < lon_search[1] or lon_search[1] < lon < lon_search[0]:
            return True

    return False


def create_database(num_rows):
    shape = fiona.open('Global_fire_atlas_V1_ignitions_2016/Global_fire_atlas_V1_ignitions_2016.shp')
    index = 0

    in_box_index = 0
    in_box_fire_ids = []

    df1 = pd.read_csv('datasets/AUS_Ignitions_2016_I.csv')
    df2 = pd.read_csv('datasets/NA_ignitions_2016_I.csv')
    id1 = df1[['fire_ID', 'location', 'state', 'state_short']]
    id2 = df2[['fire_ID', 'location', 'state', 'state_short']]
    ids = pd.concat([id1,id2])
    ids.astype({'fire_ID': 'int32'})

    with open('datasets/V4_Ignitions_2016_I.csv', mode='w') as dataset:
        writer = csv.writer(dataset, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            "fire_ID",
            "latitude",
            "longitude",
            "size",
            "perimeter",
            "start_date",
            "end_date",
            "duration",
            "speed",
            "expansion",
            'direction',
            'landcover',
            'location',
            "state",
            "state_short",
            'pop_density',
            # 'cloud_cover',
            # 'humidity',
            # 'precip_intensity',
            # 'precip_prob',
            # 'pressure',
            # 'max_temp',
            # 'uv_index',
            # 'wind_bearing',
            # 'wind_speed',
            'sentiment',
            'magnitude',
            'num_tweets',
            'total_tweets',
            'overall_sentiment',
            'overall_magnitude',
        ])

        for i in shape:
            if index > num_rows:
                return

            try:
                item = shape.next()
            except StopIteration:
                print('All entries in collection have been analysed! {} total. {} fell within bounding box'.format(index, in_box_index))
                for id in in_box_fire_ids:
                    None
                    # print(id)
                return

            dic = item['properties']
            print('fire id: {}'.format(dic['fire_ID']))

            fire_id = dic['fire_ID']
            old_ids = ids['fire_ID']

            if fire_id in old_ids.values:

                old_dataset_row = ids[ids['fire_ID'] == dic['fire_ID']]
                print(old_dataset_row)

                lat = dic["latitude"]
                long = dic["longitude"]

                in_box = True # check_coords_in_bounding_box(lat,long)
                if in_box:

                    in_box_index += 1
                    in_box_fire_ids.append(dic['fire_ID'])
                    print('inside bounding box! index: {}'.format(in_box_index))


                    try:
                        location = old_dataset_row['location'].values[0]
                    except KeyError:
                        location = old_dataset_row['location']

                    try:
                        state = old_dataset_row['state'].values[0]
                    except KeyError:
                        state = old_dataset_row['state']

                    try:
                        state_short = old_dataset_row['state_short'].values[0]
                    except:
                        state_short = old_dataset_row['state_short']

                    if location is not None:
                        writer.writerow([
                            dic["fire_ID"],
                            dic["latitude"],
                            dic["longitude"],
                            dic["size"],
                            dic["perimeter"],
                            dic["start_date"],
                            dic["end_date"],
                            dic["duration"],
                            dic["speed"],
                            dic["expansion"],
                            dic['direction_s'],
                            dic['landcover_s'],
                            location,
                            state,
                            state_short
                        ])
                        print('row {} written'.format(index))
                        index += 1
                    else:
                        print('Location is in bounding box but couldnt be found.')
                else:
                    x = 1
                    print('Coords not within bounding box for AUS {} {}'.format(lat, long))



create_database(100000)


# 12,167 fires fall within bounding box for AUS & NZ