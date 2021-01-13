import fiona
import csv
import geocode
import location_obj


# BOUNDING BOX FOR AUSTRAILIA
countries = ['Australia', 'New Zealand']
lat_search = [-10.6681857235, -46.641235447]
lon_search = [113.338953078, 178.517093541]


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

    with open('datasets/AUS_Ignitions_2016_I.csv', mode='w') as dataset:
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
            'location',
            "state",
            "state_short",
            'sentiment',
            'magnitude',
            'num_tweets'
        ])

        for i in shape:
            if index > num_rows:
                return

            try:
                item = shape.next()
            except StopIteration:
                print('All entries in collection have been analysed! {} total. {} fell within bounding box'.format(index, in_box_index))
                for id in in_box_fire_ids:
                    print(id)
                return

            dic = item['properties']
            print('fire id: {}'.format(dic['fire_ID']))

            if dic['fire_ID'] > 0:

                lat = dic["latitude"]
                long = dic["longitude"]

                in_box = check_coords_in_bounding_box(lat,long)
                if in_box:

                    in_box_index += 1
                    in_box_fire_ids.append(dic['fire_ID'])
                    print('inside bounding box! index: {}'.format(in_box_index))

                    location = geocode.geocode_lookup(lat, long, countries)

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
                            location.name,
                            location.state,
                            location.state_short,
                            ''
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