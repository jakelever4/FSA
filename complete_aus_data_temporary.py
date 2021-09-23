import pandas as pd
import fiona



fires_df = pd.read_csv('datasets/AUS_Ignitions_2016.csv')
shape = fiona.open('Global_fire_atlas_V1_ignitions_2016/Global_fire_atlas_V1_ignitions_2016.shp')
fire_ids = fires_df['fire_ID'].tolist()

def fill_variables(fires_df, shape):
    fire_ids = fires_df['fire_ID'].tolist()
    directions = []
    landcovers = []
    for i in shape:
        try:
            item = shape.next()
        except StopIteration:
            dir = pd.Series(directions)
            lc = pd.Series(landcovers)

            # fires_df['direction'] = dir
            # fires_df['landcover'] = lc
            fires_df.insert(10, 'direction', dir)
            fires_df.insert(11, 'landcover', lc)

            # fires_df = fires_df['fire_ID','latitude','longitude','size','perimeter','start_date','end_date','duration','speed','expansion','direction','landcover','location','state','state_short','sentiment','magnitude','num_tweets']

            fires_df.to_csv('datasets/aus_data_wcats.csv', index=False)
            return

        dic = item['properties']
        fire_ID = dic['fire_ID']
        print(fire_ID)
        if fire_ID in fire_ids:
            print("aus fire found")
            direction_s = dic['direction_s']
            landcover = dic['landcover_s']
            directions.append(direction_s)
            landcovers.append(landcover)

fill_variables(fires_df,shape)