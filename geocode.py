import googlemaps
import location_obj
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyAmoMO0gyuoSToFatQhJfidqkIv72f9hAE')

# Geocoding an address
# geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

def get_loc_from_results(reverse_geocode_result):
    for result in reverse_geocode_result:
        if 'administrative_area_level_2' in result['types']:
            name = result['formatted_address']
            state = ''
            for comp in result['address_components']:
                if 'administrative_area_level_1' in comp['types']:
                    state = comp['long_name'].replace(' ','')
                    state_short = comp['short_name'].replace(' ', '')


            location = location_obj.location(name, state, state_short)
            return location


def geocode_lookup(lat, long):
    # Look up an address with reverse geocoding
    reverse_geocode_result = gmaps.reverse_geocode((lat, long))
    for result in reverse_geocode_result:
        # print(result['types'])
        # print(result['formatted_address'])
        if 'country' in result['types'] and (result['formatted_address'] == 'United States' or result['formatted_address'] == 'Canada'):
            print('Found NA fire: {}'.format(result['formatted_address']))
            try:
                loc = get_loc_from_results(reverse_geocode_result)
            except KeyError:
                print('Couldnt get state & name from location obj')
                return None
            print(loc)
            return loc
        elif 'country' in result['types'] and result['formatted_address'] != 'United States':
            print('not US fire. Country: {}'.format(result['formatted_address']))
            return None


# x = geocode_lookup(62,-107)