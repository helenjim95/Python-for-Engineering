import json
import collections
from geopy.geocoders import Nominatim
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from matplotlib import pyplot as plt
import movingpandas as mpd
import contextily as ctx


def read_data(filepath) -> list[dict]:
    with open(filepath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def plot_histogram(data_list):
    event_list = []
    for (i, e) in enumerate(data_list):
        event_list.append(e['event_type'])
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    d = collections.Counter(event_list)
    min_threshold = 10
    dict_over10 = {x: count for x, count in d.items() if count > min_threshold}
    plt.bar(dict_over10.keys(), dict_over10.values())
    plt.title("Event types")
    plt.savefig('histogram.pdf')
    plt.show()


def save_osbrowser(data_list):
    osbrowser = []
    for element in data_list:
        if element['event_type'] == 'session_start':
            dictionary = {'os': element['os'], 'browser': element['browser']}
            if 'device' in element.keys():
                dictionary['device'] = element['device']
            osbrowser.append(dictionary)
    with open("device.json", "w") as output:
        json.dump(osbrowser, output)


# plot where and when you started a discord session on a map!
# These are the keys city, country_code, timestamp
def plot_location(data_list):
    locations = []
    for element in data_list:
        if element['event_type'] == 'session_start':
            geolocator = Nominatim(user_agent="MyApp")
            if 'city' in element.keys():
                location = geolocator.geocode(element['city'])
                dictionary = {'city': element['city'], 'country_code': element['country_code'], 'latitude': location.latitude, 'longitude': location.longitude, 'timestamp': element['timestamp']}
                locations.append(dictionary)
    print("doing the map preparation")
    df = pd.DataFrame(locations)
    df['timestamp'] = df["timestamp"].str[1:20]
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S')
    # df = df.set_index('timestamp')
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = GeoDataFrame(df, geometry=geometry)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(figsize=(10, 6))
    # print("plotting map")
    gdf.plot(ax=ax, label=gdf['timestamp'], legend=True, marker='o', color='red', markersize=15)
    ax.legend(loc='lower right', fontsize=12, frameon=True)
    ax.set_axis_off()
    # print("saving map")
    plt.savefig('map.pdf')
    # print("map saved")
    plt.show()


def main():
    filepath = "__files/events-2022-00000-of-00001.json"
    data_list = read_data(filepath)
    plot_histogram(data_list)
    save_osbrowser(data_list)
    plot_location(data_list)


if __name__ == "__main__":
    main()
