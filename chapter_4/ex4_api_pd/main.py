import datetime
import json
import urllib
import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt

def download_bme280(id, dates: list) -> pd.DataFrame:
#     dates: list of already formatted datestrings
    dfs = []
#     function iterates through a list of already formatted datestrings,
    for date in dates:
        try:
            url = f"https://archive.sensor.community/{date}/{date}_bme280_sensor_{id}.csv"
            # print(url)
            s = requests.get(url).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8')), sep=";")
            dfs.append(df)
        except urllib.error.HTTPError as e:
            print("url invalid")
    #     Only if there is no single frame for all the dates given, function should raise a FileNotFoundError.
    if len(dfs) == 0:
        raise FileNotFoundError
    else:
        #     and concatenates all the data found on the API into one dataframe for a specific BME280-sensor number
        #     (so, if a date isn’t there, it should just continue).
        #     The returned dataframe doesn’t use a column as the index-column wrongly and that the
        #     “default” is cleared on concatenation ignore_index=True
        big_frame = pd.concat(dfs, ignore_index=True)
        #     sort the downloads by the datestring
        big_frame.sort_values(by=['timestamp'], inplace=True)
        # for col in big_frame.columns:
        #     print(col)
        return big_frame

# converts the timestamp-column to an actual timestamp: pd.to_datetime(df[’timestamp’])
def conv_ts(sensordf) -> pd.DataFrame:
    sensordf["timestamp"] = pd.to_datetime(sensordf["timestamp"], yearfirst=True)
    return sensordf

def select_time(sensordf, after, before) -> pd.DataFrame:
    # selects a speciﬁc range from a dataframe, inbetween (so no <=, but <) two “timestrings”: boolean series comparing the ’timestamp’-column
    # use np.logical_and to get a column to select from the df for two criteria.
    # Return the subselection.
    df_inrage = sensordf[np.logical_and(after < sensordf["timestamp"], sensordf["timestamp"] < before)]
    # df_inrage.to_csv('output.csv', index=False)
    return df_inrage


def filter_df(sensordf) -> dict:
    min_T = sensordf["temperature"].min()
    min_p = sensordf["pressure"].min()
    min_T_id = sensordf.loc[sensordf["temperature"] == min_T, 'sensor_id'].iloc[0]
    min_p_id = sensordf.loc[sensordf["pressure"] == min_p, 'sensor_id'].iloc[0]
    output_dict = {"min_T": min_T, "min_p": min_p, "min_T_id": int(min_T_id), "min_p_id": int(min_p_id)}
    return output_dict

if __name__ == "__main__":
    id1 = 10881
    id2 = 11036
    id3 = 11077
    id4 = 11114
    id_list = [id1, id2, id3, id4]
    dates = [d.strftime('%Y-%m-%d') for d in pd.date_range('2022-01-28', '2022-02-23')]
    dfs = []
    for id in id_list:
        sensordf = download_bme280(id, dates)
        sensordf_convts = conv_ts(sensordf)
        after = datetime.datetime(year=2022, month=1, day=28, hour=5, minute=13)
        before = datetime.datetime(year=2022, month=2, day=3, hour=12, minute=31)
        df_inrange = select_time(sensordf, after, before)
        dfs.append(df_inrange)
    big_frame = pd.concat(dfs, ignore_index=True)
    # big_frame.to_csv('output.csv', index=False)
    output = filter_df(big_frame)
    with open('extrema.json', 'w') as file:
        json.dump(output, file)

    # plot temperature curve for 2022-01-01 for sensor 10881 & 11036
    id_list_2 = [10881, 11036]
    dates_jan_first = [d.strftime('%Y-%m-%d') for d in pd.date_range('2022-01-01', '2022-01-02')]
    dfs_jan = []
    for id in id_list_2:
        sensordf = download_bme280(id, dates_jan_first)
        sensordf_convts = conv_ts(sensordf)
        df_jan_first = select_time(sensordf, datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0), datetime.datetime(year=2022, month=1, day=1, hour=23, minute=59))
        dfs_jan.append(df_jan_first)
    big_frame_jan = pd.concat(dfs_jan, ignore_index=True)
    # big_frame_jan.to_csv('output_jan.csv', index=False)
    df_10881 = big_frame_jan.apply(lambda row: row[big_frame_jan["sensor_id"].isin([10881])])
    df_11036 = big_frame_jan.apply(lambda row: row[big_frame_jan["sensor_id"].isin([11036])])
    plt.plot(df_10881["timestamp"], df_10881["temperature"], label="10881")
    plt.plot(df_11036["timestamp"], df_11036["temperature"], label="11036")
    plt.xlabel("Timestamp")
    plt.ylabel("Temperature")
    plt.legend()
    plt.savefig("sensors.pdf")
    plt.show()
