import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
from urllib.error import HTTPError
import os
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def download_data():
    states = ["BY", "NW", "SN", "TH", "DE-total"]
    for state in states:
        try:
            url = f"https://raw.githubusercontent.com/entorb/COVID-19-Coronavirus-German-Regions/master/data/de-states/de-state-{state}.tsv"
            s = requests.get(url).content
            df = pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\t')
            df.to_csv(f'__files/{state}.csv')
        except HTTPError as e:
            print("url invalid")

def concat_files():
    df = pd.DataFrame()
    files = os.listdir("__files")
    for file in files:
        # print(file)
        df_new = pd.read_csv(f"__files/{file}")
        df_new['from_file'] = file.replace('.csv', "")
        df = df.append(df_new, ignore_index=True)
    df.to_csv("aggregate.csv")
    return df
    # print("created aggregate.csv")

def conv_ts(df):
    df["Date"] = pd.to_datetime(df["Date"], yearfirst=True)
    print("date is converted")
    return df

def annot_max(df, ax=None):
    x = df["Date"]
    y = df["Cases_Last_Week_Per_Million"]
    xmax = x[np.argmax(y)]
    ymax = y.max()
    max_state = df["from_file"][np.argmax(y)]
    text = f"Maximum n = {ymax} in {max_state} @{xmax.strftime('%Y-%m-%d')}"
    if not ax:
        ax = plt.gca()
    # bbox_props = dict(boxstyle="square,pad=0.", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.97, 0.97), **kw)


def plot(df):
    df_states = df[df["from_file"] != "DE-total"]
    df_germany = df[df["from_file"] == "DE-total"]
    fig = plt.figure()
    axes = fig.add_subplot()
    # plot the states
    for state, group in df_states.groupby('from_file'):
        axes.plot(group['Date'], group['Cases_Last_Week_Per_Million'], label=group["from_file"])
    axes.set_xlabel("Date")
    axes.set_ylabel("n/(week . million)")
    axes.legend(["BY", "NW", "SN", "TH"], loc='upper left')
    axes.grid(True)
    axes.set_title('7 day incidence/Mio of Covid-cases')
    # Set the y-Axis to scale logarithmically.
    plt.yscale('log')
    # Identify the global maximum and annotate it with an arrow pointing a the
    # maximum-point.
    annot_max(df, ax=axes)
    # Additionally, in an inset, plot the same column for the whole of Germany.
    inset_ax = inset_axes(axes, "100%", "100%", loc="lower left", bbox_to_anchor=(0.47, 0.1, 0.48, 0.15),
                          bbox_transform=axes.transAxes, borderpad=0)
    inset_ax.set_title('Incidence in whole Germany')
    # df_germany.plot(x='Date', y='Cases_Last_Week_Per_Million', ax=inset_ax)
    inset_ax.plot(df_germany['Date'], df_germany['Cases_Last_Week_Per_Million'], label=df_germany["from_file"])
    ax = plt.gca()
    ax.set_frame_on(False)
    plt.yscale('log')
    plt.savefig("plot.pdf")
    plt.show()
    print("plot is shown")


if __name__ == "__main__":
    download_data()
    df = concat_files()
    df = conv_ts(df)
    plot(df)