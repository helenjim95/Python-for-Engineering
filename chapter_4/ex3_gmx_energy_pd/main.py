import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def process_file(filepath):
    df = pd.read_csv(filepath, sep="  ", header=None)
    df.columns = ["index", "value"]
    df["index"].apply(lambda x: int(float(x)))
    df["value"].apply(lambda x: float(x))
    WINDOW_SIZE = 25

    df.loc[df["index"] >= WINDOW_SIZE, "moving_average"] = df["value"].rolling(window=WINDOW_SIZE).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=WINDOW_SIZE)
    moving_average_less25 = df["value"].rolling(window=indexer, min_periods=1).mean()
    df["moving_average_less25"] = moving_average_less25
    df_new = df["moving_average"].combine(df["moving_average_less25"], lambda x, y: x if pd.notnull(x) else y)
    # df["moving_average_less25"]
    # for i in range(WINDOW_SIZE, len(df["value"]) - WINDOW_SIZE + 1):
    #     df["moving_average"] = df.rolling(window=WINDOW_SIZE).mean()

    df_new.to_csv('average.csv', float_format='%.2f', index=False, header=False)
    # print("wrote to csv")
    plt.plot(range(len(df["value"])), df["value"])
    plt.plot(range(len(df_new)), df_new)
    plt.show()
    plt.savefig("plot.pdf")

if __name__ == "__main__":
    filepath = "__files/zif-nvt.csv"
    process_file(filepath)