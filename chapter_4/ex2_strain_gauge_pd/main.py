import math
import os
import pandas as pd
import numpy as np


def process_file(filepath):
    df = pd.read_csv(filepath, sep="\t")

    column_name = list(df.columns)
    # print(column_name)
    df[['R2_1-m/m', 'R2_2-m/m', 'R2_3-m/m']].apply(lambda x : x.astype(float))

    # print(df.describe())
    df["e_1"] = 0.5 * (df["R2_1-m/m"] + df["R2_3-m/m"] + np.sqrt(2 * ((df["R2_1-m/m"] - df["R2_2-m/m"]) ** 2 + (df["R2_2-m/m"] - df["R2_3-m/m"]) ** 2)))
    df["e_2"] = 0.5 * (df["R2_1-m/m"] + df["R2_3-m/m"] - np.sqrt(
        2 * ((df["R2_1-m/m"] - df["R2_2-m/m"]) ** 2 + (df["R2_2-m/m"] - df["R2_3-m/m"]) ** 2)))

    df.to_csv('strain_out.csv', sep=';', float_format='%.9f', index=False)
    # print("wrote to csv")

if __name__ == "__main__":
    filepath = "__files/strain_gauge_rosette.csv"
    process_file(filepath)