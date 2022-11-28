import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import distance_matrix

number_of_atoms = 0


def read_file():
    filename = "aspirin.xyz"
    global number_of_atoms
    with open(f"__files/{filename}") as xyz:
        number_of_atoms = int(xyz.readline())
        comment = xyz.readline()
        molecule = pd.read_table(f"__files/{filename}", skiprows=2, delim_whitespace=True,
                                 names=['atom', 'x', 'y', 'z'])
        molecule['coordinates'] = molecule.loc[:, ["x", "y", "z"]].apply(lambda s: s.to_numpy(), axis=1)
    return molecule


# TODO: use this method
def determine_bond(molecule):
    min_value = 1.6
    for i in range(1, number_of_atoms):
        # TODO: extract p1, p2 from cooridnates?
        distance_matrix_ = distance_matrix(molecule.loc[i, "coordinates"], molecule.loc[i + 1, "coordinates"])
        molecule.at[i, "bond"] = np.argwhere(distance_matrix_ < min_value)
    # Return (N,2)-array including all indices
    # of the distance matrix for which the distance is sufﬁcient for a “bond”.
    return molecule
# give back a (N,2)-array including all indices
# of the distance matrix for which the distance is sufﬁcient for a “bond”. Using
# these tuples, you can easily extract the start and end-position of the bond.


def plot_atom(df):
    fig = plt.figure()
    axes = fig.add_subplot(projection="3d")
    radius_c = 70
    radius_h = 25
    radius_o = 60
    colors = np.where(df["atom"] == "C", 'r', '-')
    colors[df["atom"] == "H"] = 'g'
    colors[df["atom"] == "O"] = 'b'
    shapes = np.where(df["atom"] == "C", math.sqrt(radius_c), 0)
    shapes[df["atom"] == "H"] = math.sqrt(radius_h)
    shapes[df["atom"] == "O"] = math.sqrt(radius_o)
    groups = df.groupby('atom')
    for name, group in groups:
        axes.scatter(df.x, df.y, df.z, c=colors, s=shapes, label=name)
        plt.show()

# conns = (N, 2, 3)-array including for all N-bonds the start and endpoint.
def plot_bond(conns):
    fig = plt.figure()
    axes = fig.add_subplot(projection="3d")
    conn_lines = Line3DCollection(conns,
                                  edgecolor="gray",
                                  linestyle="solid",
                                  linewidth=8)
    axes.add_collection3d(conn_lines)
    axes.legend()
    plt.savefig("plot.pdf")
    plt.show()

def main():
    molecule = read_file()
    print(molecule.head())
    plot_atom(molecule)
    # molecule = determine_bond(molecule)
    print(molecule.head())
    conns = [[[]]]
    plot_bond(conns)


if __name__ == "__main__":
    main()