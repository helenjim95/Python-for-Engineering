import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d, Axes3D

def main():
    df = pd.read_csv("__files/fes.csv", header=0, sep="\t",  names=['x', 'y', 'free_energy'])
    df1 = df.pivot('x', 'y').fillna(0)
    X = df1.index.values
    Y = df1.columns.get_level_values(1)
    Z = df1.values
    Xi, Yi = np.meshgrid(X, Y)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.contour(X, Y, Z)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(9))
    ax1.set(xlim=(-3, 2), ylim=(0.35, 2.0), xlabel='cv1', ylabel='cv2')
    ax2 = fig.add_subplot(122, projection="3d")
    surface = ax2.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
    fig.colorbar(surface, shrink=0.5, aspect=15,  cmap='RdBu_r')
    surface_2d = ax2.contour(X, Y, Z, linestyles="solid", offset=-1)
    ax2.set(aspect='equal', xlim=(-3, 2), ylim=(0.35, 2.0), zlim=(0, 160), xlabel='cv1', ylabel='cv2', zlabel='free energy (kJ/mol)')
    plt.savefig("plot.pdf")
    plt.show()



if __name__ == "__main__":
    main()