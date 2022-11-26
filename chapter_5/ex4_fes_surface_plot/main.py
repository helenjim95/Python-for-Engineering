import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import cm, ticker
from matplotlib.ticker import MultipleLocator


def contour():
    pass

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
    ax1.set_xlabel("cv1")
    ax1.set_ylabel("cv2")
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_zlabel("free energy (kJ/mol)")
    ax2.set_aspect('equal')
    ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax2.set_xlim([-3, 2])
    ax2.set_ylim([0.25, 2.0])
    ax2.set_zlim([160, 0])
    np.random.seed(19680801)

    # TODO: How to place colorbar
    cmaps = ['RdBu_r', 'viridis']
    pcm = ax2.pcolormesh(np.random.random((20, 20)), cmap=cmaps[1])
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    fig.colorbar(pcm, ax=ax2)
    plt.savefig("plot.pdf")
    plt.show()

if __name__ == "__main__":
    main()