import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import Figure, Axes

import numpy as np
import time

Re = 6371
color_map = ["rosybrown", "lightcoral", "darkred", "coral", "peru", "tan",
             "orange", "darkkhaki", "g", "teal", "b", "crimson"]


def plot_trajectory(s, st, sr=None, ifsave=False):
    f3: Figure = plt.figure(figsize=(8, 8))
    ax: Axes3D = f3.add_subplot(projection="3d")
    ax.scatter3D(st[0], st[1], st[2], label="Target Point", marker="*", color="red", s=50)
    ax.scatter3D(s[0, 0], s[0, 1], s[0, 2], label="Initial Point", marker="x", s=50)
    ax.plot3D(s[:, 0], s[:, 1], s[:, 2], label="real track")
    if sr is not None:
        ax.plot3D(sr[:, 0], sr[:, 1], sr[:, 2], label="reference track", c="r")
    lower = np.min(s[:, [0, 1, 2]]) * 1.1
    upper = np.max(s[:, [0, 1, 2]]) * 1.1

    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.set_zlabel("z(m)")
    # ax.set_xlim3d(lower, upper)
    # ax.set_ylim3d(lower, upper)
    # ax.set_zlim3d(lower, upper)
    ax.legend()
    if ifsave:
        f3.savefig("figure/track.png")
    else:
        plt.show()
    return f3


def plot_multi_trajectory(s: np.ndarray, st, ifsave=False):
    """
    :param s: shape must be (N, K, 6)
    :param st: shape must be one dimension
    :param ifsave: if save the figure
    :return: plt.Figure
    """
    f3: Figure = plt.figure(figsize=(8, 8))
    ax: Axes3D = f3.add_subplot(projection="3d")
    n = s.shape[0]
    for i in range(n):
        pass
    ax.scatter3D(st[0], st[1], st[2], label="Target Point", marker="*", color="red", s=50)
    ax.scatter3D(s[0, 0], s[0, 1], s[0, 2], label="Initial Point", marker="x", s=50)
    ax.plot3D(s[:, 0], s[:, 1], s[:, 2], label="real track")
    if sr is not None:
        ax.plot3D(sr[:, 0], sr[:, 1], sr[:, 2], label="reference track", c="r")
    lower = np.min(s[:, [0, 1, 2]]) * 1.1
    upper = np.max(s[:, [0, 1, 2]]) * 1.1

    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.set_zlabel("z(m)")
    # ax.set_xlim3d(lower, upper)
    # ax.set_ylim3d(lower, upper)
    # ax.set_zlim3d(lower, upper)
    ax.legend()
    if ifsave:
        f3.savefig("figure/track.png")
    else:
        plt.show()
    return f3


class track_visualization:
    def __init__(self):
        self.fig1: plt.Figure = plt.figure(1, figsize=(10, 8))
        self.ax1: Axes3D = self.fig1.add_axes((0, 0, 1, 1), projection="3d")
        self.ax1.set_xlim3d(-1500, 500)
        self.ax1.set_ylim3d(-1500, 2500)
        self.ax1.set_zlim3d(-1500, 1500)
        self.ax1.set_xlabel("x/m")
        self.ax1.set_ylabel("y/m")
        self.ax1.set_zlabel("z/m")
        self.ax1.set_box_aspect([0.5, 1, 0.75])

    def show_obstacle(self, pos: np.ndarray = None, ds: np.ndarray = None):
        if pos is None and ds is None:
            return 0
        obs_num = ds.size
        phi, theta = np.mgrid[0.0:2.0 * np.pi:40j, 0.0:np.pi:20j]
        for i in range(obs_num):
            xe = ds[i] * np.sin(theta) * np.cos(phi) + pos[i, 0]
            ye = ds[i] * np.sin(theta) * np.sin(phi) + pos[i, 1]
            ze = ds[i] * np.cos(theta) + pos[i, 2]
            self.ax1.plot_surface(xe, ye, ze, color='slateblue', alpha=0.6, edgecolors='khaki', linewidth=0.01)
        return 1

    def plot_fsat_track(self, s: np.ndarray, st: np.ndarray, show_fig=True, **kwargs):
        """
        :param s: shape must be (N, K, 6)
        :param st: shape must be one dimension
        :param show_fig: if save the figure
        :return: plt.Figure
        """
        al = kwargs["alpha"] if "alpha" in kwargs.keys() else 1.0  # set transparency
        ls = kwargs["linestyle"] if "linestyle" in kwargs.keys() else "-"  # set line style
        lw = kwargs["linewidth"] if "linewidth" in kwargs.keys() else 1.0  # set line width
        lb = kwargs["label"] if "label" in kwargs.keys() else "final trajectory"  # set line label
        n = s.shape[0]
        self.ax1.scatter3D(st[0], st[1], st[2], marker="o", color=color_map[0], label="target pos")
        for i in range(n):
            si = s[i]
            if i == 0:
                self.ax1.scatter3D(si[0, 0], si[0, 1], si[0, 2], marker="d", color=color_map[i], label="initial pos")
                self.ax1.plot3D(si[:, 0], si[:, 1], si[:, 2],  # plot trajectory
                                c=color_map[i], linewidth=lw, linestyle=ls, alpha=al, label=lb)
            else:
                self.ax1.scatter3D(si[0, 0], si[0, 1], si[0, 2], marker="d", color=color_map[i % len(color_map)])
                self.ax1.plot3D(si[:, 0], si[:, 1], si[:, 2], c=color_map[i % len(color_map)], linewidth=lw,
                                linestyle=ls, alpha=al)
        if show_fig:
            self.ax1.scatter3D(0, 0, 0, marker="*", label="chief sat", color="r")
            hs, las = self.ax1.get_legend_handles_labels()
            las_uni = list(set(las))
            hs_uni = [hs[las.index(la)] for la in las_uni]
            self.ax1.legend(hs_uni, las_uni)
            plt.show()
