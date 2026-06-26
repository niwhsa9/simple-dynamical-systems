import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import matplotlib.pyplot as plt
import pyvista as pv
import sys

#def draw_pose(ax, pos, quat, s=1.0):
#    r = np.array(R.from_quat(quat).as_matrix())
#    x_ax = np.array([pos, pos + r[:, 0] * s])
#    y_ax = np.array([pos, pos + r[:, 1] * s])
#    z_ax = np.array([pos, pos + r[:, 2] * s])
#    ax.plot(x_ax[:, 0], x_ax[:, 1], x_ax[:, 2], "r")
#    ax.plot(y_ax[:, 0], y_ax[:, 1], y_ax[:, 2], "g")
#    ax.plot(z_ax[:, 0], z_ax[:, 1], z_ax[:, 2], "b")

def draw_pose(plotter, pos, quat, scale=0.1):
    Rm = R.from_quat(quat).as_matrix()

    colors = ["red", "green", "blue"]
    for i in range(3):
        start = pos
        direction = Rm[:, i]

        arrow = pv.Arrow(
            start=start,
            direction=direction,
            scale=scale,
            tip_length=0.25,
            tip_radius=0.2 * scale * 10,
            shaft_radius=0.1 * scale * 10,
        )

        plotter.add_mesh(arrow, color=colors[i])


def vis_from_df(df : pd.DataFrame):
    ax = plt.figure(1).add_subplot(projection="3d")


    #draw_pose(ax, np.array([0, 0, 0]), [0, 0, 0, 1])
    xyz = df[["x", "y", "z"]].to_numpy()
    quat = df[["qx", "qy", "qz", "qw"]].to_numpy() / np.linalg.norm(
        df[["qx", "qy", "qz", "qw"]].to_numpy(), axis=1, keepdims=True
    )

    plotter = pv.Plotter()
    draw_pose(plotter, np.array([0, 0, 0]), [0, 0, 0, 1], scale=0.2)
    for p, r in list(zip(xyz, quat))[::]:
        #draw_pose(ax, p, r, s=0.1)
        draw_pose(plotter, p, r, scale=0.1)



    plotter.add_axes()
    plotter.show_grid()
    #plotter.show()
    #plt.axis("equal")
    #ax.set_box_aspect((np.ptp(xyz[:, 0]), np.ptp(xyz[:, 1]), np.ptp(xyz[:, 2])))  # aspect ratio is 1:1:1 in data space


    plt.figure()
    throttle_keys = [key for key in df.keys() if "throttle" in key]
    for i, key in enumerate(throttle_keys):
        n = len(throttle_keys)
        cols = int(np.ceil(np.sqrt(n)))  # roughly square grid
        rows = int(np.ceil(n / cols))
        plt.subplot(cols, rows, i + 1)
        plt.plot(df[key].to_numpy(), label=key)
        plt.title(f"Throttle {key}")
    plt.suptitle("Throttles")

    # Thrust
    plt.figure()
    thrust_keys = [
        key for key in df.keys() if ("prop" in key and "throttle" not in key)
    ]
    print(thrust_keys)
    for i, key in enumerate(thrust_keys):
        n = len(thrust_keys)
        cols = int(np.ceil(np.sqrt(n)))  # roughly square grid
        rows = int(np.ceil(n / cols))
        plt.subplot(cols, rows, i + 1)
        plt.plot(df[key].to_numpy(), label=key)
        plt.title(f"Thrust (N) {key}")
    plt.suptitle("Thrust")

    # Position
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(df["x"].to_numpy(), label="x")
    plt.title("x (m)")
    plt.subplot(3, 1, 2)
    plt.plot(df["y"].to_numpy(), label="y")
    plt.title("y (m)")
    plt.subplot(3, 1, 3)
    plt.plot(df["z"].to_numpy(), label="z")
    plt.title("z (m)")

    # Velocity
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(df["x_dot"].to_numpy(), label="vx")
    plt.title("x dot m/s")
    plt.subplot(3, 1, 2)
    plt.plot(df["y_dot"].to_numpy(), label="vy")
    plt.title("y dot m/s")
    plt.subplot(3, 1, 3)
    plt.plot(df["z_dot"].to_numpy(), label="vz")
    plt.title("z dot m/s")
    plt.suptitle("Velocity")

    # Angles
    plt.figure()
    plt.subplot(3, 1, 1)
    angles = R.from_quat(quat).as_euler("ZYX", degrees=True)
    plt.plot(angles[:, 0], label="yaw")
    plt.title("yaw (deg)")
    plt.subplot(3, 1, 2)
    plt.plot(angles[:, 1], label="pitch")
    plt.title("pitch (deg)")
    plt.subplot(3, 1, 3)
    plt.plot(angles[:, 2], label="roll")
    plt.title("roll (deg)")


    # Angular rate
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(df["roll_dot"].to_numpy(), label='roll dot')
    plt.title("roll dot (rad/s)")
    plt.subplot(3, 1, 2)
    plt.plot(df["pitch_dot"].to_numpy(), label='pitch dot')
    plt.title("pitch dot (rad/s)")
    plt.subplot(3, 1, 3)
    plt.plot(df["yaw_dot"].to_numpy(), label='yaw dot')
    plt.title("yaw dot (rad/s)")
    plt.suptitle("Angular Rates")

    # show surface deflections
    plt.figure()
    for joint in [col for col in df.columns if "joint" in col]:
        plt.plot(df[joint].to_numpy(), label=joint)
    plt.title("Surface Deflections")  
    plt.legend()


    #plt.show()
    return plotter

if __name__ == "__main__":
    # get first argument as csv file path
    #df = pd.read_csv(sys.argv[1])
    df = pd.read_csv(sys.stdin)
    plotter = vis_from_df(df)
    plotter.show()
    plt.show()