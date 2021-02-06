import unittest

from numpy.testing._private.utils import assert_array_almost_equal
from helper_functions import pose_spherical, config_cam

import numpy as np
import matplotlib
# removes the need for tkinter, which can't be pip installed
# agg is a non-GUI backend; plt.show will be disabled
matplotlib.use('agg')
import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
import gtsam


class TestPoses(unittest.TestCase):
    """
    class to check if poses are correct
    """
    # World coordinate system: right handed. X forward, Y left, Z up
    # Create a reference object, facing along X-axis and elevation of 0 deg. 
    az = 90
    ele = 0
    tilt = 0
    dist = 3
    # get location and rotation vector from az,ele and dist values
    loc, rot = config_cam(az, ele, dist)
    # compute transformation matrix using pose_spherical function
    computed_T = pose_spherical(az, ele, dist)
    pose_i = gtsam.Pose3(computed_T)
    # Construct pose using rotation and location vectors
    pose_j = gtsam.Pose3(gtsam.Rot3.RzRyRx(rot), loc)
    # Compares rotation vector from constructed pose with rotation vector initially computed 
    rot_p = pose_j.rotation()
    rot_from_ele = rot_p.xyz()
    # Returns true
    assert_array_almost_equal(rot_from_ele, rot)

    figure_number = 0
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()

    upright = gtsam.Rot3.Ypr(-np.pi / 2, 0., -np.pi / 2)
    pose_j = gtsam.Pose3(upright, gtsam.Point3(0, 0, 0.2))
    axis_length = 1
    gtsam_plot.plot_pose3(figure_number, pose_i, axis_length)
    gtsam_plot.plot_pose3(figure_number, pose_j, axis_length)
    x_axe, y_axe, z_axe = 10,10,10
    # Draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(0, z_axe)
    plt.legend()
    #plt.show()
    filename = "/home/sushmitawarrier/results/test_pose_clevr3.png"
    plt.savefig(filename)


if __name__ == '__main__':
    unittest.main()