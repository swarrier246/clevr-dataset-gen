import matplotlib.pyplot as plt
import json
import glob
import os, sys, math #, cv2
import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot


def plot_cameras(transform_matrix_filename: str):
    """
    Takes in transforms.json file and plot camera positions to ascertain validity
    """
    pos_x, pos_y, pos_z = [], [], []
    with open(transform_matrix_filename) as f:
        transforms = json.load(f)
    for i in transforms['frames']:
        T = i['transform_matrix']
        # append pose
        pos_x.append(T[0][3])
        pos_y.append(T[1][3])
        pos_z.append(T[2][3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos_x, pos_y, pos_z)
    ax.scatter(0, 0, 0, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

# c2w generated based on nerf's load_blender.py
trans_t = lambda t : np.array([[1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,math.cos(phi),-math.sin(phi),0],
    [0,math.sin(phi), math.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.array([
    [math.cos(th),0,-math.sin(th),0],
    [0,1,0,0],
    [math.sin(th),0, math.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def plot_render_poses_spiral_nerf():
    """
    Plots the positions of the cameras in the render_poses attribute of load_llff_data in nerf
    """
    pass

def plot_render_poses_nerf():
    """
    Plots the positions of the cameras in the render_poses attribute of load_blender_data in nerf
    """
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    print("render poses shape:", render_poses.shape)
    pos_x, pos_y, pos_z = [], [], []
    for i in range(render_poses.shape[0]):
        pos_x.append(render_poses[i][0][3])
        pos_y.append(render_poses[i][1][3])
        pos_z.append(render_poses[i][2][3])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos_x, pos_y, pos_z)
    ax.scatter(0, 0, 0, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def plot_cameras_gtsam(transform_matrix_filename: str):
    set_cams = np.load('/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000000/set_cams.npz')
    with open(transform_matrix_filename) as f:
        transforms = json.load(f)
    figure_number = 0
    fig = plt.figure(figure_number)
    axes = fig.gca(projection='3d')
    plt.cla()
    # Plot cameras
    idx = 0
    for i in range(len(set_cams['set_rot'])):
        print(set_cams['set_rot'][i])
        print("i", i)
        rot = set_cams['set_rot'][i]
        t = set_cams['set_loc'][i]
        rot_3 = gtsam.Rot3.RzRyRx(rot)
        pose_i = gtsam.Pose3(rot_3, t)
        if idx % 2 == 0:
            gtsam_plot.plot_pose3(figure_number, pose_i, 1)
        break
        idx += 1

    for i in transforms['frames']:
        T = i['transform_matrix']
        pose_i = gtsam.Pose3(T)
        #print("T", T)
        #print("pose_i",pose_i)
        #print(set_cams['set_loc'][0])
        #print(set_cams['set_rot'][0])
        # Reference pose
        # trying to look along x-axis, 0.2m above ground plane
        # x forward, y left, z up (x-y is the ground plane)
        upright = gtsam.Rot3.Ypr(-np.pi / 2, 0., -np.pi / 2)
        pose_j = gtsam.Pose3(upright, gtsam.Point3(0, 0, 0.2))
        axis_length = 1
        # plotting alternate poses
        if idx%2 == 0:
            # gtsam_plot.plot_pose3(figure_number, pose_i, axis_length)
            gtsam_plot.plot_pose3(figure_number, pose_j, axis_length)
        idx += 1
    x_axe, y_axe, z_axe = 10,10,10
    # Draw
    axes.set_xlim3d(-x_axe, x_axe)
    axes.set_ylim3d(-y_axe, y_axe)
    axes.set_zlim3d(0, z_axe)
    plt.legend()
    plt.show()


        

if __name__ == "__main__":
    # plot_cameras('/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000000/transforms_nerf.json')
    plot_cameras_gtsam('/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000000/transforms_train_4.json')
    # plot_render_poses_nerf()