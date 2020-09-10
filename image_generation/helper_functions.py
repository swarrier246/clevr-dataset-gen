import json
import glob
# import bpy, bpy_extras
import os, sys, math
import numpy as np
#from PIL import Image

def get_camera_matrix(az=0, ele=0, dist=0):
    """
    Camera-to-World transformation matrix from azimuth and elevation
    Args: az(float) - azimuth angle
    ele(float) - elevation angle
    dist(float) - distance between camera and object
    Returns: T(np.array) - 4x4 transformation matrix (same as C, but homogeneous)
    """
    R0 = np.zeros([3, 3])
    C = np.zeros([3, 4])  # camera matrix
    T = np.zeros([4,4]) # transformation matrix
    R0[0, 1] = 1
    R0[1, 0] = -1
    R0[2, 2] = 1
    az = az * math.pi / 180
    ele = ele * math.pi / 180
    cos = np.cos
    sin = np.sin
    R_ele = np.array(
        [[1, 0, 0], [0, cos(ele), -sin(ele)], [0, sin(ele), cos(ele)]]).astype('float32')
    R_az = np.array(
        [[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]]).astype('float32')
    R_rot = np.matmul(R_az, R_ele)
    R_all = np.matmul(R_rot, R0)
    C[:3, :3] = R_all

    t = np.array([dist*cos(ele)*sin(az),dist*cos(ele)*cos(az),dist*sin(ele)])
    C[:, 3] = t
    T[:3, :] = C
    T[3, 3] = 1.0
    return T
   




def create_gif(base_folder):
    '''
    Create quick gif to visualize trajectory of image captures in scene
    '''
    img_list = []
    search_path = glob.glob(os.path.join(base_folder, '*.png'))
    search_path.sort()
    for f in search_path:
        im = Image.open(f)
        img_list.append(im)
    save_file = os.path.join(base_folder, 'animated_gif.gif')
    img_list[0].save(save_file,
               save_all=True, append_images=img_list[1:], optimize=False, duration=120, loop=0)

if __name__ == "__main__":
    base_folder = '/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000000'
    create_gif(base_folder)