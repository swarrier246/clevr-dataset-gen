import json
import glob
import os, sys, math #, cv2
import numpy as np
from PIL import Image
from utils import ObjLoader
from functools import partial
from enum import Enum
import bpy
import bpy_extras
from mathutils import Matrix
from mathutils import Vector
# from render_images_custom import config_cam



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


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------
# # Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model

def get_calibration_matrix_K_from_blender(camd):
    """
    Author: R Fabbri
    Code taken from https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels    
    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K 
# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction

def get_3x4_RT_matrix_from_blender(cam):
    """
    Author: R Fabbri
    Code taken from https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))    
    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()    
    
    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    # T_world2bcam = -1*R_world2bcam @ location    # Build the coordinate transform matrix from world to computer vision camera
    T_world2bcam = -1*R_world2bcam * location # reqd * instead of # for this Blender version
    # NOTE: Use * instead of @ here for older versions of Blender
    # TODO: detect Blender version
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT
    
def get_3x4_P_matrix_from_blender(cam):
    """
    Author: R. Fabbri
    Code taken from https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    Returns:
        Projection matrix
        Calibration matrix
        RT (Rot and translation) matrices
    """
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    # return K@RT, K, RT
    return np.array(K*RT), K, RT


def get_camera_matrix(az=0, ele=0, dist=0):
    """
    Author: Amit Raj, Sushmita Warrier
    Camera-to-World transformation matrix from azimuth and elevation
    Args: az(float) - azimuth angle
    ele(float) - elevation angle
    dist(float) - distance between camera and object
    Returns: T(np.array) - 4x4 transformation matrix (same as C, but homogeneous)
    """
    R0 = np.zeros([3, 3])
    C = np.zeros([3, 4])  # camera matrix
    T = np.zeros([4,4]) # transformation matrix
    # R0[0, 1] = 1
    # R0[1, 0] = -1
    # R0[2, 2] = 1

    # trying something
    R0[0, 1] = -1
    R0[1, 0] = -1
    R0[2, 2] = -1

    az = az * math.pi / 180
    ele = ele * math.pi / 180
    cos = np.cos
    sin = np.sin
    R_ele = np.array(
        [[1, 0, 0], [0, cos(ele), -sin(ele)], [0, sin(ele), cos(ele)]]).astype('float32')
    R_az = np.array(
        [[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]]).astype('float32')

    # trying something
    # R_ele = np.array(
    #     [[1, 0, 0], [0, cos(-ele), -sin(-ele)], [0, sin(-ele), cos(-ele)]]).astype('float32')
    # R_az = np.array(
    #     [[cos(-az), -sin(-az), 0], [sin(-az), cos(-az), 0], [0, 0, 1]]).astype('float32')
    R_rot = np.matmul(R_az, R_ele)
    R_all = np.matmul(R_rot, R0)
    C[:3, :3] = R_all

    t = np.array([dist*cos(ele)*sin(az),dist*cos(ele)*cos(az),dist*sin(ele)])
    C[:, 3] = t
    T[:3, :] = C
    T[3, 3] = 1.0
    return T
   

def save_scene_data(base_folder, focal_length, file_name='transforms_train', nb_images=None):
    """
    Save scene data to npz file for ray casting visualization
    Args:
        base_folder: Path to scene data
        focal_length: Focal length of camera (hardcoded)
        nb_images: Nb of images we want to ray cast for
        
    """
    img_list = []
    focal_list = []
    extrinsics_list = []
    vertices_list = []
    images_folderpath = os.path.join(base_folder,'images', 'CLEVR_new000000')
    obj_folderpath = os.path.join(base_folder,'objfiles')
    search_path = glob.glob(os.path.join(images_folderpath, '*.png'))
    search_path.sort()
    if nb_images == None:
        nb_images = len(search_path) # process all images if nb not specified
    if len(search_path) == 0:
        raise Exception('no images found. Check file path')
    for idx, f in enumerate(search_path):
        if idx < nb_images:
            im = Image.open(f)
            im = np.array(im)
            img_list.append(im)
            # focal length is being hardcoded here, but can be extracted from the render_images_custom file using camera.data.lens
            focal_list.append(focal_length)
    transforms_path = os.path.join(images_folderpath, '{}.json'.format(file_name))
    with open(transforms_path) as f:
        transforms = json.load(f)
    for i in transforms['frames']:
        extrinsics_list.append(i['transform_matrix'])
        #print("extrinscis:", i['transform_matrix'])
        if len(extrinsics_list) == nb_images:
            break
    # Save vertices info
    obj_search_path = glob.glob(os.path.join(obj_folderpath, '*.obj'))
    obj_search_path.sort()
    for idx, filename in enumerate(obj_search_path):        
        if idx < nb_images:
            # print("vertices",ObjLoader(filename).vertices)
            vertices_list.append(ObjLoader(filename).vertices)

    save_path = os.path.join(base_folder, 'images', 'lettuce_scene000000_{}.npz'.format(file_name))
    np.savez(save_path, images=img_list, focal=focal_list, poses=extrinsics_list, verts=vertices_list)
    print("File saved")
    

class FunctionCalls(Enum):
    get_c2w_nerf_based = partial(pose_spherical)
    get_c2w_from_blender = partial(get_3x4_P_matrix_from_blender)
    get_c2w = partial(get_camera_matrix)

    def __call__(self, *args):
        return self.value(*args)

def generate_transforms_file(base_folder, codeFlag=FunctionCalls.get_c2w_nerf_based, nb_images: int =50, file_name: str = 'transforms_train'):
    """
    FOR DEBUGGING ONLY. Create different transforms files which can be read by function save_scene_data() to visualize ray casting
    Args:
        nb_images: number of images per scene
        codeFlag: 
    """
    base_dist = 9.915478183682518 # hardcoded, value taken from code output
    angle_x = 0.8575560450553894 # hardcoded, value taken from code output
    t_list = np.linspace(0,1,nb_images)

    transforms = dict()
    transforms.update({"camera_angle_x": angle_x})
    # Generate frames (list of dictionaries) for transforms. json file (nerf)
    frames = []
    for j in range(nb_images):
        img_name = './{}'.format(str(j).zfill(6))
        t = t_list[j]
        azimuth = 180 + (-180 * t + (1 - t) * 180)  # range of azimuth: 0-360 deg
        elevation = 15 * t + (1 - t) * 75    # range of elevation: 15-75 deg    
        dist = base_dist   

        if codeFlag == FunctionCalls.get_c2w:
            transformation_matrix = get_camera_matrix(azimuth, elevation, dist)
        elif codeFlag == FunctionCalls.get_c2w_from_blender:
            transformation_matrix, _, _ = get_3x4_P_matrix_from_blender(bpy.data.objects["Camera"])
        elif codeFlag == FunctionCalls.get_c2w_nerf_based:
            transformation_matrix = pose_spherical(elevation, azimuth, dist)
        else:
            print("This type is not supported")


        # create dictionary to append to frames
        # img_name = img.split('.')
        # img_name = img_name[:-1][-1]
        scene_tranformation_dict = dict()
        scene_tranformation_dict = {"file_path": img_name, "rotation": np.random.random(), "transform_matrix": transformation_matrix.tolist()}
        frames.append(scene_tranformation_dict)

    # Update transforms dict with frame information and dump into json file
    transforms.update({"frames": frames})

    # NeRF specific work
    train_path = os.path.join(base_folder, 'images', 'CLEVR_new000000')
    if not os.path.exists(train_path):
        train_path = os.mkdir(train_path)
    temp = '{}.json'.format(file_name)
    train_file = os.path.join(train_path, temp)
    with open(train_file, 'w') as outfile:
        json.dump(transforms, outfile)


def create_gif(base_folder):
    """
    Create quick gif to visualize trajectory of image captures in scene
    """
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
    base_folder = '/home/sushmitawarrier/clevr-dataset-gen/output/GPU_data_rsynced/CLEVR_new000000/'
    #create_gif(base_folder)
    transforms_file_name = 'transforms_test'

    generate_transforms_file(base_folder, codeFlag=FunctionCalls.get_c2w_nerf_based, nb_images=50, file_name=transforms_file_name)

    # save_scene_data(base_folder, 35.0, file_name=transforms_file_name, nb_images=5)