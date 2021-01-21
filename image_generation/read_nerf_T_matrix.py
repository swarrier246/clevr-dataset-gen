"""
Read transformation matrices from Nerf's sample dataset, provided here: https://github.com/bmild/nerf
The transform file has been downloaded for the lego dataset, and will be used to generate our datasets as well.
User can change filename to read one of transforms_train, transforms_val, or transforms_test
"""
import numpy as np
import json, glob, os
import gtsam

def read_T_matrix(filepath):
    base_folder = '/home/sushmitawarrier/clevr-dataset-gen/output/'
    # extract the type of file (train, val, test)
    temp = filepath.split('_')[-1]
    filetype = temp.split('.')[0]
    loc_array, rot_array = [], []
    with open(filepath) as f:
        transforms = json.load(f)
    for i in transforms['frames']:
        T = i['transform_matrix']
        pose = gtsam.Pose3(T)
        rot = pose.rotation()
        loc = np.array(pose.translation())
        rot_euler = rot.xyz()
        loc_array.append(loc)
        rot_array.append(rot_euler)
    savefile_path = os.path.join(base_folder, 'images', 'CLEVR_new000001', 'nerf_cams_{}.npz'.format(filetype))
    np.savez(savefile_path, set_loc=loc_array, set_rot=rot_array)

if __name__ == "__main__":
    filetypes = ['train', 'val', 'test']
    # save locations set for 
    for i in filetypes:
        read_T_matrix('/home/sushmitawarrier/nerf_sample_datasets/lego/transforms_{}.json'.format(i))