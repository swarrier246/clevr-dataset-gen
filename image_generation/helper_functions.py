
import glob
# import bpy, bpy_extras
import os, sys
from PIL import Image

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
    base_folder = '/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000001'
    create_gif(base_folder)