import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def add_alpha_channel(foldername):
    """
    Add alpha channel to all images in folder
    """
    search_path = os.path.join(foldername, "*.png")
    imgs = glob.glob(search_path)
    for idx in range(len(imgs)):
        image = cv2.imread(imgs[idx])
        # alpha mask = 0.0 -> 100% transparency; 255.0 -> no transparency
        mask = (image != [0,0,0]).all(-1) # all points where image is not black
        mask = mask.astype(np.uint8)
        b, g, r = cv2.split(image) # need to split into individual channels before merging alpha mask
        img_BGRA = cv2.merge((b, g, r, mask))
        assert (img_BGRA.shape)[2] == 4
        cv2.imwrite(imgs[idx], img_BGRA)




if __name__ == "__main__":
    base_folder = "/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000000/train"

    add_alpha_channel(base_folder)