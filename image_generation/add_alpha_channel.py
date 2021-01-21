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
        image = cv2.imread(imgs[idx], cv2.IMREAD_UNCHANGED)
        if image.shape[2] != 4 :
            print("Adding alpha channel")
            # alpha mask = 0.0 -> 100% transparency; 255.0 -> no transparency
            mask = np.zeros((image.shape[0], image.shape[1]))
            mask[(image != [0,0,0]).all(-1)] = 255 # all points where image is not black
            mask = mask.astype(np.uint8)
            img_BGRA = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            img_BGRA[:,:,3] = mask
            assert (img_BGRA.shape)[2] == 4
            if not cv2.imwrite(imgs[idx], img_BGRA):
                raise Exception("file nt overwritten")
        else:
            print("Image already has 4 channels")
            print(np.sum(image[:,:,-1]))
            plt.imshow(image[:,:,-1].astype("float32"), cmap='gray')
            plt.show()




if __name__ == "__main__":
    base_folder = "/home/sushmitawarrier/clevr-dataset-gen/output/images/CLEVR_new000001/train"
    add_alpha_channel(base_folder)