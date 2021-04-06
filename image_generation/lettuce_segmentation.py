""" Script to segment lettuce from background to feed in NeRF
"""
import cv2
import glob
import numpy as np
from skimage.color import rgb2lab 
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
import os

def segment_green(foldername, resize=False):
    files = os.path.join(foldername, '*.jpg')
    savepath = os.path.join(foldername, 'segmented')
    if not files: 
        raise Exception("files not found")
    file_path = glob.glob(files)
    file_path.sort()
    print(len(file_path))
    for f in range(len(file_path)):
        img = cv2.imread(file_path[f])
        plt.imshow(img)
        plt.show()
        img2 = img.copy()
        if resize:
            img = cv2.resize(img2, 
                            (int(img2.shape[1] * 0.5), int(img2.shape[0] * 0.5)) 
                            ) #scaling by half
        img2 = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # segment out green
        mask_image = np.zeros_like(img, np.uint8)
        mask = (hsv[:,:,0]>25) & (hsv[:,:,0]<45) & (hsv[:,:,1]>95)
        mask_image[mask>0] = 255
        opened_mask = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
        opened_mask_gray = cv2.cvtColor(opened_mask, cv2.COLOR_BGR2GRAY)
        ret, labels = cv2.connectedComponents(opened_mask_gray)
        mask = np.array(labels, dtype=np.uint8)
        for label in range(1,ret):
            mask[labels == label] = 255
        mask_image = np.zeros_like(img, np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key = cv2.contourArea)
        cv2.drawContours(mask_image, [c], -1, (255,255,255), -1)
        mask_final = np.zeros_like(img, np.uint8)
        mask_final[mask_image>0] = img[mask_image>0]
        plt.imshow(mask_final)
        plt.show()
        break

        #foldername = os.path.dirname(file_path[f])
        save_filename = os.path.join(savepath, os.path.basename(file_path[f]))
        cv2.imwrite(save_filename, mask_final)

if __name__ == "__main__":
    segment_green('/home/sushmitawarrier/lettuce_nonblurred')