import cv2
import glob
import numpy as np
import os
import cv2.aruco as aruco
import matplotlib.pyplot as plt

def get_aruco_dict():
    """
    Detect what type of marker has been used in the images
    """
    # use only 1 image, which has all markers
    image = cv2.imread('/home/sushmitawarrier/Desktop/datasets/lettuce_real/lettuce-rework/images/DSC_0265.png')
    # define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    list_markers = []
    for (arucoName, arucoDict) in ARUCO_DICT.items():
        # load the ArUCo dictionary, grab the ArUCo parameters, and
        # attempt to detect the markers for the current dictionary
        arucoDict = cv2.aruco.Dictionary_get(arucoDict)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            image, arucoDict, parameters=arucoParams)
        
        # if at least one ArUco marker was detected display the ArUco
        # name to our terminal
        if len(corners) > 0:
            print("[INFO] detected {} markers for '{}'".format(
                len(corners), arucoName))
            list_markers.append(arucoName)
    return list_markers

def get_poses(foldername):
    """
    Get camera poses from aruco markers placed on images. Each marker is 2.5x2.5cm and aligned with the center of the object.
    """
    searchpath = os.path.join(foldername, "*.png")
    files = glob.glob(searchpath)
    files.sort()
    for i in files:
        image = cv2.imread(i)
        image2 = image.copy()
        #print("image: ", i)
        marker_type = get_aruco_dict()
        aruco_dict = aruco.Dictionary_get(marker_type)
        parameters = aruco.DetectorParameters_create()
        #parameters.adaptiveThreshConstant = 10
        corners, ids, rejectedImgPoints = aruco.detectMarkers(image, aruco_dict, parameters=parameters)
        if len(corners) > 0: # atleast 1 marker was detected
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                print("[INFO] ArUco marker ID: {}".format(markerID))

            aruco.drawDetectedMarkers(image2, corners)
        else:
            print("no markers found")
        plt.imshow(image2)
        plt.show()

if __name__ == '__main__':
    get_poses('/home/sushmitawarrier/Desktop/datasets/lettuce_real/lettuce-rework/images')
