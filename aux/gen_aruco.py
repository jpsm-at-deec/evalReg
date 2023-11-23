import numpy as np
import argparse
import cv2
import sys

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

#arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_4X4_50"])
ssize = 150
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
sel = [1,2,3,4,5,6,7,8,9]
for sid in sel:
	tag = np.zeros((ssize, ssize, 1), dtype="uint8")
	tag = cv2.aruco.generateImageMarker(arucoDict, sid, ssize, tag, 1)
	cv2.imwrite("00"+str(sid)+".png", tag)
	cv2.imshow("ArUCo Tag", tag)
	cv2.waitKey(0)

sel2 = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]


for bid in sel2:
	tag = np.zeros((ssize, ssize, 1), dtype="uint8")
	tag = cv2.aruco.generateImageMarker(arucoDict, bid, ssize, tag, 1)
	cv2.imwrite("0"+str(bid)+".png", tag)
	cv2.imshow("ArUCo Tag", tag)
	cv2.waitKey(0)

