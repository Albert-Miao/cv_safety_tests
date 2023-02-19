import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

matrix_coefficients = np.load("calibration_matrix.npy")
distortion_coefficients = np.load("distortion_coefficients.npy")

max_dictionary_types = 20
num_tests_per_dictionary = 20

dict_num = 0 
final_dict_num = None

while True: 
    danger = -1 
    ret, frame = cap.read()

    arucoDict = cv2.aruco.getPredefinedDictionary(dict_num)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    for _ in range(num_tests_per_dictionary): 
        (corners, ids, rejected) = detector.detectMarkers(frame)
        print("ids: ", ids)

        if ids is not None:
            for i in range(0, len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                            distortion_coefficients)
                rotation = R.from_rotvec(rvec[0, 0])
                r_euler_angle = rotation.as_euler('zxy', degrees=True)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                cv2.imshow('test', frame)

            final_dict_num = dict_num
            break     
    
    if final_dict_num is not None: 
        break
    # increment dictionary num 
    dict_num += 1
    if dict_num > max_dictionary_types: 
        dict_num = 0 

    cv2.imshow('test', frame)
    cv2.waitKey(1)


cv2.aruco.drawDetectedMarkers(frame, corners)
cv2.imshow('test', frame)
cv2.waitKey(0)

print("Found final dictionary num: ", final_dict_num)
