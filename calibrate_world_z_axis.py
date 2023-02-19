"""
File used to calibrate the z axis of the world frame 
"""
import cv2 
import numpy as np 
from scipy.spatial.transform import Rotation as R

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

matrix_coefficients = np.load("calibration_matrix.npy")
distortion_coefficients = np.load("distortion_coefficients.npy")

baseline_angle_degrees = 0 #107 # the baseline angle to compare the tilt against when computing safety 

while True:
    danger = -1
    ret, frame = cap.read()

    # arucoDict = cv2.aruco.getPredefinedDictionary(10)
    arucoDict = cv2.aruco.getPredefinedDictionary(16)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    (corners, ids, rejected) = detector.detectMarkers(frame)
    print("ids: ", ids)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
            rotation = R.from_rotvec(rvec[0, 0])
            r_euler_angle = rotation.as_euler('zxy', degrees=True)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(image=frame,cameraMatrix=matrix_coefficients, distCoeffs=distortion_coefficients, rvec=rvec, tvec=tvec, length=0.02, thickness=2) 

            print("Detected marker z axis: ")
            print(rotation.as_matrix()[:, -1])
            print()

    cv2.imshow('test', frame)
    cv2.waitKey(1)
