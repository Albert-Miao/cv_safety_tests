import os
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

matrix_coefficients = np.load("calibration_matrix.npy")
distortion_coefficients = np.load("distortion_coefficients.npy")

while True:
    danger = -1
    ret, frame = cap.read()

    arucoDict = cv2.aruco.getPredefinedDictionary(10)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    (corners, ids, rejected) = detector.detectMarkers(frame)

    if len(corners) > 0:
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            # cv2.aruco.drawAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
            print(rvec)

            danger += (rvec[0, 0, 0] ** 2 + rvec[0, 0, 1] ** 2) ** (1 / 2)

    cv2.putText(frame, str(danger), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('test', frame)
    cv2.waitKey(1)
