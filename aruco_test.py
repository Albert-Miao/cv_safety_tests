import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_object_angle_degrees(rvec, world_z_axis): 
    """
    Get the safety cost associated a certain object 
    Method: 
        Coordinate frame: 
            - x,y axes of the obstacle are in the plane of the obstacle
            - the z axis points upwards from the obstacle (up from the plane)
        Tipping angle: 
            - Get the orientation of the object
            - Get the z axis of the object frame 
            - Get the dot product of the z axis of the object frame with the world frame
            - This gives you the tipping angle between the upright cylinder and the tipped one. 
        Safety: 
            - To gauge the safety we return a sum of the absolute value of the roll and pitch of the object

    args: 
        - rvec: rotation vector that you get from cv2.aruco. pose estimation function 
        - world_z_axis: after calibration define the z axis in the world frame (use an aruco marker to do this)
    returns: 
        - safety_theta: The safety cost of the object in RADIANS
    """

    precision = 5 # precision in decimals

    rvec = np.array(rvec[0, 0])
    rotation = R.from_rotvec(rvec)
    rotmat = rotation.as_matrix()
    z_axis = rotmat[:, -1]

    # world_z_axis = np.array([0, 0, 1]).reshape(z_axis.shape)

    angle = np.arccos(np.round(np.dot(z_axis, world_z_axis), decimals=precision))
    angle_degrees = angle/np.pi * 180

    abs_angle_degrees = np.abs(angle_degrees)

    return abs_angle_degrees

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

matrix_coefficients = np.load("calibration_matrix.npy")
distortion_coefficients = np.load("distortion_coefficients.npy")

world_z_axis = np.array([0.0128116 , -0.95702334, -0.2897278 ]) # np.array([0, 0, 1])

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
            cv2.drawFrameAxes(image=frame,cameraMatrix=matrix_coefficients, distCoeffs=None, rvec=rvec, tvec=tvec, length=0.02, thickness=2	) 

            object_angle_degrees = get_object_angle_degrees(rvec=rvec, world_z_axis=world_z_axis)

            # print("rotmat: ", rotation.as_matrix())
            print("Angle : ", object_angle_degrees)
            # print("Safety cost: ", object_safety_cost)
            print("\n")

            danger += (rvec[0, 0, 0] ** 2 + rvec[0, 0, 1] ** 2) ** (1 / 2)

            # # show the last image and halt if you detect an aruco marker - click on image to proceed
            # cv2.imshow('test', frame)
            # cv2.waitKey(0)
            # print()

            cv2.putText(frame, str(object_angle_degrees), corners[i][0][0].astype(np.int), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.putText(frame, str(danger), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imshow('test', frame)
    cv2.waitKey(1)
