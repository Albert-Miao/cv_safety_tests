import os
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
writer = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30,
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while True:

    ret, frame = cap.read()


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    masked = cv2.inRange(hsv, (80, 130, 50), (130, 220, 220))

    contours, _ = cv2.findContours(masked, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        continue
    contours = np.array(contours)[np.array(list(map(lambda cnt: cv2.contourArea(cnt) > 200, contours)))]
    centers = []

    for c in contours:
        M = cv2.moments(c)

        # Contour must be minimum size
        # if M['m00'] < 50:
        #     continue

        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        centers.append([x, y])

    masked = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
    for center in centers:
        masked = cv2.drawMarker(masked, center, color=(255, 0, 0))

    upper = centers[0]
    lower = centers[1]

    y_dist = np.abs(upper[1] - lower[1])
    x_dist = np.abs(upper[0] - lower[0])

    angle = int(np.arctan(x_dist/y_dist) * 180 / np.pi)
    masked = cv2.putText(masked, str(angle), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('frame', masked)
    writer.write(masked)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()