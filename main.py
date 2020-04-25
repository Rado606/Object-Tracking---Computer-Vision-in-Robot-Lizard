######################################################
# Object Tracking - Computer Vision for Lizard Robot #
######################################################

import cv2
import numpy as np
import math as math
import imutils


# Turning on a camera, O is number of a device
stream=cv2.VideoCapture(0)

# Definition of blue colour in HSV
LowerHSV=(100, 150, 0)
UpperHSV=(140, 255, 255)

'''
# Definition of green colour in HSV
LowerHSV=(29, 86, 6)
UpperHSV=(64, 255, 255)
'''

while True:
    # Grabbing and Decoding next video frame
    # Returns true if grabbing was OK (else - false) and grabbed frame (else - NULL)
    ret, frame = stream.read()

    if frame is None:
        break

    # Resizing the frame
    frame=imutils.resize(image=frame, width=600)

    # Calculating the center of the photo
    # center_img is a center point of the image (x,y)
    # height is y, width is x
    height_img = np.size(frame, 0)
    width_img = np.size(frame, 1)
    center_img = (int(width_img / 2), int(height_img / 2))

    # Blur filter, conversing RGB to HSV
    blurImg = cv2.GaussianBlur(src=frame, ksize=(5, 5), sigmaX=30)
    hsvImg = cv2.cvtColor(src=blurImg, code=cv2.COLOR_BGR2HSV)
    maskHSV = cv2.inRange(src=hsvImg, lowerb=LowerHSV, upperb=UpperHSV)

    # Erode, Dilate
    maskErode = cv2.erode(src=maskHSV, kernel=(11, 11), iterations=2)
    maskDilate = cv2.dilate(src=maskErode, kernel=(11, 11), iterations=2)
    mask = maskDilate.copy()

    # Finding contours
    contours = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    center=None

    if len(contours)>0:
        # Find the biggest contour
        maxContour=max(contours, key=cv2.contourArea)
        ((x, y), radius)= cv2.minEnclosingCircle(maxContour)

        # Return dictionary m10, m00, m01, x=m10/m00, y=m01/m00
        M=cv2.moments(maxContour)

        if M["m00"] != 0 and radius>10:

            # Center of mass,
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Draw the circle and centroid on the frame,
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.circle(frame, center_img, 5, (0, 0, 255), -1)

            # Draw the line between two points
            cv2.line(img=frame, pt1=center, pt2=center_img, color=(120, 31, 53), thickness=2)

            # Calculate the vector between two points x, change y to -y
            vectorCenterPoint_Img = (center[0] - center_img[0], center_img[1] - center[1])
            vectorTorString=str(vectorCenterPoint_Img)

            # Distance between points
            distance = math.sqrt(math.pow(vectorCenterPoint_Img[0], 2) + math.pow(vectorCenterPoint_Img[1], 2))

            # Return the floor of distance
            distance = math.floor(distance)
            cv2.putText(frame, vectorTorString, center_img, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, lineType=cv2.LINE_8)

            # print(distance)

    if cv2.waitKey(delay=1) & 0xFF == ord('q'):
        break

    cv2.imshow("Frame", frame)

    '''
    cv2.imshow("blurImg", blurImg)
    cv2.imshow("hsvImg", hsvImg)
    cv2.imshow("maskHSV", maskHSV)
    cv2.imshow("maskErode", maskErode)
    cv2.imshow("maskDilate", maskDilate)
    '''

# Close capturing devices
stream.release()

# Destroy all of the HighGUI
cv2.destroyAllWindows()

