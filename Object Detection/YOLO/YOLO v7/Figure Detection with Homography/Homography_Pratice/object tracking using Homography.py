import cv2
import numpy as np

imgbefore = cv2.imread("pan2.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(imgbefore, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture(4)

#Features
sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
#img = cv2.drawKeypoints(img,kp_image,img)

#Feature matching
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True :
    _, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #trainimage

    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe,None)
    #grayframe = cv2.drawKeypoints(grayframe, kp_grayframe,grayframe)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    good_points = []
    for m, n in matches :
        if m.distance < 0.8*n.distance:
            good_points.append(m)

    img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)
    


    #cv2.imshow("Image", img)
    #cv2.imshow("Frame", frame)
    cv2.imshow("img3", img3)

    key = cv2.waitKey(1)
    if key==27:
        break