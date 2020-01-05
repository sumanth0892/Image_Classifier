#Corner detectors and Feature detection algorithms.
#Import the necessary libraries
import cv2
import numpy as np

#1. Harris Corner detector
image = ''
img = cv2.imread(image)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,4,5,0.01)
dst = cv2.dilate(dst,None)
img[img>0.01*dst.max()] = [0,0,0]
cv2.imshow('Harris Corner',img)
cv2.waitKey()

#2. Scale Invariant Feature Transform
image = ''
img = cv2.imread(image)
grayImg = cv2.cvtColor(img,COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(grayImg,None)
IMAGE = cv2.drawKeypoints(img,keypoints,img,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT Features',IMAGE)
cv2.waitKey()

#3. Speeded Up Robust Features
img = cv2.imread(image)
grayImg = cv2.cvtColor(img,COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create(15000)
kp,des = surf.detect(grayImg,None)
IMAGE = cv2.drawKeypoints(img,kp,None,(0,255,0),4)
cv2.imshow('SURF Features',IMAGE)
cv2.waitKey()

#4. Features from Accelerated Segment Test


