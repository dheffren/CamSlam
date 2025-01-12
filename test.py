
import numpy as np
from PIL import Image
from camGeom import calculate_essential_matrix, normalizeHomogenousCoordinates, checkEpipolarConstraint

import cv2
##  OV 2640 wide angle lens focal length prolly 75 mm. 
## Guess for the principal point: Width/2, height/2. 
## FULL FRAME SENSOR: Sensor size: 36 mm x 24 mm
##I think the resolution is 1632 x 1232 (width by height) might not be true for mine though. Not sure if change "resolution". Doesn't affect prolly. 

"""
Camera matrix calculation:

fx = f*imageWidth/sensorWidth = 75 mm * 1632 px / 36 mm  = 3400 px 
fy = _____ = 75 * 1232 px / 24 mm = 3850 px
cx ~= imageWidth/2 = 816 px
cy ~= imageHeight/2 = 616 px
s ~= 0

SO camera matrix is: 

K = [3400 0 816
     0 3850 616
     0 0     1]
Can get normalized image coordinates now. 
"""



"""
Essential matrix relates corresponding points in stereo images. 
If have homogenous normalized image coordinates. if coorespond to same 3d point, then 
y'T E y = 0. 
"""
numKeypoints = 1000
K = np.array([[3400, 0, 816], [0, 3850, 616], [0, 0, 1]])
orb = cv2.ORB_create(nfeatures = numKeypoints)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

img1 = np.array(Image.open('20241221192645.jpg'))
img2 = np.array(Image.open('20241221192639.jpg'))
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2= orb.detectAndCompute(img2, None)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints
list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]
matches_kp1 = np.asarray(list_kp1)
matches_kp2 = np.asarray(list_kp2)

# Remove duplicate matches
combine_reduce = np.unique(np.concatenate((matches_kp1, matches_kp2),
                                        axis=1),
                         axis=0)
points1 = combine_reduce[:, 0:2]
points2 = combine_reduce[:, -2:]
points1Hom = normalizeHomogenousCoordinates(points1, K)
points2Hom = normalizeHomogenousCoordinates(points2, K)
#swap to get coords of second with respect to first. 
essentialMatrix, R, t = calculate_essential_matrix(points2Hom, points1Hom)
goodPoints = checkEpipolarConstraint(points2Hom, points1Hom, essentialMatrix)
points1adj = points1[goodPoints]
points2adj = points2[goodPoints]
print("Essential matrix: ", essentialMatrix)
print(R)
#just using the scale factor to be the VALUE of t. 
tadj = t/np.linalg.norm(t)
print(tadj)