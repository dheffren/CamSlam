
from flask import Flask, Response
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from camGeom import calculate_essential_matrix,estimateFundamentalMatrix, poseEstimationFromEss,normalizeHomogenousCoordinates, checkEpipolarConstraint, computeDLT
from calibrate import mainCalibration, undistortImage
from camGeomRepeat import estimate_fundamental_matrix
from visualOd import *


def doVisSLAM():
    """
    TODO: Fix homography
    
    """
    numKeypoints = 1000
    orb = cv2.ORB_create(nfeatures = numKeypoints)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    numPics =3
    location = "motionImages/image"
    imSize = (15,22)
    #calibrate somehow. Do i need the intrinsic matrix? 
    listImages = loadImages(location, numPics)
    #listImages = listImages[::-1]
    
    K, newK, mapx, mapy, roi, rvecs, tvecs = mainCalibration()

    mapInit = False
    currFrame = 0
    currI= undistortImage(listImages[currFrame], mapx, mapy, roi).astype(np.uint8)
    kp1, des1 = computeFeatures(currI, orb, numKeypoints)
    currFrame +=1
    firstI = currI
    prevI = currI
    minMatches = 100
    while not mapInit and currFrame < len(listImages):
        
        currI = undistortImage(listImages[currFrame], mapx, mapy, roi).astype(np.uint8)
        kp2, des2 = computeFeatures(currI, orb, numKeypoints)
        currFrame+=1
        key1, key2, q1, q2 = getImMatchesAlt(kp1, des1, kp2, des2, bf)
        if q1.shape[0]<minMatches:
            continue
        points1 = key1[q1, :]
        points2 = key2[q2, :]
        F, scoreF, inliersF = computeFundamentalMatrix(points1, points2, newK)
        H, scoreH, inliersH = computeHomography(points1, points2)
        ratio = scoreH/(scoreH + scoreF)
        ratioThreshold = 0.45
        if ratio > ratioThreshold:
            inliers = inliersH
            T = H
            print("Chose hom")
        else:
            inliers = inliersF
            T = F
        inlier1 = points1[inliers, :]
        inlier2 = points2[inliers, :]
        print("Inlier 1 shape: ", inlier1.shape)
        print("Inlier 2 shape: ", inlier2.shape)
        relPose, validFrac = estrelpose(T, inlier1, inlier2, newK)
        R  = relPose[0]
        t = relPose[1]
        if validFrac <.9:
            continue
        X = triangulate(inlier1, inlier2, relPose[0], relPose[1], newK)
        R1c = np.identity(3)
        t1c = np.zeros((3,))
        print("ready to visualize")
        visualize_3d_points_and_cameras(X, [(R1c, t1c), (R,t)])
        kp1 = kp2
        des1 = des2
        prevI = currI

def computeFeatures(frame, orb, numKeypoints):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(grayFrame, None)
    return kp, des

def computeFundamentalMatrix(points1, points2, K):
   F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
   inliers = mask.ravel() == 1
   points1a = np.concatenate([points1, np.ones((points1.shape[0], 1))], axis=1)
   points2a = np.concatenate([points2, np.ones((points1.shape[0], 1))], axis=1)
   val = np.sum((points2a @ F @ points1a.T)**2)
   return F, val, inliers
def triangulate(points1, points2, R, t, K):
    """
    TODO: Cut out outliers from this method. 
    """
    # t needs to be s*t
    hom1p = np.concatenate([points1, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom2p = np.concatenate([points2, np.ones(shape=(points1.shape[0], 1))], axis=1)
    R1c = np.identity(3)
    t1c = np.zeros((3,))
    extMat1 = np.concatenate([R1c, t1c[:, np.newaxis]], axis=1)
    P1 = K@extMat1
    R2c = R @ R1c
    t2c = (R@t1c[:, np.newaxis] + t[:, np.newaxis])[:, 0]
    P2 = np.concatenate([R2c, t2c[:, np.newaxis]], axis=1)
    #need to CHOOSE which ones to include in the end result. Some triangulations may not be a good idea. 
    #X = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    #print(X.shape)
    X = computeDLT(hom1p, hom2p, P1, P2)
    return X
def computeHomography(points1, points2):
   H, mask = cv2.findHomography(points1, points2, cv2.FM_RANSAC)
   points1a = np.concatenate([points1, np.ones((points1.shape[0], 1))], axis=1)
   points2a = np.concatenate([points2, np.ones((points1.shape[0], 1))], axis=1)
   val = np.sum(np.linalg.norm((points2a.T - H @ points1a.T)**2, axis=0))
   print(H)
   inliers = mask.ravel()==1
   return H, val, inliers
def estrelpose(F, points1, points2, K):
    """
    Need points1 and points2
    """
    E =   K.T @ F @ K
    hom1p = np.concatenate([points1, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom2p = np.concatenate([points2, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom1 = hom1p @ np.linalg.inv(K).T
    hom2 = hom2p @ np.linalg.inv(K).T
    s = 1
    R1c = np.identity(3)
    t1c = np.zeros((3,))
    extMat1 = np.concatenate([R1c, t1c[:, np.newaxis]], axis=1)
    P1 = K@extMat1
    R1, R2, t1, t2 = poseEstimationFromEss(E, hom1, hom2)
    disp = False
    #hope these are supposed to be pixel inputs. 
    P2, R, t, f = poseEstimation(hom1p, hom2p, R1, R2, t1, t2, P1, K,K , s, disp)

    return (R,t), f 

doVisSLAM()