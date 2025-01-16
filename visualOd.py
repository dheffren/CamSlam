from flask import Flask, Response
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from camGeom import calculate_essential_matrix,estimateEssentialMatrix, poseEstimationFromEss,normalizeHomogenousCoordinates, checkEpipolarConstraint, computeDLT
from calibrate import mainCalibration, undistortImage
from camGeomRepeat import estimate_fundamental_matrix

import math

def doVisualOdometry():

    """
    Have trajectory of images representing motion of a camera. 
    Monocular, so only can know the answer up to some scale factor. 
    
    Steps: 
    1. Estimate pose of second view relative to first view via estimating essential matrix, decompose into camera location and orientation. 
    2. Bootstrap estimating camera trajectory using global bundle adjustment: Eliminate outliers via constraint.; 
    Find 3d to 2d correspondences between points triangulated from previous two views and current view. Compute world camera pose. Use bundle adjustment to reduce drift. 
    3. Estimating remaining camera trajectory using windowed bundle adjustment. 
    
    REVIEW EPIPOLAR PROJECTION. 
    """
    numKeypoints = 1000
    orb = cv2.ORB_create(nfeatures = numKeypoints)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    numPics =3
    location = "motionImages/frame"
    imSize = (15,22)
    #calibrate somehow. Do i need the intrinsic matrix? 
    listImages = loadImages(location, numPics)
    #listImages = listImages[::-1]
    
    K, newK, mapx, mapy, roi, rvecs, tvecs = mainCalibration()
   
    
    listKeypoints = []
    listKeypoints
    listDescriptors = []
    triangles = []
    undistortedImages = []
    listX = []
    listKeypointsS = []
    listQueryPairs = []
    listPoses = []

    #for i in range(2):
        #frame = listImages[i]
        #undi= undistortImage(frame, mapx, mapy, roi).astype(np.uint8)
        #undistortedImages.append(undi)
        #grayFrame = cv2.cvtColor(undi, cv2.COLOR_BGR2GRAY)
        #kp2, des2 = orb.detectAndCompute(grayFrame, None)
        #listKeypoints.append(kp2)
        #listDescriptors.append(des2)
    frame = listImages[0]
    undi= undistortImage(frame, mapx, mapy, roi).astype(np.uint8)
    undistortedImages.append(undi)
    grayFrame = cv2.cvtColor(undi, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(grayFrame, None)
    listKeypoints.append(kp1)
    listDescriptors.append(des1)
    frame = listImages[1]
    undi= undistortImage(frame, mapx, mapy, roi).astype(np.uint8)
    undistortedImages.append(undi)
    grayFrame = cv2.cvtColor(undi, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(grayFrame, None)
    listKeypoints.append(kp2)
    listDescriptors.append(des2)
    #queries are the features which match between the two. 
    #other method with queries mixes up order. 
    #points1, points2, query1, query2 = getImMatches(kp1, des1, kp2, des2, bf)
    #TODO: CHeck that the alt messing with the image ordering isn't what's screwing up the epipole calculation. It can't be, right? 
    keypoints1, keypoints2, query1, query2= getImMatchesAlt(kp1, des1, kp2, des2, bf)
    #print(keypoints2s.shape)
    print(query2)
    print(keypoints1.shape)
    #points1a= keypoints1[queries1.astype(bool), :]
    points1 = keypoints1[query1]
    points2 = keypoints2[query2]
   # print(points1a)
    print(points1)
    print(points1.shape)
    #print(points1a.shape)
    #points2a= keypoints2[queries2.astype(bool), :]
    #print(points2.shape)
    #print(points2a.shape)
    #print(points1a[0:10])
    #print(points1[0:10])


    #pose of second relative to first. 
    #note: inliers is based on the input points1, points2 size, not like queries. 
    #X, R, t, inliers = triangulate(points1, points2, undistortedImages[0], undistortedImages[1], newK, newK)
    R, t, inliers = estRelPose(points1, points2, undistortedImages[0], undistortedImages[1], newK, newK)
    print(inliers)
    print(inliers.shape)
    print(query1)
    print(len(query1))
    query1 = query1[inliers]
    query2 = query2[inliers]
    listQueryPairs.append((query1, query2))
    listPoses.append((R,t))
    listKeypointsS.append((keypoints1, keypoints2))

    for i in range(2, numPics):
        frame = listImages[i]
        undi= undistortImage(frame, mapx, mapy, roi).astype(np.uint8)
        undistortedImages.append(undi)
        grayFrame = cv2.cvtColor(undi, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(grayFrame, None)
        #previous frame. -2 because just added another.
        kp1, des1 = listKeypoints[-1], listDescriptors[-1]
        #keypoints 1 contains a list of all keypoints for that frame. Doesn't change with matches. 
        keypoints1, keypoints2, query1, query2 = getImMatchesAlt(kp1, des1, kp2, des2, bf)
        #need to get matches of query 1 with PREVIOUS query2. 
        image1 = undistortedImages[i-1]
        image2 = undistortedImages[i]
        points1 = keypoints1[query1]
        points2 = keypoints2[query2]
        #
        R, t, inliers = estRelPose(points1, points2, image1, image2, newK, newK)
        #adjust for inliers. 
        print(inliers.shape)
        print(inliers)
        query1 = query1[inliers]
        query2 = query2[inliers]
        #maybe instead need to find correspondences between 3d and 2d. 
        #the matlab file doesn't use the calculate rel pose here of R and t. 
        worldPoints, imagePoints3 = triangulatePrevAndCurrentPoints((keypoints1, keypoints2), (query1, query2), listKeypointsS[-1], listQueryPairs[-1], listPoses[-1], (R,t), newK)
        print(worldPoints.shape)
        print(imagePoints3.shape)
        listQueryPairs.append((query1, query2))
        listPoses.append((R,t))
        listKeypointsS.append((keypoints1, keypoints2))
        #NOW estimate the world camera pose given the image points and world points information. 
        #not sure how to do this or what this means. 



        #still need bundle adjustement. 


def triangulatePrevAndCurrentPoints(currKeypoints, currQueries, prevKeypoints, prevQueries, prevPose, currentPose, K, s = 1):
    """
    TODO: DLT is underdetermined. CHeck what needs to be changed here. We get VERY different results than what is expected. 
    something is wrong here. 
    INSTEAD OF CHANGING THE STUFF BEFORE DLT, DO DLT THEN CHANGTE STUFF to see which of the 3d points correspond to our two d points. 
    Step 1: Triangulate points from previous ones (assuming that world coordinates, the one before last is I 0)
    But only triangulate ones that match up with valid from THIS one. 
    Step 2: Project those points into current view. 
    """
    ckey1, ckey2 = currKeypoints
    cq1, cq2 = currQueries
    pkey1, pkey2 = prevKeypoints
    pq1, pq2 = prevQueries
    #keypoints should be the same
    assert(np.all(pkey2 == ckey1))
    #querys should be different. 
    key1 = pkey1
    key2 = pkey2
    key3 = ckey2

    R1cam = np.identity(3)
    t1c = np.zeros((3,))
    R2r, t2r = prevPose
    R3r, t3r = currentPose
    #gets Pose of R2cam in world coordinates
    R2cam = R2r@R1cam
    #should I have diff s values? 
    t2c = (R2r@t1c[:, np.newaxis] + s*t2r[:, np.newaxis])[:, 0]
    #pose of R3cam in world coordinates. 
    R3cam = R3r@R2cam
    t3c = (R3r@t2c[:, np.newaxis] + s*t3r[:, np.newaxis])[:, 0]
    ext1 = np.concatenate([R1cam, t1c[:, np.newaxis]], axis=1)  
    ext2 = np.concatenate([R2cam, t2c[:, np.newaxis]], axis=1)  
    ext3 = np.concatenate([R3cam, t3c[:, np.newaxis]], axis=1)
    P1 = K@ext1
    P2 = K@ext2
    P3 = K@ext3
    #Need to limit the 3d points to those which are in BOTH query matches. 
    #Need points1 and points2 to already have first pair and second pair filtered out. 
    cqueries1 = np.zeros((ckey1.shape[0],),dtype = int)
    cqueries2 = np.zeros((ckey2.shape[0],), dtype = int)
    cqueries1[cq1] = 1
    print(cqueries1)
    cqueries2[cq2] = 1
    pqueries1 = np.zeros((pkey1.shape[0],),dtype = int)
    pqueries2 = np.zeros((pkey2.shape[0],), dtype = int)
    pqueries1[pq1] = 1
    print(cqueries1)
    pqueries2[pq2] = 1
    #NEed to go to an ordered list. 
    # cq1 is a list of ordered values, as is pq2. Want to get the intersection of the two lists, but in 
    # two orderings: One preserving the order of the first, and the other preserving the second. Want list of 0, 1s with equal length to filter out the other queries. 
    # Which ones of a in b? Which ones of b in a? Then get a and b. 

    resp = np.isin(pq2, cq1)
    resc = np.isin(cq1, pq2)
    #sort them 

    q1adj = pq1[resp]
    pq2adj = pq2[resp]
    cq2adj = cq1[resc]
    q3adj = cq2[resc]

    #sort both so the middle shared list is ordered. 
    indicesp = np.lexsort((q1adj, pq2adj))
    q1s = q1adj[indicesp]
    pq2s = pq2adj[indicesp]
    indicesq = np.lexsort((q3adj, cq2adj))
    cq2s = cq2adj[indicesq]
    q3s = q3adj[indicesq]
    assert(np.all(cq2s == pq2s))
    #these points are LINED UP by matches. 
    points1 = key1[q1s]
    points2 = key2[cq2s]
    points3 = key3[q3s]
    p1hom = np.concatenate([points1, np.ones((points1.shape[0], 1))], axis=1)
    p2hom = np.concatenate([points2, np.ones((points2.shape[0], 1))], axis=1)
    p3hom = np.concatenate([points3, np.ones((points3.shape[0], 1))], axis=1)
    worldPoints = computeDLT(p1hom, p2hom, P1, P2)
    imagePoints3 = worldPoints@P3.T
    imagePoints3/=imagePoints3[:, 2][:, np.newaxis]
    imagePoints3 = imagePoints3[:, 0:2]
    print(imagePoints3)
    print(p3hom)
    return worldPoints, imagePoints3

def estRelPose(points1, points2, image1, image2, K1, K2):
    hom1p = np.concatenate([points1, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom2p = np.concatenate([points2, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom1 = hom1p @ np.linalg.inv(K1).T
    hom2 = hom2p @ np.linalg.inv(K2).T
    # Their method is MUCH better than mine. 
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    E = K2.T @ F @ K1
    inlierIndices = mask.ravel() == 1
    print("Shape of ind: ", inlierIndices.shape)
    print("inlier indices: ", inlierIndices)
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]

    #it's actually more complicated to compute R and t, 4 possible options. Need to address this by choosing ones such that keypoints in front of BOTH cameras. 
    #E, R, t = calculate_essential_matrix(hom1[0:100], hom2[0:100])
    #E= estimateEssentialMatrix(hom1, hom2)
    #THIS FORMULA IS WRONG. Needs to be newK @ E @ newK.T
    
    #F = np.linalg.inv(newK.T) @ E @ np.linalg.inv(newK)
    #inliers1 = points1
    #inliers2 = points2
    #print("F: ", F)
    lines = cv2.computeCorrespondEpilines(inliers1,1, F = F)
    #print(lines)
    plot_epipolar_lines(image1, image2, inliers1, inliers2, F)
    #get real R and t: 

    goodPoints = checkEpipolarConstraint(hom1, hom2, E)
    print("good points: ", goodPoints)

    doubleCheck = np.logical_and(inlierIndices, goodPoints)
    
    points1adj = points1[doubleCheck]
    points2adj = points2[doubleCheck]
    hom1padj = hom1p[doubleCheck]
    hom2padj = hom2p[doubleCheck]
    hom1adj = hom1[doubleCheck]
    hom2adj = hom2[doubleCheck]
 
    
    display_keypoint_matches(image1, image2, points1adj, points2adj)
    s = 10
    R1c = np.identity(3)
    t1c = np.zeros((3,))
    extMat1 = np.concatenate([R1c, t1c[:, np.newaxis]], axis=1)
    P1 = K1@extMat1
    R1, R2, t1, t2 = poseEstimationFromEss(E, hom1adj, hom2adj)
    disp = True
    P2, R, t = poseEstimation(hom1padj, hom2padj, R1, R2, t1, t2, P1, K1, K2, s, disp)
    return R, t, doubleCheck
def triangulate(points1, points2, image1, image2, K1, K2):
    hom1p = np.concatenate([points1, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom2p = np.concatenate([points2, np.ones(shape=(points1.shape[0], 1))], axis=1)
    hom1 = hom1p @ np.linalg.inv(K1).T
    hom2 = hom2p @ np.linalg.inv(K2).T
    # Their method is MUCH better than mine. 
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    E = K2.T @ F @ K1
    inlierIndices = mask.ravel() == 1
    print("Shape of ind: ", inlierIndices.shape)
    print("inlier indices: ", inlierIndices)
    inliers1 = points1[mask.ravel() == 1]
    inliers2 = points2[mask.ravel() == 1]

    #it's actually more complicated to compute R and t, 4 possible options. Need to address this by choosing ones such that keypoints in front of BOTH cameras. 
    #E, R, t = calculate_essential_matrix(hom1[0:100], hom2[0:100])
    #E= estimateEssentialMatrix(hom1, hom2)
    #THIS FORMULA IS WRONG. Needs to be newK @ E @ newK.T
    
    #F = np.linalg.inv(newK.T) @ E @ np.linalg.inv(newK)
    #inliers1 = points1
    #inliers2 = points2
    #print("F: ", F)
    lines = cv2.computeCorrespondEpilines(inliers1,1, F = F)
    #print(lines)
    plot_epipolar_lines(image1, image2, inliers1, inliers2, F)
    #get real R and t: 

    goodPoints = checkEpipolarConstraint(hom1, hom2, E)
    print("good points: ", goodPoints)

    doubleCheck = np.logical_and(inlierIndices, goodPoints)
    
    points1adj = points1[doubleCheck]
    points2adj = points2[doubleCheck]
    hom1padj = hom1p[doubleCheck]
    hom2padj = hom2p[doubleCheck]
    hom1adj = hom1[doubleCheck]
    hom2adj = hom2[doubleCheck]
    #homin1p =  np.concatenate([inliers1, np.ones(shape=(inliers1.shape[0], 1))], axis=1)
    #homin2p=  np.concatenate([inliers2, np.ones(shape=(inliers2.shape[0], 1))], axis=1)
    #homin1 = homin1p@np.linalg.inv(K1).T
    #homin2 = homin2p@np.linalg.inv(K2).T
    
    display_keypoint_matches(image1, image2, points1adj, points2adj)
    s = 10
    R1c = np.identity(3)
    t1c = np.zeros((3,))
    extMat1 = np.concatenate([R1c, t1c[:, np.newaxis]], axis=1)
    P1 = K1@extMat1
    R1, R2, t1, t2 = poseEstimationFromEss(E, hom1adj, hom2adj)
    disp = True
    P2, R, t = poseEstimation(hom1padj, hom2padj, R1, R2, t1, t2, P1, K1, K2, s, disp)
    X = computeDLT(hom1padj, hom2padj, P1, P2)
    R2c = R @ R1c
    t2c = (R@t1c[:, np.newaxis] + s*t[:, np.newaxis])[:, 0]
    visualize_3d_points_and_cameras(X[:, 0:3], [(R1c, t1c), (R2c, t2c)])
    return X, R, t, doubleCheck 
def estWorldPose():
    return
def normalizeX(X):
    avg = np.mean(X)
    s = np.std(X)
    print(avg)
    print(s)
    norm = (X-avg)/s 
    print(norm.shape)
    average = np.linalg.norm(norm, axis=-1)
    print(average)
    Xn = X[np.abs(average) <= 3]
    return Xn
def testMatrix(K):
    """
    Had problems because some points were on the principal plane
    GOT EVERYTHING TO WORK BY NOT PUTTING IN NORMALIZED COORDS INTO DLT. 
    """
    #this is written as z x y
    #worldPoints = np.array([[1,1,1],[1,-1, 1], [1,1,0], [1,-1, 0], [.5, 1, 1], [0, -1, 1], [1,0,0], [1,0,1]], dtype = np.float64)
    #worldPoints[:, :] = worldPoints[:, [1, 2, 0]]
    #x y z. X is right l eft. y is forward toward monitor. Z is up. 
    worldPoints = np.array([[1,1,1], [1,1,0], [-1,1, 1], [-1,1,0], [0, 1, 0], [0, 1, 1], [0, .5, 0],[.5, .5, 1], [-.5, .5, 1], [0, .5, 1]])
    print(worldPoints)
    print(worldPoints.shape)
    #not sure about right handedness or left handedness of coordinate system. 
    #origin  of Camera relative to world. pab. 
    #3x1


    Ct1 = np.array([-1, 0, .5], dtype = np.float64)[:, np.newaxis]
    #coordinates of camera 1 in terms of basis of World. rx ry rz
    rx = np.array([1,-.5, -.5], dtype = np.float32)
    rx/=np.linalg.norm(rx)
    #z going up forward and right. 
    rz = np.array([1,1,1], dtype = np.float32)
    rz/= np.linalg.norm(rz)
    print("dot: ", np.dot(rx, rz))
    ry = np.cross(rz, rx)
    ry/=np.linalg.norm(ry)
    print(np.dot(rx,ry))
    print(np.dot(ry,rz))
    print("Ry: ", ry)
    assert(np.dot(rx,ry) == 0.0 and np.dot(ry, rz) == 0.0 and np.dot(rx, rz) == 0.0)
    #Rab. Have xa
    Rwc1 = np.stack([rx, ry, rz],axis=1)
    print("Rwc1: ", Rwc1)
    #Have pab and rab, and coordinates xa. Want xb. 
    #so, Rab.T(xa - pab) = xb
    #multiply by Rwc1.T on right = mult by Rwc1 on left. 
    #THIS IS RIGHT. 
    
    cam1Points = (worldPoints - Ct1.T)@Rwc1
    R1cam = Rwc1.T
    t1c = (-Rwc1.T@Ct1)[:, 0]
    print(cam1Points[0, 0]*rx + cam1Points[0, 1]*ry + cam1Points[0, 2]*rz + Ct1.T)

    Ct2 = np.array([1,.25, 0], np.float64)[:, np.newaxis]
    rx = np.array([0,0,1], np.float32)
    rx/= np.linalg.norm(rx)
    ry = np.array([1,1,0], np.float32)
    ry/=np.linalg.norm(ry)
    rz = np.array([-1, 1, 0],dtype = np.float32)
    rz/=np.linalg.norm(rz)
    print(np.dot(rx,ry))
    print(np.dot(ry, rz))
    print(np.dot(rx, rz))
    assert(np.dot(rx,ry) == 0 and np.dot(ry, rz) == 0 and np.dot(rx, rz) == 0)
    Rwc2 = np.stack([rx, ry, rz], axis=1)
    print("here: ", Rwc2)
    cam2Points = (worldPoints - Ct2.T)@Rwc2
    print("cam points thing: ", cam2Points[0, 0]*rx + cam2Points[0, 1]*ry + cam2Points[0, 2]*rz + Ct2.T)
    R2cam = Rwc2.T
    t2c = (-Rwc2.T@Ct2)[:, 0]
    I  = np.identity(3)
    z = np.zeros(shape = (3,))
    visualize_3d_points_and_cameras(worldPoints, [(I, z), (R1cam, t1c)])
    visualize_3d_points_and_cameras(worldPoints, [(I,z),(R2cam, t2c)])
    visualize_3d_points_and_cameras(worldPoints, [(R1cam, t1c), (R2cam, t2c)])
    homWorld = np.concatenate([worldPoints, np.ones((worldPoints.shape[0], 1))], axis=1)
    print(homWorld)
    extMat1 = np.concatenate([R1cam, t1c[:, np.newaxis]], axis=1)
    extMat2 = np.concatenate([R2cam, t2c[:, np.newaxis]], axis=1)
   # visualize_3d_points_and_cameras(X[:, 0:3], [(np.identity(3), np.zeros(shape = (3,))), (R, t)])
    print("ext mat2: ", extMat2)
    P1 = K@extMat1
    P2 = K@extMat2
    print("P1: ", P1)
    print("P2: ", P2)
    print("hom wo rld: ", homWorld)
    proj1 = homWorld@P1.T
    print("proj1 pre div: ", proj1)
    proj1/=proj1[:, 2][:, np.newaxis]
    proj2 = homWorld @P2.T
    print("proj2 pre div: ", proj2)
    proj2/=proj2[:, 2][:, np.newaxis]
    print("Proj1: ", proj1)
    print("proj2: ", proj2)

    normCoords1= proj1 @ np.linalg.inv(K).T
    normCoords2 = proj2 @ np.linalg.inv(K).T
    print()
    #F = estimate_fundamental_matrix(proj1[:, 0:2], proj2[:, 0:2])
    #E, R, t =  calculate_essential_matrix(normCoords1, normCoords2)
    E = estimateEssentialMatrix(normCoords1, normCoords2)
    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
    print(proj1.shape)
    print(proj2.shape)
    #Check epipolar constraint.
    print(np.diag(proj2@F@proj1.T))
    s = 2
    print(F)

    R1,R2, t1, t2 = poseEstimationFromEss(E, normCoords1, normCoords2)
    """
    Rl = [R1, R2]
    tl = [t1, t2]
    count = []
    for i in range(4):
        R = Rl[i%2]
        t = tl[i>1]
        #need this because actually P1 isn't the identity. We have R and t relative to P1. 
        #erro;r here: t1 was getting overwritten!!!
        Rp = R @ R1cam
        tp = (R@ t1[:, np.newaxis] + s*t[:, np.newaxis])[:, 0]
        extMat2 = np.concatenate([Rp, tp[:, np.newaxis]], axis=1)
        #extMat2 = np.concatenate([R, s*t[:, np.newaxis]], axis=1)
        P2 = K@extMat2
        
        X = computeDLT(proj1, proj2, P1, P2)[:, 0:3]
        depth = (X@ Rp.T + s*tp)[:, 2]
        otherDepth = (X@P1.T@np.linalg.inv(K).T)[:, 2]
        #otherDepth = X[:, 2]
        #print("OTher depth: ", otherDepth)
        val = np.sum(np.logical_and(depth>=0, otherDepth>=0))
        print(val)
        count.append(val)
        #visualize_3d_points_and_cameras(X, [(np.identity(3), np.zeros(shape = (3,))), (R, s*t)])
    
    #once I get R and t of second coords relative to the first one. I think it's the secon d] 
    countArr = np.array(count)
    maxCount = np.max(countArr)
    i = np.argmax(countArr)
    print("Max count: ", maxCount)          
    print("i: ", i)
    R = Rl[i%2]
    t = tl[i>1]
    #WITH SCALING THIS WORKS. Makes sense because we get a value for t without knowing the specific distances? 
    #essential matrix doesn't provide scale. 
    Rp = R @ R1cam
    tp = (R@ t1[:, np.newaxis] + s*t[:, np.newaxis])[:, 0]
    extMat2 = np.concatenate([Rp, tp[:, np.newaxis]], axis=1)
    #extMat1 = np.concatenate([np.identity(3), np.zeros((3,1))], axis=1)
    #P1 = K@extMat1
    P2 = K@extMat2
    visualize_3d_points_and_cameras(worldPoints, [(I, z), (R, s*t)])
    visualize_3d_points_and_cameras(worldPoints, [(R1cam, t1), (R@R1cam, (R@t1[:, np.newaxis] + s*t[:, np.newaxis])[:, 0])])
    TDCoords = computeDLT(proj1, proj2, P1, P2)
    print(TDCoords)
    pixels1 = TDCoords @ P1.T
    print(pixels1)
    pixels2 = TDCoords @ P2.T
    print(pixels2)
    #doesn't work well.
    points3d = cv2.triangulatePoints(projMatr1=extMat1, projMatr2 = extMat2, projPoints1 = normCoords1[:, 0:2].T, projPoints2 = normCoords2[:, 0:2].T).T
    print(points3d)
    visualize_3d_points_and_cameras(TDCoords, [(R1cam, t1),(Rp, tp)])
    """
    P2a, Rp, tp = poseEstimation(proj1, proj2, R1, R2, t1, t2, P1, K, None, s,False)
    X = computeDLT(proj1, proj2, P1, P2a)
    print(np.linalg.norm(X[:, 0:3] - worldPoints, axis=-1))
    visualize_3d_points_and_cameras(X[:, 0:3], [(R1cam, t1c), (Rp, tp)])
def poseEstimation(x1, x2,R1, R2, t1, t2, P1, K1, K2, s, disp=False):
    """
    x1 and x2 are pixel homogenous coordinates (already normalized)
    R1, R2, t1, t2 are the poses we wish to test. Note these are relative to the first camera. 
    P1 is the projection matrix of the first camera. Depending if we have a world coordinate system already set up, this may be 
    K1, K2 are the intrinsic matrices of the cameras. 
    """
    if P1 is None:
        P1 = K1 @ np.concatenate([np.identity(3), np.zeros(shape = (3,1))], axis=1)
    if K2 is None:
        K2 = K1
    Rl = [R1, R2, -R1, -R2]
    tl = [t1, t2]
    Rt1 = np.linalg.inv(K1)@P1
    #could do this comp more fancy, but this is simpler. 
    R1c, t1c = Rt1[:, 0:3], Rt1[:, 3]
    print("Det of Rc: ", np.linalg.det(R1c))
    count = []
    for i in range(len(Rl)):
        for j in range(len(tl)):
            R = Rl[i]
            t = tl[j]
            print("Det R: ", np.linalg.det(R))
            #need this because actually P1 isn't the identity. We have R and t relative to P1. 
            Rp = R @ R1c
            tp = (R@ t1c[:, np.newaxis] + s*t[:, np.newaxis])[:, 0]
            extMat2 = np.concatenate([Rp, tp[:, np.newaxis]], axis=1)
            P2 = K2@extMat2
            
            X = computeDLT(x1, x2, P1, P2)
            #depth = (X@ Rp.T + s*tp)[:, 2]
            depth2 = (X@P2.T@np.linalg.inv(K2).T)[:, 2]
            depth1 = (X@P1.T@np.linalg.inv(K1).T)[:, 2]
            print(depth2)
            print(depth1)
            val = np.sum(np.logical_and(depth1>=0, depth2>=0))
            print("Num both in front: ", val)
            count.append(val)
            if disp:
                visualize_3d_points_and_cameras(X, [(R1c, t1c), (Rp, tp)])
    countArr = np.array(count)
    maxCount = np.max(countArr)
    i = np.argmax(countArr)
    print("Max count: ", maxCount)          
    print("i: ", i)
    R = Rl[math.floor(i/len(tl))]
    t = tl[i%len(tl)]
    #need this because actually P1 isn't the identity. We have R and t relative to P1. 
    Rp = R @ R1c
    tp = (R@ t1c[:, np.newaxis] + s*t[:, np.newaxis])[:, 0]
    extMat2 = np.concatenate([Rp, tp[:, np.newaxis]], axis=1)
    P2 = K2@extMat2
    
    return P2, R, t

def loadImages(name, numIm):#
    listFrames = []
    for i in range(numIm):
        fullName = name + str(i) + ".jpg"
        frame = cv2.imread(fullName)
        print("frame shape: ", frame.shape)
        listFrames.append(frame)
    return listFrames
def rotation_matrix_to_rpy(R):
    """
    Converts a 3x3 rotation matrix to roll, pitch, and yaw angles.
    Args:
        R: 3x3 rotation matrix
    Returns:
        roll, pitch, yaw (in radians)
    """
    # Check for gimbal lock
    if np.isclose(R[2, 0], -1.0):  # r31 = -1
        pitch = np.pi / 2
        roll = 0
        yaw = np.arctan2(R[0, 1], R[0, 2])
    elif np.isclose(R[2, 0], 1.0):  # r31 = 1
        pitch = -np.pi / 2
        roll = 0
        yaw = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        # Normal case
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return roll, pitch, yaw

def getImMatches(kp1, des1, kp2, des2, bf):
    """
    TODO: Edit this method so it returns a mask of which indices were included and which weren't. 
    Adjust other method because the unique is messing up the ordering. 
    """
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    #print("matches: ", matches)
    #print("here")
    #print(len(kp1))
    #print(len(kp2))
    #print(des1)
    query1= np.array([mat.queryIdx for mat in matches])
    query2 = np.array([mat.trainIdx for mat in matches])
    queries1 = np.zeros((des1.shape[0],),dtype = int)
    queries2 = np.zeros((des2.shape[0],), dtype = int)
    queries1[query1] = 1
 
    queries2[query2] = 1

    kp1mat = np.asarray([kp1[i].pt for i in range(des1.shape[0])])
    kp2mat = np.asarray([kp2[i].pt for i in range(des2.shape[0])])
    
    # Extract matched keypoints
    list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
    list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

    matches_kp1 = np.asarray(list_kp1)
    matches_kp2 = np.asarray(list_kp2)
    print(matches_kp1[0:10])
    print(kp1mat[query1] [0:10])
    print(matches_kp1.shape)
    print(matches_kp2.shape)
    
    # Remove duplicate matches
    combine_reduce = np.unique(np.concatenate((matches_kp1, matches_kp2),
                                            axis=1),
                            axis=0)
    points1 = combine_reduce[:, 0:2]
    points2 = combine_reduce[:, -2:]
    print("points1: ", points1[0:10])
    return points1, points2, query1, query2
def getImMatchesAlt(kp1, des1, kp2, des2, bf):
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    #print("matches: ", matches)
    #print("here")
    #print(len(kp1))
    #print(len(kp2))
    #print(des1)
    query1= np.array([mat.queryIdx for mat in matches], dtype =int)
    query2 =  np.array([mat.trainIdx for mat in matches], dtype=int)
    queries1 = np.zeros((des1.shape[0],),dtype = int)
    queries2 = np.zeros((des2.shape[0],), dtype = int)
    queries1[query1] = 1
    print(queries1)
    queries2[query2] = 1
   
    kp1mat = np.asarray([kp1[i].pt for i in range(des1.shape[0])])
    kp2mat = np.asarray([kp2[i].pt for i in range(des2.shape[0])])

    
    return kp1mat, kp2mat, query1, query2
def myPlotEpi(image1, image2, points1, points2, F):
    lines = points1 @ F.T
    lines
    image2_with_lines = image2.copy()

def plot_epipolar_lines(image1, image2, points1,points2, F):
    """
    Plots epipolar lines on the second image given points in the first image and the fundamental matrix.
    Args:
        image1: First image (as a numpy array).
        image2: Second image (as a numpy array).
        points1: Nx2 array of points in the first image.
        F: Fundamental matrix (3x3).
    """
    # Ensure the images are in color (for visualization)
    if len(image1.shape) == 2:  # Grayscale
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:  # Grayscale
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # Compute epipolar lines in the second image for points in the first image
    points1_homogeneous = np.hstack([points1, np.ones((points1.shape[0], 1))])  # Convert to homogeneous coordinates
    epilines2 = F @ points1_homogeneous.T  # Compute epipolar lines in the second image
    epilines2 = epilines2.T  # Shape: Nx3
    square = np.sqrt(np.sum((epilines2**2)[:, 0:2], axis=-1))
    print(epilines2)
    epilines2 = epilines2/square[:, np.newaxis]
    print(epilines2)
    # Plot the epipolar lines on the second image
    image2_with_lines = image2.copy()
    for r in epilines2:
        # Line equation: ax + by + c = 0
        a, b, c = r
        x0, y0 = 0, int(-c / b)  # Line intersects the left border (x = 0)
        x1, y1 = image2.shape[1], int(-(c + a * image2.shape[1]) / b)  # Line intersects the right border (x = width)
        color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for each line
        cv2.line(image2_with_lines, (x0, y0), (x1, y1), color, 1)

    # Plot the images
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.scatter(points1[:, 0], points1[:, 1], c='red', s=50, label='Points in Image 1')
    plt.legend()
    plt.title("Image 1 with Points")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image2_with_lines, cv2.COLOR_BGR2RGB))
    plt.scatter(points2[:, 0], points2[:, 1], c='red', s=50, label='Points in Image 2')
    plt.title("Image 2 with Epipolar Lines")
    plt.axis("off")

    plt.show()
def display_keypoint_matches(img1, img2, points1, points2):
    """
    Displays keypoint matches between two images.
    Args:
        img1: First image (as a numpy array).
        img2: Second image (as a numpy array).
        points1: Nx2 array of keypoints in the first image.
        points2: Nx2 array of keypoints in the second image.
    """
    # Ensure the images are in color (for visualization)
    if len(img1.shape) == 2:  # Grayscale image
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:  # Grayscale image
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Create a combined image by stacking the two images side by side
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    combined_img = np.zeros((height, width, 3), dtype=np.uint8)
    combined_img[:img1.shape[0], :img1.shape[1]] = img1
    combined_img[:img2.shape[0], img1.shape[1]:] = img2

    # Shift points2 coordinates to the second image's position
    points2_shifted = points2 + np.array([img1.shape[1], 0])

    # Draw matches
    for pt1, pt2 in zip(points1, points2_shifted):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for each match
        cv2.line(combined_img, pt1, pt2, color, 1)
        cv2.circle(combined_img, pt1, 5, color, -1)
        cv2.circle(combined_img, pt2, 5, color, -1)

    # Display the combined image
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Keypoint Matches")
    plt.show()

def plot_camera(ax, R, t, scale=0.1, color='blue'):
    """
    Plots a camera in 3D space using its rotation and translation.
    Args:
        ax: The matplotlib 3D axis.
        R: Rotation matrix (3x3).
        t: Translation vector (3,).
        scale: Scale of the camera visualization.
        color: Color of the camera frame.
    """
    # Camera center in world coordinates
   
    camera_center = -R.T @ t
    

    # Camera axes
    x_axis = camera_center + scale * R.T @ np.array([1, 0, 0])
    y_axis = camera_center + scale * R.T @ np.array([0, 1, 0])
    z_axis = camera_center + scale * R.T @ np.array([0, 0, 1])
  
    # Plot the camera center
    ax.scatter(*camera_center, c=color, label='Camera Center')

    # Plot the axes
    ax.plot([camera_center[0], x_axis[0]], 
            [camera_center[1], x_axis[1]], 
            [camera_center[2], x_axis[2]], c='red')
    ax.plot([camera_center[0], y_axis[0]], 
            [camera_center[1], y_axis[1]], 
            [camera_center[2], y_axis[2]], c='green')
    ax.plot([camera_center[0], z_axis[0]], 
            [camera_center[1], z_axis[1]], 
            [camera_center[2], z_axis[2]], c='blue')
    
def visualize_3d_points_and_cameras(points_3d, cameras):
    """
    Visualizes 3D points and camera positions in 3D space.
    Args:
        points_3d: Nx3 array of 3D points.
        cameras: List of tuples (R, t) where R is a 3x3 rotation matrix
                 and t is a 3x1 translation vector.
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    points_3d = np.array(points_3d)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='black', s=10, label='3D Points')

    # Plot cameras
    for i, (R, t) in enumerate(cameras):
        plot_camera(ax, R, t, scale=0.5, color=f'C{i}')

    # Set labels and equal scaling
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Points and Camera Visualization")
    ax.legend()
    plt.show()

doVisualOdometry()