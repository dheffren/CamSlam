import numpy as np
import cv2
import random
from skimage import img_as_float32
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
from skimage import filters
from skimage.color import rgb2gray
from skimage import feature
import math
from camGeomRepeat import normalize_coordinates

def normalizeHomogenousCoordinates(points, calMatrix):
    """
    Points are nx2
    Note the normalization they suggest on pg 282 of Harley is the SAME as this one. 
    """
    cx = calMatrix[0, 2]
    cy = calMatrix[1,2]
    fx = calMatrix[0,0]
    fy = calMatrix[1,1]
    normX = (points[:, 0] - cx)/fx
    normY = (points[:, 1] - cy)/fy
    ones = np.ones((points.shape[0], 1))
    pointsNhom = np.concatenate([normX[:, None], normY[:, None], ones], 1)

    #ALTERNATE DEFININTION(NOT FROM CHATGPT). 
    K1 = np.linalg.inv(calMatrix)
    print(K1)
    pointsHom = np.concatenate([points, ones], 1)
    #DO I NEED TO RENORMALIZE. NO BC WAY CAMERA MATRIX IS CONSTRUCTED. 
    pointsNhom2 = pointsHom @ K1.T


    #THE TWO METHODS ARE THE SAME, IF YOU CALCULATE OUT THE INVERSE. 
 
    return pointsNhom2
def calculate_essential_matrix(points1, points2):
    #assuming points1 and points2 are lists of keypoint matches. 
    """

    THIS METHOD IS COMPLETELY AND UTTERLY WRONG. NOT SURE WHY.
    USE ESTIMATE FUNDAMENTAL MATRIX INSTEAD. 
    Use the eight points algorithm 
    Put in normalized homogenous image coordinates. 
    points1: nummatches x 3 list of points
    points2: nummatches x 3 list of points


    IDEA: 
    y = (y1, y2, 1) y' = (y1', y2', 1) can write constraint as a dot product (overdetermined). 
    If Let E = e (e11 ... e13 e21 ...), and y~ =y'yT laid out same way.  then we have e*y~ = 0
    Each corresponding point is a vector. 


    How to solve homogenous linear equation: 
    e is right singular vector of Y corresponding to singular value = 0. Unique vector if at least 8 LI vectors. 
    If noise, may not be true. So find an e minimizing the square with norm 1. 
    So, choose e as left singular vector corresponding to smallest singular value of Y. 

    Need 2 sv = and nonzero, other 0. If need it to satisfy, find matrix E' of rank 2 minimizing distance from previous estimation. 
   [-0.08896872  0.79528298 -0.23954101 -0.37062592 -0.35259264  0.01169702
  0.09753705  0.17500507 -0.01635358]
  [[ 0.08896872 -0.79528298  0.23954101]
 [ 0.37062592  0.35259264 -0.01169702]
 [-0.09753705 -0.17500507  0.01635358]]
 get same singular vector both ways. 
     """
    #need to get in normalized coordinates which means I need the camera matrix which means I need focal length, distortion and center of image in pixels. 
   
    # have n x 3, nx 3. Want outer product of each pair. 
    #i had it poitns2 poitns 1 after figuring it out, but now trying original. 
    #switch the order to get points2 = y~, points1 = y
    #points1[:, 0:2], T = normalize_coordinates(points1[:, 0:2])
    #points2[:, 0:2],T = normalize_coordinates(points2[:, 0:2])
    outerProd = points2[:, None, :]*points1[:, :, None]
    print("outerprod: ", outerProd.shape)
    #not sure if this will do what I want it to. 
    #print(points1[0, 1]*points2[0, 0])
    
    # Want each column of Y to represent one data point. 
    Y = np.reshape(outerProd, (outerProd.shape[0], -1)).T
    print(Y[:, 0])
    print(points2[0,0]*points1[0, 0])
    print(points2[0,0]*points1[0,1])

    #Now we have an 9xN array
    U, S, VT = np.linalg.svd(Y, full_matrices =True)
    print("U shape: ", U.shape)
    print("S shape: ", S.shape)
    print("VT shape: ", VT.shape)
    #column representing smallest sv. 
    print(S)
    leftSingularVec = U[:, -1]
    print("left singular vec: ", leftSingularVec)
    
    Eest = np.reshape(leftSingularVec, (3,3))
    print("this is estimated E: ", Eest)

    #Ve is already the hermitian conjugate. 
    Ue, Se, Ve = np.linalg.svd(Eest, full_matrices = True)
    print("SE: ", Se)
    m = (Se[0] + Se[1])/2
    Sm = np.diag(Se)
    #this is the differentiating step. 
    Sm[-1, -1] = 0
    Sm[0,0] = m
    Sm[1,1] = m
    E = Ue@Sm@Ve
   
    U, S, VT = np.linalg.svd(E, full_matrices = True)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    print(W)
    Z= np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    Sd = np.diag(S)
    tM = U@W@Sd@(U.T)
    #Problem is that the singular values are supposed to be equal and they're close to it but not exactly. 
    #print(U@W@ np.diag(S)@W.T@VT)
    #print((W.T @ np.diag(S)).T)
    tA = -1*VT.T@Z.T@ VT
    R = U@(W.T)@VT

    
    #print(np.linalg.det(R))
    #print(tM @ R)
    #print(U@np.diag(S)@VT)
    #print(tA@R)
    print("singular value check: ", S)
    #print(np.trace(tM@R))
    #print(np.trace(E))
    t1 = -tM[1, 2]
    t2 = tM[0, 2]
    t3 = -tM[0, 1]
    t  = np.array([t1, t2, t3])
    # R should represent the rotation from on epose to another.
    print("R: ", R)
    print("t: ", t)
    # t has an unknown scale factor we don't know how to calculate. 
    return E, R, t 
def estimateEssentialMatrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Try to implement this function as efficiently as possible. It will be
    called repeatedly for part IV of the project

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
    """
    ########################
    # TODO: Your code here #
    ########################

    # This is an intentionally incorrect Fundamental matrix placeholder
    #F_matrix = np.array([[0, 0, -.0004], [0, 0, .0032], [0, -0.0044, .1034]])
    n = points1.shape[0]
    assert(n == points2.shape[0])
    ones = np.ones(shape = (n,1))
    #added from extra credit. 
    points1, T1 = normalize_coordinates(points1[:, 0:2], math.sqrt(2))
    points2, T2 = normalize_coordinates(points2[:, 0:2], math.sqrt(2))

    homogenousPoints1 = np.hstack((points1, ones))
    homogenousPoints2 = np.hstack((points2, ones))
    assert(homogenousPoints1.shape == (n, 3))
    assert(homogenousPoints2.shape == homogenousPoints1.shape)
    
    #should coopy it to the right. 
    leftStack = np.tile(homogenousPoints1, reps = 3)
    
    #copy each element in row 3 times. 
    rightStack = np.repeat(homogenousPoints2, repeats = 3, axis=1)
    assert(rightStack.shape == (n, 9))
    assert(leftStack.shape == rightStack.shape)
    assert(np.isclose(rightStack[0, 6:],np.ones(shape = (3,)), atol = .0001).all())
    
    product = leftStack*rightStack
    
    assert(product.shape == (n,9))
    U,S, Vh = np.linalg.svd(product, compute_uv= True, full_matrices =False)
    E = Vh[-1,:]
    E = np.reshape(E, newshape = (3,3))
    #enforce the rank deficiency constraint. 
    U,S,Vh = np.linalg.svd(E, compute_uv = True, full_matrices = False)
    S[-1] = 0
    #enforces identical SV econstraint. Not sure if should have this. 
    m = (S[0] + S[1])/2
    #S[0] = m
    #S[1] = m
    E_matrix = U@np.diagflat(S)@Vh
    #have normalized F at this point
    #added from extra credit. 
    E_matrix= np.transpose(T2)@E_matrix@T1
    return E_matrix
def checkEpipolarConstraint(points1, points2, E):

    tol = .001
    #think i had this backwards. 
    epiCheck = np.diag(points2@ E @ points1.T)
    print(epiCheck)
    booleanCheck = np.abs(epiCheck) <=tol
    return booleanCheck
def computeDLT(points1, points2, P1, P2):
    """
    DLT: Used for triangularization. 
    We have x = aPX where P projection, a scale, X homogenous 3d point, x inhomogenous (bc scale factor). 
    We have x X PX = 0
    (x y 1) x (p1T X, ... p3TX] = blah = 0)
    Get AX = 0 using poitns from both images. 
    Minimize L2 norm of Ax subject to norm x = 1. This is equivalent to rayleigh quotient, ie smallest signular value, ie singular vector. 
    We need P for this to work. Don't we get it somehow from the pose estimation? However there is some ambiguity. 
    
    Homogenous normalized points1, homogenous normalized points2

    have nx3x3 matrix where each 3x3 is the cross product of that point times this other thing. 

    Problem with method: Was comparing homogenous coordinates to the matrix from pixel coordinates. 
    """
    #elementwise need 0,1 = -x[2] 0, 2 x[1] 1,2 = -x[0]
    #this allows us to write A as x1' mat @P1 
    print("Points 1 shape: ", points1.shape)
    #check this first
    points1Alt = computeLinCrossMulti(points1)
    points2Alt = computeLinCrossMulti(points2)

    print("Points1Alt: ", points1Alt[0:10])
    print("points1AltShape: ", points1Alt.shape)
    #cut out last row because lin combo of the first two. 
    points1Mat = (points1Alt @ P1[np.newaxis, :, :])[:, 0:2, :]
    points2Mat = (points2Alt @ P2[np.newaxis, :, :])[:, 0:2, :]
    print(points1[0,0]*P1[2, :] - P1[0, :])
    print(points1Mat[0,0])
    print(points1Mat[0])
    
    print("points1mat shape: ", points1Mat.shape)
    #nx4x4
    pointsMat = np.concatenate([points1Mat, points2Mat], axis=1)
    print("Points mat shape: ", pointsMat.shape)
    #NOW: Want to solve AX = 0 where X is the homogenous 3d point. 
    #use least squares. Right singular vector of A corresponding to smallest singular value. 
    U, S, VT = np.linalg.svd(pointsMat)
    print(U.shape)
    print(S.shape)
    print(VT.shape)
    print(U[0])
    print(S[0])
    print(VT[0])
    print(U[1])
    print(S[1])
    print(VT[1])
    Ua, Sa, VTa = np.linalg.svd(pointsMat[1, :, :], full_matrices=False)
    print(Ua)
    print(Sa)
    print(VTa)
    #
    #will want a ROW of VT for each. 
    solns = VT[:, -1, :]
    print(solns.shape)
    print("solns: ", solns)
    #This will get us the values for THe 3d coordinates. Then we can renormalize to get the actual 3d coordinates. 
    #once we have these, we have a point cloud. 
    #should be 4 dimensional. 
    X = solns / (solns[:, -1][:, np.newaxis])
    return X


def computeLinCrossMulti(points):
    print("Shape points: ", points.shape)
    pointsAlt = np.zeros(shape = (points.shape[0], 3, 3))
    pointsAlt[:, 0, 1] = -1*points[:, 2]
    pointsAlt[:, 0, 2] = points[:, 1]
    pointsAlt[:, 1, 2] = -1*points[:, 0]
    pointsAlt = pointsAlt - np.transpose(pointsAlt, [0, 2, 1])
    print(pointsAlt[0])
    return pointsAlt

def poseEstimationFromEss(E,  points1, points2):
    """

    If you have normalized camera matrices P = [I 0] P' = [R t] then essential matrix is [t]xR = R[R^T t]x
    Assumption: Need that E has 2 equal singular values, and one singular value of 0. 
    S = kUZU^T where U orthogonal. (can write any skew mat this way). 
    Note Z = diag(1,1,0)W up to sign. Up to scale, S = U diag(1,1,0) WU^T 
    E = SR = u diag(1,1,0) WU^T R): SVD of E with 2 equl singular values. 

    E = U diag(1,1,0) V^T (up to scaling).
    2 possible factorizations: 
    S = UZU^T
    R = UWV^T or UW^T V^T. 
    This ignores signs. 
    SR = UZU^T UWV^T = UZWV^T ZW = (1 0 0 = Udiag(1,1,0)VT = E
                                    0 1 0 
                                    0 0 0)
    ZWT = -diag(1,1,0)

    We get t = u3 last col of U. 
    Don't know sign of E or t. 
    So 4 choices for second camera matrix. 2 choices for R, + or - for t. 

    To test if point in front of both cameras, do we need to take two inner products and the one which are both positive works. 
    """
    #orthogonal. 
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    #skew symm
    Z= np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    #the assuption is E = diag(1,1, 0)V^T but this isn't really true. However U and V should be close. 
    U, S, VT =  np.linalg.svd(E, full_matrices = True)
    print("S is: ", S)
   
    R1 = U@W@VT
    R2 = U@W.T@VT
    Se = U@Z@U.T
    t1 = U[:, 2]
    t2= -t1

    #triangulation: 
    #if points1 is in normalized image coordinates. Scaling it backward will get the 3d  coordinate (assuming coord is I 0 for cam 1)
    #corresponding points2. Need to do pseudo-inverse of [R t] to get a 4d point then scale away. 
    p1 = points1[0]
    p2 = points2[0]
    #tdpoint1 = np.concatenate([np.identity(3), np.zeros(shape = (3,1))],axis=1).T @ p1[:, np.newaxis]
    tdpoint1 = np.concatenate([p1, np.ones((1,))], axis=0)
    print("td point 1: ", tdpoint1)
    mat = np.concatenate([R1, t1[:, np.newaxis]],axis=1)
    Matpinv = np.linalg.pinv(mat)
    print("Pseudo size: ", Matpinv.shape)
    """
    Ux, Ex, VTx = np.linalg.svd(mat, full_matrices = True)
    r = np.count_nonzero(Ex)
    Exi = np.zeros(shape = Ex.shape)
    Exi[0:r] = 1/Ex[0:r]
    Exp = np.diag(Exi)
    print(Exp.shape)
    pseud = VTx.T @ Exp @ Ux.T

    tdpoint2 = pseud @ p2[:, np.newaxis]
    """
    tdpoint2 = Matpinv @ p2[:, np.newaxis]
    print("td point2 : ", tdpoint2)
    print(R1.T@(p2[:, np.newaxis] - t1[:, np.newaxis]))
    #somehow need to check which ones work. 
    return R2, t1
def checkProjectionOnPointPair():
    return

    
def calcCrossProdLin(x):
    """
    X is a numpy array of size 3. 3x1 or 1x3 is fine.
    Makes skew symmetric matrix.  
    """
    xH = np.zeros((3,3))
    xH[0,1] = -x[2]
    xH[1,0] = x[2]
    xH[0,2] = x[1]
    xH[2,0] = -x[1]
    xH[1,2] = -x[0]
    xH[2,1] = x[0]

    return xH






