import numpy as np
import math
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import functools
from distortion import remap



def calibrateCamera():
    """
    Calibrate the camera using the Scaramuzza method. 
    Here are the steps: 
    1. Get calibration images from multiple different angles and places of the camera. 
    2. Use those values to solve for camera extrinsic parameters (for each view). Did this before but slightly different. 
    3. Solve for a0 ... aN in our mode.. Compute unknwon ti3 for each pose. 
    4. Refine estimation of extrinsic. Then refine intrinsic. Repeat. 
    5. Iterative search for center Oc. 
    6. Refine nonlinearly via maximum likelihood. 


    JUST DO THE UNDISTORT POINTS OR SWITCH TO OPEN CV


    8 x 11 
    """
    path = "calibrationImages/"
    nrows = 8
    ncols = 11
    listImages = []
    listPoints = []
    i = 0
    for file in os.listdir(path):
        if i ==0:
            i+=1
            continue
        print(path + file)
        image = cv2.imread(path + file)
        #kernel = np.array([[-1, -1, -1], [-1,9,-1], [-1, -1,-1]])
        #image = cv2.filter2D(image, -1, kernel)
        #print(image.shape)
        plt.imshow(image)
        plt.show()
        # Color-segmentation to get binary mask
        
        lwr = np.array([0, 0, 143])
        upr = np.array([179, 61, 256])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        plt.imshow(hsv)
        plt.show()
        
        msk = cv2.inRange(hsv, lwr, upr)
        plt.imshow(msk)
        plt.show()
        # Extract chess-board
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dlt = cv2.dilate(msk, krn, iterations=5)
        res = 255 - cv2.bitwise_and(dlt, msk)
        plt.imshow(msk)
        plt.show()
        # Displaying chess-board features
        res = np.uint8(res)# Color-segmentation to get binary mask
       
        """
        """
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.equalizeHist(gray)

        # Apply Gaussian blur or Median blur to reduce noise
        #res= cv2.GaussianBlur(gray, (3,3), 0)
        #plt.imshow(res, cmap="gray")
        #plt.show()
        ret, corners = cv2.findChessboardCorners(res, (5,5),
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
        print(corners)
        fnl = cv2.drawChessboardCorners(image, (5,5), corners, ret)
        plt.imshow( fnl)
        plt.show()
        if not ret:
            print("Didn't find checkerboard")
            assert(False) 

        print(corners.shape)
        listPoints.append(corners)
        listImages.append(np.array(Image.open(path + file)))
        i+=1
def loadImages():
    listImages = []
    path = "calibrationImages/"
    for file in os.listdir(path):
        image = cv2.imread(path + file)
        listImages.append(image)
    return listImages
def doUndistort():
    listImages = loadImages()
    imHeight = listImages[0].shape[1]
    imWidth = listImages[0].shape[0]
    #vals = (imWidth*imHeight*np.arange(0, 1, .05)).astype(int)
    #val = 26*imHeight + 150
    a = np.array([110.0330, 0.0031, 0, 0])
    t = np.array([165.9686, 125.9877])
    A = np.array([[1,0], [0,1]])
    scaleFactor = .85
    h,w = listImages[0].shape[0:2]
    mapX, mapY, origGrid = undistortFisheye((w,h), A, t, a, scaleFactor)
    im = remap(listImages[0], mapX, mapY, None, None)
    """"""
    #print(vals)
    #print(imWidth*imHeight)
    #originalImageIndices = origGrid[vals]
    #print(originalImageIndices)
    #valsWeWant = [vals]
    #print(valsWeWant)
    fig, ax = plt.subplots()
    ax.imshow(listImages[0])
    #x = originalImageIndices[:, 0]
    #y = originalImageIndices[:, 1]
    #xh = valsWeWant[:, 0]
    #yh = valsWeWant[:, 1]
    #ax.scatter(y,x, color = 'red', marker = 'o')
    #ax.scatter(yh, xh, color = "blue", marker = "x")
    #ax.scatter(im[val][1], im[val][0], color = 'green', marker='o' )
    #ax.scatter(origGrid[val][1], origGrid[val][0], color = 'purple', marker='x' )
    cv2.imwrite('fisheyeresult.png', im)
    coords = []
    cid = fig.canvas.mpl_connect('button_press_event', functools.partial(onclick, fig = fig, ax = ax, coords=coords, im=im, imWidth = imWidth, imHeight = imHeight))
    plt.show()
    
def makeOnClick(fig, ax):
    return onclick(fig, ax)
def onclick(event, fig, ax, coords, im, imWidth, imHeight):
    ix, iy = event.xdata, event.ydata
    print (f'x = {ix}, y = {iy}')
    #index = int(iy)*imHeight + int(ix)
    index = int(iy) + int(ix)*imWidth
    ax.scatter(im[iy,ix], im[iy, ix], color = 'green', marker='o' )
    ax.scatter(ix,iy , color = 'purple', marker='x' )
    
    plt.show()
def distortionOpenCV(images):
    """
    images
    """
    here = cv2.fisheye.calibrate(images, points, size)
    return

def undistortFisheye(imSize, A, t, a, scaleFactor):
    f = min(imSize[0], imSize[1])/2
    #where did i get this? 
   
    #principalPoints = np.zeros((2,))
    #focalLength = np.array([im.shape[0], im.shape[1]])/2 *scaleFactor
    focalLength = f *scaleFactor
    #is image transposed yet? CHGECK MESHGRID DIMS. 
    xR = np.arange(imSize[0])
    yR = np.arange(imSize[1])
    print(xR.shape)
    print(yR.shape)
    yv, xv = np.meshgrid(yR, xR)
    print(xv)
    print(yv)
    #Kh = np.identity(2)*focalLength
    #need to alter if different A. 
    K = np.array([[focalLength,0, t[0]],[0, focalLength, t[1]], [0, 0, 1]])
    


    print(xv.shape)
    print(yv.shape)
    #grid = np.reshape(np.stack([xv, yv], axis=2), (-1, 2))
    grid = np.stack([xv, yv, np.ones(imSize)], axis=2)
    origGrid = np.copy(grid[:, :, 0:2])
    #convert from pixel coordinates to normalized image coordinates. 
    grid =  grid@np.linalg.inv(K).T
    prod = grid[:, :, :, np.newaxis] @ (grid[:, :, np.newaxis, :])
    print("Prod shape: ", prod.shape)
    r = np.sqrt(np.trace(prod, axis1=2, axis2=3) - 1)
    #
    s = polyInv(a, r)
    scaledPixels = grid*s[:, :, np.newaxis]
    gridScaled = scaledPixels[:, :, 0:2]
    

    
    
    print("Grid pixel scale not transformed: ", gridScaled)
    #Need to alter A somewhere here. This doesn't make sense because I don't know hte difference between the two camera matrices!

    grid = (gridScaled@A.T+ t)
    mapX = grid[:, :, 0]
    mapY = grid[:, :, 1]
    print("Pixel scale: ",grid)
    return mapX.T, mapY.T, origGrid
def polyInv(a, r):
    
    pList = []
    rows = r.shape[0]
    #print(r)
    r = np.reshape(r, (-1, ))
    for i in range(r.shape[0]):
       # print(r[i])
        poly = np.poly1d([a[3]*r[i]**4, a[2]*r[i]**3, a[1]*r[i]**2, -1, a[0]])
       #poly = np.poly1d([a[3]*r[i]**4, a[2]*r[i]**3, a[1]*r[i]**2, -1, a[0]][::-1])
        roots = poly.roots
        pRoot= roots[-1]
        pList.append(pRoot)


    pp = np.array(pList, dtype = "float32")
    #hopefully preserves order. 
    p = np.reshape(pp, (rows, -1))
    return p
def calcPoly(a, p):
    """
    p is size N
    """   
  
    val = a[0] + a[1]*p**2 + a[2]*p**3 + a[3]*p**4
    return val

def estExtIn(m, M):
    """
    This is ONE image. 
    m is size N x 2
    M is size N x 2 (z coord is assumed 0) these are calculated by hand by measuring. 
    """
    M = np.concatenate([-m[:, 1]*M, m[:, 0]*M, -m[:, 1], m[:, 0]], axis=1)
    U, s, VT = np.linalg.svd(M)
    Hguess = U[:, -1]
    #know r1, r2 orthonormal so 
    r11 = Hguess[0]
    r21 = Hguess[2]
    r12 = Hguess[1]
    r22 = Hguess[3]
    dot = r11*r12 + r21*r22 
    r31r32 = -1*dot
    sq1 = r11**2  + r21**2
    sq2 = r12**2 + r22**2
    r31sq = .5*((sq2-sq1) + math.sqrt((sq2-sq1)**2 + 4*dot**2))
    c = math.sqrt(1/(sq1 + r31sq))
    r31 = math.sqrt(r31sq)
    r32 = r31r32/r31

    R = np.zeros(shape = (3, 2))
    R[0, 0] = r11
    R[0, 1] = r12
    R[1, 0] = r21
    R[1,1] = r22
    R[2, 0] = r31
    R[2, 1] = r32
    R = c*R
    t1t2 = c*np.array(Hguess[4:])
    
    return R, t1t2
def estIntIn(m, M, R, t1t2, deg):
    """
    m: K x N x 2
    M : K x N x 2
    R: K x 3 x 2
    t1t2: K  x 2
    """
    #K x Nx 3 vector of these things. 
    CABD = R[:, np.newaxis, :, :]@M

    A= CABD[:, :, 1] + t1t2[:, np.newaxis, 1]
    C = CABD[:, :, 0] + t1t2[:, np.newaxis, 0]
    D = m[:, :, 0]*CABD[:, :, 2] 
    B = m[:, :, 1]*CABD[:, :, 2]
    #this shoudl get the square? 
    P = np.sqrt(m@m.T)
    K = m.shape[0]
    N = m.shape[1]
    bigMat = np.zeros(shape = (K*N, deg+K))
    outMat = np.zeros(shape = (K*N, ))
    for i in range(K):
        Ai = A[i, :].T
        Ci = C[i, :].T
        Bi = B[i, :].T
        Di = D[i, :].T
        index = i*N*2
        bigMat[index:index + N, 0] = Ai
        bigMat[index + N: index + 2*N, 0] = Ci
        outMat[index: index + N] = Bi
        outMat[index + N: index + 2*N] = Di
        rhoi = P[i].T
        for j in range(2, deg+1):
            bigMat[index: index + N, j-1] = Ai * np.power(rhoi, j)
            bigMat[index + N: index + 2*N, j-1] = Ci * np.power(rhoi, j)
        bigMat[index: index + N, N + i] = -m[i, :, 1]
        bigMat[index + N: index + 2*N, N+i] = -m[i, :, 0]
    params = np.linalg.pinv(bigMat)@outMat
    return params

def estExt(m, a):
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



#calibrateCamera()\
doUndistort()