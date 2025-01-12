import numpy as np
import math


def undistort(K, distCoeffs, Knew, R, imSize):
    """
    Implementation of initUndistortRectifyMap using numpy. 
    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html


    used in original method writing  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    """

    #Step 1: Get all pixel coordinates in vector form.
    #Im Size is w x h (we input w x h into this method so that this works out. )
    #had error mixing up xR and yR
    xR = np.arange(imSize[0])
    yR = np.arange(imSize[1])
    #first index is in row direction ,second in column direction, must be careful. 
    yv, xv= np.meshgrid(yR, xR)
    print(xv)
    print(yv)
   
    #BE CAREFUL. Because x coordinate first, the first input to array is row but that's second dim. 
    grid = np.stack([xv, yv, np.ones(imSize)], axis=2)
    print(grid[700, 100,:])
    print("here is gtrid : ", grid)
    normalizedGrid = grid@np.linalg.inv(Knew).T
    print(normalizedGrid.shape)
    #has to do with rectification
    rotatedGrid = normalizedGrid @ np.linalg.inv(R).T
    #Now, the coordinates aren't homogenous anymore. Fix this. 
    homogenousNewGrid = rotatedGrid/rotatedGrid[:, :, 2][:, :, np.newaxis]
    #gets r squared value for each pixel. 
    #instead can't we just do homogrid * homogrid sum first two values, 
    prod = homogenousNewGrid[:, :, :, np.newaxis] @ (homogenousNewGrid[:, :, np.newaxis, :])
    print("Prod shape: ", prod.shape)
    rs = np.trace(prod, axis1=2, axis2=3) - 1
    print(rs[1,7])
    print(np.dot(homogenousNewGrid[1,7], homogenousNewGrid[1,7])-1)
    k1 = distCoeffs[0]
    k2 = distCoeffs[1]
    p1 = distCoeffs[2]
    p2 = distCoeffs[3]

    k3 = 0
    k4 = 0
    k5 = 0
    k6 = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    tx = 0
    ty = 0
    if len(distCoeffs)>4:
        k3 = distCoeffs[4] 
    if len(distCoeffs)>5: 
        k4 = distCoeffs[5]
        k5 = distCoeffs[6]
        k6 = distCoeffs[7]
    if len(distCoeffs)>8:
        s1 = distCoeffs[8]
        s2 = distCoeffs[9]
        s3 = distCoeffs[10]
        s4 = distCoeffs[11]
    if len(distCoeffs)>12:
        tx = distCoeffs[12]
        ty = distCoeffs[13]
    coeffRadial = (1 + k1*rs + k2* rs**2 + k3*rs**3)/(1+ k4*rs + k5*rs**2 + k6*rs**3)
    #extra dim which doesn't mean anything? 
    uRad = (coeffRadial[:, :, np.newaxis]*homogenousNewGrid)[:, :, 0:2]
    
    uTanx = 2*p1*np.prod(homogenousNewGrid, axis=2) + p2*(rs + prod[:, :, 0, 0])
    uTany = 2*p2*np.prod(homogenousNewGrid, axis=2)+ p1*(rs + prod[:, :, 1,1])
    scaleX = s1*rs + s2*rs**2
    scaleY = s3*rs + s4*rs**2
    uTan = np.stack([uTanx + scaleX, uTany + scaleY], axis=2)
    xypp = uRad + uTan

    Rty = np.array([[math.cos(ty), 0, -math.sin(ty)],[0,1,0], [math.sin(ty), 0, math.cos(ty)]])
    Rtx = np.array([[1, 0, 0],[0,math.cos(tx),math.sin(tx)], [-math.sin(tx), 0, math.cos(tx)]])
    Rtytx = Rty @ Rtx
    Rc = np.array([[Rtytx[2,2], 0, -Rtytx[0, 2]],[0, Rtytx[2,2], -Rtytx[1,2]], [0, 0, 1]])
    print(Rc.shape)
    print(Rtytx.shape)
    print(xypp.shape)
    #homogenous operation

    #If existing diimensions, use CONCATENATE. ONly use stack if adding new dim, all other dimensions are identical. 
    
    pDist = np.concatenate([xypp, np.ones(shape = imSize)[:, :, np.newaxis]], axis=2) @ Rtytx.T @ Rc.T
    #divide out scale factor
    homoDist = pDist/pDist[:, :, 2][:, :, np.newaxis]
    #convert from normalized coordinates to pixel coordinates. 
    pixelDist = (homoDist @ K.T)[:, :, 0:2]
    print("Pixel dist shape: ", pixelDist.shape)
    #this returns our map x and map y. Now return 600 x 800
    return pixelDist[:, :, 0].T, pixelDist[:, :, 1].T
def remap(distIm, mapx, mapy, interpolation, borderMode):
    """
    Implementation of openCV remap method, to be applied after undistort. 
    distIm - distorted image to be undistorted. 
    map1, map2: for each pixel of the output image, it gets the corresponding "distorted image pixel location". May not be an integer.
    map 1 is the distorted x coordinate for each undistorted pixel location, map2 is distorted y coordinate.  

    Goal: For each output pixel, have corresponding xy pixel coord in distorted image. Might be noninteger. So use interpolation to fill in the image values
    at that point in the new image. 

    For each pixel, let I1 be the pixel rounding both numbers down. Let deltaU and deltaV displacement from four neighboring pixels. 
    Bilinear: Take weighted average of the four surrounding pixels. 
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

    Assuming the image is in size w,h

    Here assume image is in size h,w
    """
    h,w, c = distIm.shape
    undistortedImage = np.zeros(distIm.shape)
    #top left coordinates of pixel. Could be negative, in which case further away. 
    lowerX = np.floor(mapx).astype(int)
    lowerY = np.floor(mapy).astype(int)
    #x distance of pixel from nearest
    deltaX = (mapx - lowerX)[:, :, np.newaxis]
    #y distance of pixel from nearest. 
    deltaY = (mapy - lowerY)[:, :, np.newaxis]
    #bottom right corner of image. 
    upperX = np.ceil(mapx).astype(int)
    upperY = np.ceil(mapy).astype(int)

    #need to pad the input distIm array the appropriate amount. 
    #add ceiling s and floors so we get integers. 
    #want one more than the m ax value. 
    #should these not be size - 1. 
    maxX = max(int(np.max(upperX)), w-1) 
    maxY = max(int(np.max(upperY)), h-1) 
    minX = min(int(np.min(lowerX)), 0)
    minY = min(int(np.min(lowerY)), 0)
    #need to add one to maxX and minX because there IS an integer with u = np.max(upperX) so np.max + minX would get out of bounds. Need to go one beyond that. 
    paddedInput = np.zeros(shape = (maxY - minY+1, maxX - minX+1, 3))
    print("padded shape: ", paddedInput.shape)
    #goal of this is to convert the original image into this padded image. 
    # so if there are indices to access OUT of this range, then it will be 0
    paddedInput[ -minY:-minY + h, -minX:-minX + w, :] = distIm
    """
    print(paddedInput[-minX, -minY])
    print(paddedInput[-minX-1, -minY])
    print(paddedInput[-minX, -minY - 1, :])
    """
    #should adjust for padding. 
    """
    print("padded", paddedInput)
    print("low X", lowerX)
    print("low Y", lowerY)
    print("low X -", lowerX - minX)
    print("low Y -", lowerY - minY)
    print("biggest here:", np.max(lowerX - minX), np.max(lowerY-minY), np.max(upperX-minX), np.max(upperY-minY))
    """
    #need to index with two separate arrays. Not sure you can use one multidim array. 
    I1 = paddedInput[lowerY - minY, lowerX - minX,  :]
    I2 = paddedInput[upperY - minY, lowerX - minX, :]
    I3 = paddedInput[lowerY - minY, upperX - minX, :]
    I4 = paddedInput[upperY - minY, upperX - minX, :]
    
    #these are now the weighted pixel values of the resulting grid. Just need to round? 
    dist = (1-deltaX)*(1-deltaY)*I1 + (1-deltaX)*(deltaY)*I2 + (deltaX)* (1-deltaY)*I3 + (deltaX)*(deltaY)*I4
    print("Dist: ", dist)
    #not sure how to round here but this looks fine. 
    undistortedImage = np.round(dist)
    print("Rounded dist: ", undistortedImage)
    return undistortedImage
