def doSLAM():
    """
    Monocular SLAM: 
    Easy outline:
    1. Feature extraction
    2. Match features to previous image(s)
    3. Compute essential matrix between images. 
    4. Decompose essential matrix into pose. 
    5. Triangulate initial point cloud
    6. Check if good. 


    """
    numPics = 10
    location = "calibrationImages/frame"
    #calibrate somehow. Do i need the intrinsic matrix? 
    listImages, objPoints, imPoints = loadImages(numPics, location, imSize)
    imageShape = listImages[0][:, :, 0].shape
    #Need to undistort images. '

    
