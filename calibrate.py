from flask import Flask, Response
import cv2
import requests
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from distortion import undistort, remap
with open("info.txt", "r") as file: 
    text = file.read()
ESP32_CAM_IP = text
app = Flask(__name__)
def mainCalibration():
    """
    Goal: Calibrate a monocular camera. (Distorted or undistorted)
    Do this using openCV

    Step 1: Obtain images of multiple checkerboard patterns from multiple different angles. Find  corner points of pattern. 
    
    Step 2: 
    """
    numPics = 10
    location = "calibrationImages/frame"
    imSize = (15,22)
    listImages, objPoints, imPoints = loadImages(numPics, location, imSize)
    imageShape = listImages[0][:, :, 0].shape
    
    #print(imPoints)
    #print(objPoints)
    #step 2: Calibrate camera

    #understand what all of these steps are doing. Perhaps implement them by yourself to truly understand. 
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imPoints, imageShape[::-1], None, None)
    img = listImages[0]
    h,w = img.shape[:2]
    """
    Returns an undistorted camera intrinsic matrix based on free scaling parameter. 
    Uses distortion coefficients. 
    img size is in x by y.

    The idea here is that 
    alpha is a scaling parameter between 0 and 1. 
    0 means image is zoomed in and shifted so that ONLY valid pixels are visible (ie no black areas after rectifying)
    1 means the rectified image is decimated and shifted so all pixels from original images from the cameras are retained in the rectified image.
    
    so alpha = 1 will have ALL the black areas which come from image undistortion. This means that ALL original pixels are remapped into the new image. 

    alpha = 0: the camera matrix implicitly resizes the image so that the only stuff within the pixels 0 : img Width, 0:img Height are VALID. Thus the roi values
    will be the entire UNDISTORTED OUTPUT. However, this output is a CROPPED version of the other ones. 
    As alpha increases, the image is more zoomed out. thus, the rectangle of "valid" points will DECREASE in size. However, the rectangles of these two alphas CORRESPOND to eachother at diff levels of zoom

    It seems like with alpha=0, the camera intrinsics of the undistorted image are close to those of the original image. 
    With alpha = 1, the focal length in the x and y directions have decreased. 

    It does seem as alpha inc, focal lengths dec. 
    not exactly sure what the center principal point means here. 
    print("original K: ", K)
    print("new k: ", newK)
    print("new K half: ", newKhalf)
    print("New k 0: ", newK0)
    print("ROI: ", roi)
    print("ROI half: ", roi2)
    print("ROI 0: ", roi0)

    This new matrix is an estimation of the camera matrix we DESIRE according to the value of alpha we chose. This is in order to make alll the pixels in the image or whatever. 
    """
    newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1, (w,h))
    newKhalf, roi2 = cv2.getOptimalNewCameraMatrix(K, dist, (w,h),.5, (w,h))
    newK0, roi0 = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
    
    """
    dst = cv2.undistort(img, K, dist, None, newK)
    
    """

    """
    Computes undistortion and rectification transformation map. 
    
    If camera matrix = new camera matrix with no distortion, then the undistorted image = distorted image. 

    For monocular camera: newCameraMatrix = cameraMatrix or computed by getOptCamMat for control over scaling. 
    In stereo, the new matrix is p1 or p2 computed by stereoRectify. 

    New camera is oriented according to R. 

    Builds map for inverse mapping algorithm used by remap. 

    Take a pixel (u,v) in the rectified (corrected) image. This function computes corresponding source coordinates. 
    Outline in words
    Take (u,v) \to (x,y) in normalized image coordinates
    change of basis from rectified camera coordiantes to unrectified coordinates. This has to do with stereo image calibration, where they warp both images so their new 
    epipolar lines are horizontal. This simplifies stereo matching but doesn't CHANGE it. It still searches on lines, whether they're horizontal or vertical. 
    convert back into Homogenous coordinates. 

    d

    Calculate the radial distances to get the distorted normalized image coordinates. 

    then, apply the PLANAR distortion. 
    
    Once this is done, get the mapping function by going from normalized coordinates to pixel coordinates on original distorted image. 
    so function from corrected image pixels to the distorted image values. 
    REMAP. 
    so for each of the pixels in the undistorted image, you find the nearest distorted image approximation to that via interpolation. (you get a distorted pixel which may be a decimal or may not be in the grid)
    if it's not in the grid, you return black. Otherwise you interpolate. Then, you have the new image filled out. 

    Note that there is no guarantee that the new undistorted image contains all the pixels or things from the initial image. This actually depends on updated camera matrix obtained which adjusts for the zoom factor. 
    If we end up getting a zoomed out image with black pixels, regardless, the returned image is the same size as the original. Thus, if we crop out the black areas, we get a smaller image. 


    """
    #mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newK, (w,h), 5)
    print(dist)
    print(dist.shape)
    #print("width by height: ", (w,h))
    #adding dist[0] to make dimensions work. 
    # now method comes out with h x w. But does calculations in w,h 
    mapx, mapy = undistort(K, dist[0], newK,np.identity(3), (w,h))

  
    #gets a distorted image. However, it is UNCROPPED. How so? 
    #I guess based on the camera matrix some points of the undistorted image may have NO points from the distorted image pointing to them. Thus, it would be black. 
    #(Basically if the answer is negative or bigger than the value). 
    #take transpose because want x then y. not sure if this is the best way to do this. 

    #my remap and cv2 remap look essentially the same. 
    #putting in shape of y x x
    dst = remap(img, mapx, mapy, cv2.INTER_LINEAR, None)
    #dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, None)
    #x, y are top left coord of rectangle. 
    #w,h are the width and height of the rectangle. 
    x, y, w, h = roi
    #crops the image so only VALID pixels are allowed. 
    #print("Uncropped shape: ", dst.shape)
    cv2.imwrite('calibresultUncropped.png', dst)
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)
    print("cropped shape: ", dst.shape)
    return K, newK, mapx, mapy,  roi, rvecs, tvecs
def undistortImage(im, mapx, mapy, roi):
    dst = remap(im, mapx, mapy, cv2.INTER_LINEAR, None)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    print(dst.shape)
    return dst
def loadImages(numPics, name, imSize, disp = False):
   
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    listImages = []
    objPoints = []
    imgPoints = []
    grayFrames = []
  
    objp = np.zeros((imSize[0]*imSize[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:imSize[0],0:imSize[1]].T.reshape(-1,2)
    for i in range (numPics):
        fullName = name + str(i) + ".jpg"
        frame = cv2.imread(fullName)
        print("frame shape: ", frame.shape)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("GRAY SHAPE" , grayFrame.shape)
        ret, corners = cv2.findChessboardCorners(grayFrame, imSize, None)
        if not ret:
            print("Corners not found")

            continue
        objPoints.append(objp)
        #check these parameters. This refines the corners of the guess. 
        
        corners2 = cv2.cornerSubPix(grayFrame, corners, (11,11), (-1,-1), criteria)
        imgPoints.append(corners2)
        listImages.append(frame)
        if disp:
            cv2.drawChessboardCorners(frame, imSize, corners2, ret)
            cv2.imshow("img", frame)
            cv2.waitKey(500)
    return listImages, objPoints, imgPoints
        
def captureImages():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    numPics = 10
    i= 0
    imSize = (15, 22)
    objp = np.zeros((imSize[0]*imSize[1],3), np.float32)
    cap = cv2.VideoCapture(ESP32_CAM_IP + 'stream')
    listImages = []
    objPoints = []
    imgPoints = []
    if not cap.isOpened():
        print("Error: Unable to connect to the ESP32-CAM stream.")
        return
    while i<numPics:
        print("i is: ", i)
        ret, frame = cap.read()
            
        if not ret:
            break
        else:
        
            #put frame in grayscale. 
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayFrame, imSize, None)
                # Encode the frame as JPEG
            ret1, buffer = cv2.imencode('.jpg', frame)
            frame2 = buffer.tobytes()

            # Yield the frame to be displayed in the browser
            yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n\r\n')
            if not ret:
                print("Corners not found")

                continue
            objPoints.append(objp)
            #check these parameters. This refines the corners of the guess. 
            corners2 = cv2.cornerSubPix(grayFrame, corners, (11,11), (-1,-1), criteria)
            imgPoints.append(corners2)
            listImages.append(frame)
            cv2.imwrite( "calibrationImages/frame" + str(i) + ".jpg", frame)
            i+=1


@app.route('/video_feed')
def video_feed():
    return Response(captureImages(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=5000, debug = True)
    mainCalibration()