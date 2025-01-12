
#https://en.wikipedia.org/wiki/Eight-point_algorithm
from flask import Flask, Response
import cv2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from camGeom import calculate_essential_matrix, normalizeHomogenousCoordinates, checkEpipolarConstraint
with open("info.txt", "r") as file: 
    text = file.read()
ESP32_CAM_IP = text
app = Flask(__name__)
K= np.array([[3400, 0, 816], [0, 3850, 616], [0, 0, 1]])
numKeypoints = 1000
orb = cv2.ORB_create(nfeatures = numKeypoints)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
def doSlam():
   
    listKeypoints = []
    listDescriptors = []
    listRotations = []
    listTranslations = []
    cap = cv2.VideoCapture(ESP32_CAM_IP + 'stream')
    #Capture the video feed from ESP32-CAM
    if not cap.isOpened():
        print("Error: Unable to connect to the ESP32-CAM stream.")
        return
    while True:
        ret, frame = cap.read()
        print(frame.shape)
        
        if not ret:
            break
        else:
            kp1, des1 = orb.detectAndCompute(frame, None)
            if len(listKeypoints) != 0: 
                
                kp2 = listKeypoints[-1]
                des2 = listDescriptors[-1]
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
                listRotations.append(R)
                #just using the scale factor to be the VALUE of t. 
                tadj = t/np.linalg.norm(t)
                listTranslations.append(tadj)
            listKeypoints.append(kp1)
            listDescriptors.append(des1)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to be displayed in the browser
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(doSlam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>ESP32-CAM Stream</h1><p>Go to <a href='/video_feed'>/video_feed</a> to view the stream.</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
