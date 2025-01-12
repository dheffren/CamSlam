
from flask import Flask, Response
import cv2
import requests
from PIL import Image
import numpy as np
#use localhost: 5000/video_feed to do stuff. 
#GPU ENV
app = Flask(__name__)
"""
need to figure out how to adjust the resolution on the camera, why it's so grainy, if we can run it without connecting to a computer, and if 
"""
# ESP32-CAM IP address (replace with your ESP32-CAM IP)
with open("info.txt", "r") as file: 
    text = file.read()
ESP32_CAM_IP = text
# Function to fetch the MJPEG stream from ESP32-CAM
def gen_frames():
    cap = cv2.VideoCapture(ESP32_CAM_IP + 'stream')
    #Capture the video feed from ESP32-CAM
    if not cap.isOpened():
        print("Error: Unable to connect to the ESP32-CAM stream.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame to be displayed in the browser
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h1>ESP32-CAM Stream</h1><p>Go to <a href='/video_feed'>/video_feed</a> to view the stream.</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug = True)
