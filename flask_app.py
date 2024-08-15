import cv2
import time
import threading
from flask import Response, Flask


import argparse
from utils.camera import CameraDisplay
from utils.yolo import nms, filter_boxes
from utils.viz import display_result
import torch
import time
import cv2
import onnxruntime as ort
from models.pruned_my_tinyyolo2 import PrunedMyTinyYoloV2
import numpy as np


global now
now = time.time()

global ort_sess
ort_sess = ort.InferenceSession('onnx/pruned.onnx')

# Image frame sent to the Flask object
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global thread_lock 
thread_lock = threading.Lock()

# GStreamer Pipeline to access the Raspberry Pi camera
GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'

# Create the Flask object for the application
app = Flask(__name__)

# Define a callback function (your detection pipeline)
# Make sure to first load all your pipeline code and only at the end init the camera
#%store -r net
def infer(image):
    global now, ort_sess

    fps = f"{int(1/(time.time() - now))}"
    now = time.time()
    image = image[0:320,0:320, :]
    #image2 = np.transpose(image, (-1, 0, 1))
    #image2 = np.expand_dims(image2, axis=0)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert image to torch
    # from 320 x 320 x 3 to 1 x 3 x 320 x 320
    torch_image2 = torch.from_numpy(image)
    torch_image = torch.zeros([1, 3, 320, 320])
    #torch_image = torch.zeros([1, 3, int(320 / downscale), int(320 / downscale)])
    
    # from BGR to RGB and from uint8 to float
    for i in range(3):
        torch_image[0, 2-i, :, :] = torch_image2[:, :, i] / 256
    
    
    '''if downscale != 1:
        for i in range(torch_image.size()[2]):
            torch_image[:, :, i, i] = torch.mean(torch_image3[:, :, downscale*i:downscale*i+down_add, downscale*i:downscale*i+down_add])
    else:
        torch_image = torch_image3'''
    
    # calculate result
    #input is a 1 x 3 x 320 x 320 image
    #torch_image = torch_image.to(torch.device("cuda"))
    #output = torch.from_numpy(ort_sess.run(None, {'input': image2.astype(np.float32)})[0])
    output = torch.from_numpy(ort_sess.run(None, {'input': torch_image.numpy()})[0])
    #output = net(torch_image)
    #output = output.cpu()
    
    #output is a 32 x 125 x 10 x 10 tensor
    #filter boxes based on confidence score (class_score*confidence)
    output = filter_boxes(output, 0.1)
    #filter boxes based on overlap
    output = nms(output, 0.25)
    
    
    # draw result on camera image
    for out1 in output:
        for out in out1:
            #convert relative to absolute width
            w = int(out[2] * 320)
            h = int(out[3] * 320)
            # convert middle point to upper left corner
            x = int(out[0] * 320 - int(w/2))
            y = int(out[1] * 320 - int(h/2))
            # draw
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, f"{int(out[4]*100)}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    
    # prints current frame with output
    #display_result(torch_image, output, torch.zeros([1,10,6]), file_path='yolo_prediction.png')

    cv2.putText(image, "fps="+fps, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (100, 255, 0), 2, cv2.LINE_AA)
    
    return image

def captureFrames():
    global video_frame, thread_lock

    # Video capturing from OpenCV
    #video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    video_capture = cv2.VideoCapture(0, cv2.CAP_ANY)

    while True and video_capture.isOpened():
        return_key, frame = video_capture.read()
        if not return_key:
            break

        frame = infer(frame)
        # Create a copy of the frame and store it in the global variable,
        # with thread safe access
        with thread_lock:
            video_frame = frame.copy()
        
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    video_capture.release()
        
def encodeFrame():
    global thread_lock
    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encoded_image) + b'\r\n')

@app.route("/")
def streamFrames():
    return Response(encodeFrame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", default=False, type=bool, choices=[True, False])
    # parser.add_argument()
    return parser
# check to see if this is the main thread of execution
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # Create a thread and attach the method that captures the image frames, to it
    process_thread = threading.Thread(target=captureFrames)
    process_thread.daemon = True

    # Start the thread
    process_thread.start()

    # start the Flask Web Application
    # While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
    # the host machine's localhost and is discoverable by other machines on the same network 
    app.run("0.0.0.0", port=8000)
    