import streamlit as st
import cv2
import torch
import numpy as np
import time
import os
import onnxruntime as ort

from models.my_tinyyolov2 import MyTinyYoloV2
from models.pruned_my_tinyyolo2 import PrunedMyTinyYoloV2

from utils.camera import CameraDisplay
from utils.dataloader import VOCDataLoaderPerson
from utils.yolo import nms, filter_boxes
from utils.viz import display_result

# from ultralytics import YOLO

def infer(engine, confidence, placeholder, with_onnx=False, max_images=10):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    myloader = VOCDataLoaderPerson(train=False, batch_size=1, shuffle=True)
    for idx, (inputs, targets) in enumerate(myloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        t_start = time.time()
        #input is a 1 x 3 x 320 x 320 image
        if with_onnx:
            outputs = torch.from_numpy(engine.run(None, {'input': inputs.numpy()})[0])
        else:
            outputs = engine(inputs)
        "output is of a tensor of size 32 x 125 x 10 x 10"
        #output is a 32 x 125 x 10 x 10 tensor
        
        #filter boxes based on confidence score (class_score*confidence)
        outputs = filter_boxes(outputs, confidence)
        
        #filter boxes based on overlap
        outputs = nms(outputs, 0.25)
        t_end = time.time()
        inf_time = t_end - t_start

        fig = display_result(inputs, outputs, targets, file_path='yolo_prediction.png')
        
        placeholder.empty()
        st.pyplot(fig)
        st.write(f"Inference time: {inf_time}")

        if idx == max_images - 1:
            break
         
def init_empty_camera():
    '''Workaround to camera failing the first time around'''
    def __callback(image):
        return image
    cam = CameraDisplay(__callback, lazy_camera_init=True)
    cam.start()
    cam.stop()
    cam.release()

def main():

    st.sidebar.title('Settings')
    # Choose the model
        # Inference Mode
    data_src = st.sidebar.radio(
            'Data Source :', ('Camera', 'Images', 'Webcam'), index=0)

    st.title(f'TinyYolo Person Detection ✨')
    description = st.markdown(f"""
            This demonstrator can be used to test different Yolo models. You can use load your model and create your desired setup with the settings in the sidebar on the left. You can choose to infer you model on the Jetson Camera or a random set of images from the VOC dataset. The other settings depend on your choice of data source. Enjoy! 👷
            """)
    sample_img = cv2.imread('logo.png')
    left, center, right = st.columns([0.25, 0.55, 0.2], vertical_alignment="center")
    left.write(' ')
    right.write(' ')
    FRAME_WINDOW = center.image(sample_img, channels='BGR', caption="Inference example with our pruned TinyYolov2 model")

    # Confidence
    confidence = st.sidebar.slider('Detection Confidence', min_value=0.0, max_value=1.0, value=0.1)

    if data_src == "Images":
        model_type = st.sidebar.selectbox(
            'Choose YOLO Model', ('PrunedMyTinyYoloV2', 'MyTinyYoloV2')
        )

        path_model_file = st.sidebar.text_input(
            f'Path to {model_type} Model (.pt or .onnx):',
            f'models/configs/voc_pruned4.pt'
        )

        max_imgs = st.sidebar.slider('No. of images: ', min_value=1, max_value=50, value=10)

        if st.sidebar.checkbox('Load Model'):
            description.markdown("Inference will start below...")

            if not os.path.exists(path_model_file):
                st.sidebar.warning('File not found!', icon="⚠️")
            
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

            if ".pt" in path_model_file:
                yolo = {"PrunedMyTinyYoloV2": PrunedMyTinyYoloV2, "MyTinyYoloV2": MyTinyYoloV2}
                mynet = yolo[model_type](num_classes=1)
                # load pretrained weights
                sd = torch.load(path_model_file)
                mynet.load_state_dict(sd)
                mynet.to(device)
                mynet.eval()
                st.sidebar.success(
                    'PyTorch Model Loaded Successfully!',
                    icon="✅"
                )
                infer(engine=mynet, confidence=confidence, placeholder=FRAME_WINDOW, max_images=max_imgs)
            elif ".onnx" in path_model_file:
                ort_sess = ort.InferenceSession(path_model_file)
                st.sidebar.success(
                    'Onnx Session Setup Successful!',
                    icon="✅"
                )
                infer(engine=ort_sess, confidence=confidence, placeholder=FRAME_WINDOW, with_onnx=True, max_images=max_imgs)

            else:
                st.sidebar.warning('File type not supported!', icon="⚠️")


    # Camera
    if data_src == 'Camera' or data_src == 'Webcam':
        path_model_file = st.sidebar.text_input(
            f'Path to ONNX Model:',
            f'onnx/pruned.onnx'
        )

        if not os.path.exists(path_model_file):
            st.sidebar.warning('File not found!', icon="⚠️")

        if ".onnx" not in path_model_file:
            st.sidebar.warning('File type not supported!', icon="⚠️")

        if st.sidebar.checkbox('Load ONNX Model'):
            description.markdown("You loaded the ONNX model. You can now start the real-time inference with your chosen camera.")

            st.session_state.prev_time = 0

            ort_sess = ort.InferenceSession(path_model_file)
            st.sidebar.success(
                'Onnx Session Setup Successful!',
                icon="✅"
            )
            FRAME_WINDOW.image(np.zeros((320, 320, 3), dtype=np.uint8))
            #init_empty_camera()
            
            def custom_callback(image, prev_time):
                curr_time = time.time()
                fps = f"{int(1/(curr_time - prev_time))}"
                st.session_state.prev_time = curr_time

                image = image[0:320,0:320, :]
                cv2.putText(image, "fps="+fps, (2, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (100, 255, 0), 2, cv2.LINE_AA)
                
                # convert image to torch
                # from 320 x 320 x 3 to 1 x 3 x 320 x 320
                image =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                torch_image2 = torch.from_numpy(image)
                torch_image = torch.zeros([1, 3, 320, 320])
                #torch_image = torch.zeros([1, 3, int(320 / downscale), int(320 / downscale)])
                
                # from BGR to RGB and from uint8 to float
                for i in range(3):
                    torch_image[0, 2-i, :, :] = torch_image2[:, :, i] / 256
                
                # calculate result
                #input is a 1 x 3 x 320 x 320 image
                output = torch.from_numpy(ort_sess.run(None, {'input': torch_image.numpy()})[0])

                #output is a 32 x 125 x 10 x 10 tensor
                #filter boxes based on confidence score (class_score*confidence)
                output = filter_boxes(output, confidence)
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
                
                return image, curr_time
                                    
            if 'camera_on' not in st.session_state:
                st.session_state.camera_on = False

            if 'camera_init' not in st.session_state:
                st.session_state.camera_init = False

            def click_start_camera():
                st.session_state.camera_on = True
                st.session_state.camera_init = True
                st.session_state.prev_time = 0

            def click_stop_camera():
                st.session_state.camera_on = False

            _, middle_column, right_column = st.columns([0.1, 0.5, 0.3], vertical_alignment="center")
            middle_column.button('Start camera', on_click=click_start_camera)
            right_column.button('Stop camera', on_click=click_stop_camera)

            error = False
            if data_src == 'Webcam':
                try:
                    cam = CameraDisplay(custom_callback, streamlit=True, is_webcam=True)
                except RuntimeError:
                    st.sidebar.warning("Camera cannot be initialized. Choose another data source and reload the model!", icon="⚠️")
                    error = True
            else:
                try:
                    cam = CameraDisplay(custom_callback, streamlit=True, is_webcam=False)
                except RuntimeError:
                    st.sidebar.warning("Camera cannot be initialized. Choose another data source and reload the model!", icon="⚠️")
                    error = True

            if st.session_state.camera_on and st.session_state.camera_init:
                cam.start()
                while cam is not None:
                    FRAME_WINDOW.image(cam.image_widget, caption="Inference is running on camera capture!")
            elif not st.session_state.camera_on and st.session_state.camera_init:
                cam.stop()
                cam.release()
                FRAME_WINDOW.image(np.zeros((320, 320, 3), dtype=np.uint8))
                st.warning("Stopped camera. Click the start button again for inference!", icon="⚠️")
            else:
                if error:
                    st.warning("Camera is unavailable. Please, check settings!", icon="⚠️")
                else:
                    st.success("Camera is ready to start.", icon="✅")

if __name__ == "__main__":
    main()