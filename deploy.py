import streamlit as st
import av
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
import cv2
import numpy as np
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from deployment_helper_funcs import predict_rt, predict_video, process_frame_drowsiness, reset_calibration
from run import train_model


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    calib = st.session_state.get('calibrating', False)
    processed_img, _ = process_frame_drowsiness(img, calibrate_mode=calib)
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")


with st.sidebar:
    st.image("./Assets/Logo.png")
    st.markdown("# Driver Drowsiness Detection", )
    st.subheader("Prevent accidents by alerting the driver")
    st.markdown('## Menu')
    choice = st.radio("Menu", ["Train","Upload","Real Time", "Logs"], label_visibility='collapsed')

if choice == 'Train':
    st.title("Train the model")
    st.markdown("### Model Architecture")

    col1, col2= st.columns([3,2])

    with col1:
        st.code('''
            model = Sequential([
                    Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)),
                    BatchNormalization(),
                    MaxPooling2D(),
                    Dropout(0.1),

                    Conv2D(32, 5, activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(),
                    Dropout(0.1),

                    Conv2D(64, 10, activation='relu'),
                    BatchNormalization(),
                    MaxPooling2D(),
                    Dropout(0.1),


                    Conv2D(128, 12, activation='relu'),
                    BatchNormalization(),

                    Flatten(),

                    Dense(512, activation='relu'),
                    Dropout(0.1),
                    Dense(1, activation='sigmoid')
                ])
        ''')
    with col2:
        st.image('./Assets/cnn_arch.png')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Dataset:** Dataset2 (`Active Subjects` & `Fatigue Subjects`)")
    with col2:
        epochs = st.number_input('Number of Epochs', min_value=1, value=5, step=1)
    
    col1, col2 = st.columns([1, 9])
    train_btn_state = False

    with col1:
        if st.button('Train', disabled=train_btn_state):
            with col2:
                with st.spinner('Training CNN model on Dataset2...'):
                    train_btn_state = True
                    train_dir = os.path.join('..', 'Dataset2') if os.path.exists(os.path.join('..', 'Dataset2')) else os.path.join('.', 'Dataset2')
                    train_model(train_dir, epochs=int(epochs), process_data=True)
                st.success('Training Complete!')

if choice == 'Upload':
    st.title('Upload Video For Prediction')
    video_file = st.file_uploader('Video', type=['mp4'])

    if video_file is not None:
        with open(predict_video(video_file=video_file), 'rb') as out_file:
            st.video(out_file.read(), use_container_width=True)

if choice == 'Real Time':
    st.title("Real Time Drowsiness Detection!")
    st.caption("Hybrid Dual-Stream AI: Facial Landmarks (EAR/MAR) + Deep Learning CNN Spatial Inference")

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        if st.button("Start 3-Sec Calibration (Neutral Eyes Open)"):
            st.session_state['calibrating'] = True
            st.info("Calibrating baseline eye aspect ratio... Keep eyes open normally.")
    with col_c2:
        if st.button("Reset Thresholds to Default"):
            reset_calibration()
            st.session_state['calibrating'] = False
            st.success("Thresholds reset to default (EAR 0.22).")

    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        ctx = webrtc_streamer(
            key="driver-drowsiness-detection",
            video_frame_callback=video_frame_callback,
            video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True),
        )


    

if choice == 'Logs':
    st.title('Logs')

    log_data = ''
    with st.spinner('Loading Logs'):
        with open(os.path.join('.', 'Logs', 'debug.log')) as log_file:
            lines = log_file.readlines()[-100:]
            lines.reverse()
            for line in lines:
                log_data += line
    st.markdown(
    f'''
    ```log
        {log_data}
    '''
    )