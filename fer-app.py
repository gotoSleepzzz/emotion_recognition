import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import queue
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from streamlit_webrtc.config import RTCConfiguration
from streamlit_webrtc.webrtc import WebRtcMode


face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

classifier = load_model('./model.h5')

emotion_labels = ["Angry", "Disgust", "Fear",
                  "Happy", "Sad", "Surprise", "Neutral"]

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

FRAME_WINDOW = st.image([])


def detect_faces(input_img):
    input_img = np.asarray(input_img)
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,)
    if len(faces) < 0:
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), input_img

    for (x, y, w, h) in faces:
        cv2.rectangle(input_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    except:
        return (0, 0, 0, 0), np.zeros((48, 48), np.uint8), input_img
    return (x, w, y, h), roi_gray, input_img


def make_label(rect, face, img):
    if np.sum([face]) != 0.0:
        roi = face.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # make a prediction on the ROI, then lookup the class
        preds = classifier.predict(roi)[0]
        label = emotion_labels[preds.argmax()]
        label_position = (rect[0]-15, rect[2] + 25)
        cv2.putText(img, label, label_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        cv2.putText(img, "No Face Found", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return img
    pass

def web_cam_regco():
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")

            rect, face, img = detect_faces(image)
            img_w_label = make_label(rect, face, img)
            #img_w_label = cv2.cvtColor(img_w_label, cv2.COLOR_RGB2BGR)
            return img_w_label


    webrtc_ctx = webrtc_streamer(
        key="emotion-recognite",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def main():
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Emotion Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        rect, face, img = detect_faces(our_image)

        img = make_label(rect, face, img)

        st.image(img)


    webcam = st.checkbox('Using webcam')
    if webcam:
        web_cam_regco()

    




if __name__ == '__main__':
    main()
