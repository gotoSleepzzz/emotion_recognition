import streamlit as stl
import cv2
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

classifier = load_model('./model.h5')

emotion_labels = ["Angry", "Disgust", "Fear",
                  "Happy", "Sad", "Surprise", "Neutral"]

FRAME_WINDOW = stl.image([])


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


def main():
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Emotion Recognition WebApp</h2>
    </div>
    </body>
    """
    stl.markdown(html_temp, unsafe_allow_html=True)
    webcam = stl.checkbox("Using webcam")
    image_file = stl.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        stl.text("Original Image")
        stl.image(our_image)

    if stl.button("Recognise"):
        rect, face, img = detect_faces(our_image)

        img = make_label(rect, face, img)

        stl.image(img)

    if webcam:
        cam = cv2.VideoCapture(0)
        while webcam:
            ret, frame = cam.read()

            rect, face, img = detect_faces(frame)
            img_w_label = make_label(rect, face, img)
            img_w_label = cv2.cvtColor(img_w_label, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(img_w_label)

        cam.release()
    else:
        stl.write('Webcam stopped')


if __name__ == '__main__':
    main()
