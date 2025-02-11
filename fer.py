import cv2
import numpy as np
from keras.models import model_from_json #type: ignore
import streamlit as st
def main():
    st.set_page_config(layout='centered',page_title="FER",page_icon='facial-recognition.png')
    emo_dict={
        0:'Angry',
        1:'Disgust',
        2:'Fear',
        3:'Happy',
        4:'Neutral',
        5:'Sad',
        6:'Surprise'
    }
    json_file=open('model/fer_json0','r')
    model_loaded=json_file.read()
    json_file.close()
    fer_model=model_from_json(model_loaded)

    fer_model.load_weights('model/fer_model0.weights.h5')
    print("model_loaded")
    st.title('Facial Emotion Recognition(FER)')
    col1,col2=st.columns(2)
    with col1:
        run=st.button("Start")
    with col2:
        mode=st.toggle('Live',True)
    frame_window=st.image([],channels='RGB')
    if mode:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("second.mp4")
    while run:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1280, 720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = fer_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emo_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame0=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_window.image(frame0,channels='RGB')
        
    cap.release()
    cv2.destroyAllWindows()

main()