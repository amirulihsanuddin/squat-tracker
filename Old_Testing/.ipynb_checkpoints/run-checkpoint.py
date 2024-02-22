#pip install mediapipe opencv-python pandas scikit-learn

#import all the libraries needed
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt

#To set up the mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#To detect the lines in csv
landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    
#import up down model
with open('coords-cleaned.pkl', 'rb') as f:
    model1 = pickle.load(f)
    
#import good-goodmorning-cave in-cave out model
with open('good-gm-cvin-cvout.pkl', 'rb') as f:
    model2 = pickle.load(f)
    
    
#r"C:\Users\eddcr\Downloads\Wrong Squat\cave in backlight\squat (2).mp4"
#r"C:\Users\eddcr\Downloads\FYP Dataset\squat (39).mp4"
#r"C:\Users\eddcr\Downloads\Fit3D Dataset\squat\squat (2).mp4"
#video_path = r"C:\Users\eddcr\Downloads\FYP Dataset\squat (39).mp4"
counter = 0
current_stage = ''
cap = cv2.VideoCapture(0)
pause = False  # Flag to indicate whether the video is paused
    
# initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detection
        results = pose.process(image)

        # Recolor image to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_language_class1 = model1.predict(X)[0]
            body_language_prob1 = model1.predict_proba(X)[0]
            print(body_language_class1, body_language_prob1)
            
            body_language_class2 = model2.predict(X)[0]
            body_language_prob2 = model2.predict_proba(X)[0]
            print(body_language_class2, body_language_prob2)
            
            # counter
            if body_language_class1 == 'up' and body_language_prob1[body_language_prob1.argmax()] >= .7:
                current_stage = 'up'
            elif current_stage == 'up' and body_language_class1 == 'down' and body_language_prob1[body_language_prob1.argmax()] >= .7:
                current_stage="down"
                counter +=1
                print(current_stage)
            
            #Get status box
            cv2.rectangle (image, (0,0), (480, 60), (245, 117, 16), -1)
            
            
            #Display Up or Down
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class1.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)
            
            #display probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob1[np.argmax(body_language_prob1)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)
            
            #display counter
            cv2.putText(image, 'COUNT'
                        , (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter)
                        , (175,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)
            
            #Display Class
            cv2.putText(image, 'FORM'
                        , (260,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class2.split(' ')[0]
                        , (265,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)
            
            #display probability
            cv2.putText(image, 'FORM PROB'
                        , (385,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob2[np.argmax(body_language_prob2)],2))
                        , (390,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2, cv2.LINE_AA)
            
        except Exception as e:
            print("Error")
            pass

        k = cv2.waitKey(1)
        if k == 27:  # press Esc key to exit
            break

        if not pause:
            display_image = cv2.resize(image, (1280, 720))
            cv2.imshow('Mediapipe Feed', image)
        
        
        
    cap.release()
    cv2.destroyAllWindows()

    if cv2.waitKey(0) & 0xFF == ord('p'):
    # If 'p' is pressed after a video, pause before proceeding to the next video
        pause = True