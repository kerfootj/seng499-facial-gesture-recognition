import numpy as np
import cv2
import dlib

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml') 
smile_cascade = cv2.CascadeClassifier('cascades/haarcascade_smile.xml') 

def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 15) 

        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame 

while(True):
    # Capture frame-by-frame
    _, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and smiles   
    canvas = detect(gray, frame)    

    # Display the resulting frame
    cv2.imshow('icu', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()