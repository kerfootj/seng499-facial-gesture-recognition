import sys
import cv2
import face_recognition
from keras import models
from keras.preprocessing.image import img_to_array
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml') 
classifier = models.load_model('models/model_v6_23.hdf5')

emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def face_detector(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img

    for (x, y, w, h) in faces: 
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 0), 2) 
        roi_gray = gray[y:y + h, x:x + w] 

        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation = cv2.INTER_AREA)
        except:
            return (x,w,y,h), np.zeros((48,48), np.uint8), img
        return (x, w, y, h), roi_gray, img

def detect_from_video():

    cap = cv2.VideoCapture(0)

    while(True):
        _, frame = cap.read()
        rect, face, image = face_detector(frame)

        if np.sum([face]) != 0.0:
            roi = face.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = emotion_labels[preds.argmax()]
            label_position = (rect[0] + int((rect[1]/2)), rect[2] + 25)

            cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

        else:
            cv2.putText(image, "No Face Found", (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,2, (0,255,0), 3)

        # Display the resulting frame
        cv2.imshow('icu', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def detect_from_image(src):
    img = cv2.imread(src)
    rects, faces, image = face_detector(img)

    face_locations = face_recognition.face_locations(image)
    top, right, bottom, left = face_locations[0]
    face_image = image[top:bottom, left:right]

    # make a prediction on the ROI, then lookup the class
    preds = classifier.predict(face_image)[0]
    label = emotion_labels[preds.argmax()]

    #Overlay our detected emotion on our pic
    label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
    i =+ 1
    cv2.putText(image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        detect_from_video()
    else:
        detect_from_image(sys.argv[1])

