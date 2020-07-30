import csv
from keras import models
from keras.preprocessing.image import img_to_array
import numpy
import os
from sklearn.metrics import classification_report

classifier = models.load_model(os.path.normpath('generated_models/batch_256.epochs_100.hdf5'))
# From https://www.kaggle.com/debanga/facial-expression-recognition-challenge/data?
test_emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
emotion_mapping = [0, 1, 2, 3, 5, 6, 4]

with open ('test_data/test.csv', mode = 'r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    total_classifications = 0
    correct_classifications = 0
    y_true = []
    y_pred = []

    for row in csv_reader:   
        dimensions_array = numpy.fromstring(row['pixels'], sep = " ").reshape((1, 48, 48, 1))
        prediction = classifier.predict(dimensions_array)[0]
        predicted_label = test_emotion_labels[prediction.argmax()]
        true_label = test_emotion_labels[int(row['emotion'])]
        y_pred.append(predicted_label)
        y_true.append(true_label)
        if((predicted_label == true_label)): 
            correct_classifications+=1
        total_classifications+=1

    print("Accuracy is: %s" % ((correct_classifications/total_classifications)*100))
    print(classification_report(y_true = y_true, y_pred = y_pred, labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]))


