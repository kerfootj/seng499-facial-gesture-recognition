import csv
from keras import models
from keras.preprocessing.image import img_to_array
import numpy
import os

classifier = models.load_model(os.path.normpath('generated_models/batch_128.epochs_100.hdf5'))
classifier_emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
# From https://www.kaggle.com/debanga/facial-expression-recognition-challenge/data?
test_emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

with open ('test_data/test.csv', mode = 'r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    total_classifications = 0
    correct_classifications = 0
    for row in csv_reader:   
        dimensions_array = numpy.fromstring(row['pixels'], sep = " ").reshape((1, 48, 48, 1))
        # Using the code gives the same result...about 35% accuracy on a larger dataset, which seems wrong
        # dimensions_array_2 = img_to_array(dimensions_array.astype("float") / 255.0)
        # dimensions_array_3 = numpy.expand_dims(dimensions_array_2, axis=0)
        prediction = classifier.predict(dimensions_array)[0]
        label = classifier_emotion_labels[prediction.argmax()]
        if((label == test_emotion_labels[int(row['emotion'])])): 
            correct_classifications+=1
        total_classifications+=1

    print("Accuracy is: %s" % ((correct_classifications/total_classifications)*100))
