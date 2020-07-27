import csv
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import regularizers
from keras.regularizers import l1
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
import numpy
import os

number_classes = 7
image_row, image_col = 48, 48
batch_size = 512
number_train_samples = 199

x_train = []
y_train = []

with open ('test_data/test_data.csv', mode = 'r', encoding='utf-8-sig') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:   
        dimensions_array = numpy.fromstring(row['pixels'], sep = " ").reshape((48, 48, 1))
        x_train.append(dimensions_array)
        y = numpy.zeros(7)
        y[int(row['emotion'])] = 1 
        y_train.append(y)

# print(numpy.asarray(x_train))

# Create a sequential model
model = Sequential()
# Add layers. Apply penalty (regualizer) to weights
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001),input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(7, kernel_size=(1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Conv2D(7, kernel_size=(4, 4), activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Flatten())
model.add(Activation("softmax"))
# Summarize
model.summary()
x_train = numpy.asarray(x_train)
y_train = numpy.asarray(y_train)

# Compile model
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])


filepath = os.path.join('./generated_models/{epoch}.hdf5')

checkpoint = keras.callbacks.ModelCheckpoint(filepath,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max',
                                             save_freq='epoch'
                                             )
callbacks = [checkpoint]
epochs = 150
model_info = model.fit(
    x = x_train,
    y = y_train,
    validation_data=(x_train, y_train),
    steps_per_epoch=number_train_samples,
    epochs=2,
    callbacks=callbacks
)

print(model_info)
