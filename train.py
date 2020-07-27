import keras
from keras.preprocessing.image import ImageDataGenerator

number_classes = 7
image_row, image_col = 48, 48
batch_size = 512

training_data = 'icml_face_data.csv'

training_generator = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      shear_range=0.3,
      zoom_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')



