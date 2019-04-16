import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import *
from keras.models import load_model
from keras.optimizers import *
from imutils import paths
import numpy as np
import random
import cv2
import os
 
data = []
labels = []

#keras.backend.set_image_data_format('channels_first')

print("...Loading Images...")
imagePaths = sorted(list(paths.list_images('Dataset')))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (170,170))
	image = img_to_array(image) 
	data.append(image)
 
	#classs label extraction
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

data = np.array(data, dtype="float") / 255.0

labels = np.array(labels)

#binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
vertical_flip=True, fill_mode="nearest")

#Initialization of the Model
print("...Compiling the Model...")
model = Sequential()

model.add(Conv2D(filters = 32, input_shape = [170, 170, 3], kernel_size = 7, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 3, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 2, padding = 'same'))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 512, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu'))
model.add(Conv2D(filters = 512, kernel_size = 2, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 2, padding = 'same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(24, activation = 'softmax'))

optimizer = Adam(lr = 0.0001)	

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#model.fit(x = trainX, y = trainY, epochs = 25, batch_size = 32)

print("Training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 32,
    epochs=25, verbose=1)


model.summary()

test_loss, test_acc = model.evaluate(testX, testY, verbose = 1)

print(test_acc)

#Save the model
model.save_weights('/home/deep-bro/Downloads/Projects/Sign language detection/Weights_Final.h5')

print("Saved model to disk")

print("Done and Dusted!!")


