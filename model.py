## Load training data
import csv
import cv2
import numpy as np

lines = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)
## end loading training data


## Generator
import csv

samples = []
with open('data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from random import shuffle, random


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:                        
                name = 'data/data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                #center_image = cv2.resize(center_image, (200, 66), cv2.INTER_AREA)
                center_angle = float(batch_sample[3])
                
                if random() >= .5 and abs(float(batch_sample[3])) > 0.1:
                    center_image = cv2.flip(center_image, 1)
                    center_angle = -float(batch_sample[3])
                    
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
#end generator


# Nvidia Network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.optimizers import Adam

model = Sequential()
#normalize
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

model.add(Convolution2D(24,
                        5, 5,
                        border_mode='valid',
                        subsample=(2,2),
                        activation='relu'))
#model.add(Dropout(0.4))

model.add(Convolution2D(36,
                        5, 5,
                        border_mode='valid',
                        subsample=(2,2),
                        activation='relu'))
#model.add(Dropout(0.7))

model.add(Convolution2D(48,
                        5, 5,
                        border_mode='valid',
                        subsample=(2,2),
                        activation='relu'))
#model.add(Dropout(0.6))

model.add(Convolution2D(64,
                        3, 3,
                        border_mode='valid',
                        activation='relu'))
#model.add(Dropout(0.5))

model.add(Convolution2D(64,
                        3, 3,
                        border_mode='valid',
                        activation='relu'))
#model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(1164, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(1))


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

adam = Adam(lr=0.001)
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=5, verbose=1)

#save model
model.save('model.h5')
