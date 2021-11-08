import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

lines = []
data_folder = 'data'
with open(data_folder+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
# remove the first line holding the headers
lines = lines[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2) #potentially redundant to Keras ..


def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)

        for offset in range(0,num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            measurements = []
            # steering correction hyper-parameter
            correction = 0.2
            for line in batch_samples:
                for i in range(3):
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    current_path = data_folder+'/IMG/' + filename
                    image = cv2.imread(current_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    measurement = float(line[3])
                    # i = 0 - center
                    if(i==0):
                        measurements.append(measurement)
                    # i = 1 - left -> + correction
                    if(i==1):
                        measurements.append(measurement + correction)
                    # i = 1 - right -> - correction
                    if(i==2):
                        measurements.append(measurement - correction)

            augmented_images, augmented_measurements = [], []
            for image, measurement in zip (images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set batch size
batch_size = 32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

#LeNet Model
if(0):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    model.add(Conv2D(6,5,5, activation="relu"))
    model.add(MaxPooling2D())
    # Dropout new in Lenet
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

# produces model_lenet.h5

#Nvidia Model
if(1):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Conv2D(24, (5,5), subsample=(2,2),activation="relu"))
    model.add(Conv2D(36, (5,5), subsample=(2,2),activation="relu"))
    model.add(Conv2D(48, (5,5), subsample=(2,2),activation="relu"))
    model.add(Conv2D(64, (3,3),activation="relu"))
    model.add(Conv2D(64, (3,3),activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

# produces model_nvidia.h5 -> renamed to model.h5 to comply for report

model.compile(loss='mse', optimizer='adam')
#history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6, verbose = 1)
history_object = model.fit_generator(train_generator, steps_per_epoch = np.ceil(len(train_samples)/batch_size), validation_data = validation_generator, validation_steps = np.ceil(len(validation_samples)/batch_size), epochs = 5, verbose = 1)
model.save('model.h5')

print(history_object.history.keys())
print(history_object.history['loss'])
print(history_object.history['val_loss'])


fig = plt.figure(1)
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
fig.tight_layout()
fig.savefig('history_object_nvidia.jpeg')



