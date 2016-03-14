from keras.models import Sequential
import keras.layers as layers
from keras.optimizers import SGD
from keras.preprocessing import image
import os
from PIL import Image
import glob
import numpy as np
from sklearn.preprocessing import normalize
import keras.regularizers as regularizers

train_data_name_list = ['1', '2', '3']
test_data_name_list = ['T1', 'T2']
X_train = []
y_train = []
X_test = []
y_test = []


def load_daimler_data(dir, X, y, flat_flag):
    # data pixel : 18x36 = 648
    x_pos = []
    x_neg = []
    for f in glob.glob(os.path.join(os.getcwd(), dir, "ped_examples", "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        if flat_flag == True:
            arr = arr.reshape((1, 648))
        x_pos.append(arr)
    for f in glob.glob(os.path.join(os.getcwd(), dir, "non-ped_examples", "*.pgm")):
        img = Image.open(str(f))
        arr = image.img_to_array(img)
        if flat_flag == True:
            arr = arr.reshape((1, 648))
        x_neg.append(arr)
    for i in x_pos:
        X.extend(i)
        y.append(1)
    for i in x_neg:
        X.extend(i)
        y.append(0)

print 'loading pedestrian dataset...'
for i in train_data_name_list:
    load_daimler_data(i, X_train, y_train, flat_flag=False)
for i in test_data_name_list:
    load_daimler_data(i, X_test, y_test, flat_flag=False)

print 'converting dataset to numpy format...'
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print 'normalizing dataset...'
for i in X_train:
    normalize(i, copy=False)
for i in X_test:
    normalize(i, copy=False)
print 'data preparation complete'

print 'building model...'
model = Sequential()

model.add(layers.Reshape(input_shape=(36, 18), dims=(1, 36, 18)))
model.add(layers.Convolution2D(nb_filter=24, nb_row=5, nb_col=5,
                               activation='relu', border_mode='valid'))
model.add(layers.Convolution2D(nb_filter=48, nb_row=3, nb_col=3,
                               activation='relu', border_mode='valid'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(2000, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

# "class_mode" defaults to "categorical". For correctly displaying accuracy
# in a binary classification problem, it should be set to "binary".
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')
print 'building complete'

print 'training model...'
model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=20,
          verbose=1,
          shuffle=True,
          show_accuracy=True,
          validation_split=0.1)
print 'training complete'

print 'evaluationg model...'
score = model.evaluate(X_test, y_test, batch_size=20, verbose=1, show_accuracy=True)
print score
