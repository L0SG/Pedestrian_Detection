from keras.models import Sequential
import keras.layers as layers
from keras import callbacks
import numpy as np
from sklearn.preprocessing import normalize
from lib import data_handler

train_data_name_list = ['1', '2', '3']
test_data_name_list = ['T1', 'T2']
X_train = []
y_train = []
X_test = []
y_test = []

print 'loading pedestrian dataset...'
for i in train_data_name_list:
    data_handler.load_daimler_data(i, X_train, y_train)
for i in test_data_name_list:
    data_handler.load_daimler_data(i, X_test, y_test)

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
model.add(layers.Convolution2D(nb_filter=12, nb_row=5, nb_col=5,
                               activation='relu', border_mode='valid'))
model.add(layers.MaxPooling2D())
model.add(layers.Convolution2D(nb_filter=24, nb_row=3, nb_col=3,
                               activation='relu', border_mode='valid'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(500, activation='relu'))
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
          nb_epoch=100,
          batch_size=32,
          verbose=1,
          shuffle=True,
          show_accuracy=True,
          validation_split=0.1,
          callbacks=[callbacks.EarlyStopping(patience=5, verbose=True)])
print 'training complete'

print 'evaluating model...'
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1, show_accuracy=True)
print 'test accuracy : ' + str(score[1])

print 'saving architecture and weights...'
json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights('weights.h5')
