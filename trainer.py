from keras.models import Sequential
from keras.preprocessing import image
import keras.layers as layers
from keras import callbacks
import numpy as np
from sklearn.cross_validation import train_test_split
from lib import data_handler
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--positive", help="input positive training dataset path")
parser.add_argument("-n", "--negative", help="input negative training dataset path")
parser.add_argument("-s", "--patchsize", help="pixel size of training dataset as tuple (height, width)")
parser.add_argument("-P", "--positivedatasize", help="number of positive training data")
parser.add_argument("-N", "--negativedatasize", help="number of negative training data")
args = parser.parse_args()

if args.positive:
    train_dir_pos = args.positive
if args.negative:
    train_dir_neg = args.negative
if args.patchsize:
    patchsize = args.patchsize
if args.positivedatasize:
    datasize_pos = args.positivedatasize
if args.negativedatasize:
    datasize_neg = args.negativedatasizea

X_train = []
y_train = []

# temporary variable declaration
train_dir_pos = "/mnt/hgfs/Shared/DaimlerBenchmark/Data/TrainingData/Pedestrians/48x96"
train_dir_neg = "/mnt/hgfs/Shared/DaimlerBenchmark/Data/TrainingData/NonPedestrians"
patchsize = (96, 48)
datasize_pos = 15000
datasize_neg = 15000
test_split = 0.1
###########################################################################################
print 'loading positive pedestrian dataset...'
data_handler.load_data_general(train_dir_pos, X_train, y_train,
                               format='pgm', label=(1,0), datasize=datasize_pos)

print 'loading negative dataset...'
data_handler.load_data_random_patches(train_dir_neg, X_train, y_train,
                                      format='pgm', label=(0,1), patchsize=patchsize, datasize=datasize_neg)

print 'converting dataset to numpy format...'
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

# split training data for test usage
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_split)

print 'data preparation complete'

print 'training data shape : ' + str(X_train.shape)


print 'building model...'
model = Sequential()
model.add(layers.BatchNormalization(axis=1, input_shape=X_train[0].shape))
model.add(layers.Convolution2D(nb_filter=24, nb_row=13, nb_col=13,
                               activation='relu', border_mode='valid'))
model.add(layers.MaxPooling2D())
model.add(layers.Convolution2D(nb_filter=48, nb_row=9, nb_col=9,
                               activation='relu', border_mode='valid'))
model.add(layers.MaxPooling2D())
model.add(layers.Convolution2D(nb_filter=96, nb_row=5, nb_col=5,
                               activation='relu', border_mode='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print 'building complete'

print 'augmenting training files...'
datagen = image.ImageDataGenerator(featurewise_center=True,
                                   samplewise_center=False,
                                   featurewise_std_normalization=True,
                                   samplewise_std_normalization=False,
                                   zca_whitening=False,
                                   rotation_range=0.05,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.,
                                   zoom_range=0.05,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   dim_ordering='th')
datagen.fit(X_train)

print 'training model...'
model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    samples_per_epoch=len(X_train),
                    nb_epoch=100,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks=[callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='auto')])
print 'training complete'

print 'evaluating model...'
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print 'test accuracy : ' + str(score[1])

print 'saving architecture and weights...'
json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights('weights.h5')
