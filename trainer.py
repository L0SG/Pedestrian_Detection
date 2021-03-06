from keras.models import Sequential
from keras.preprocessing import image
import keras.layers as layers
from keras import callbacks
import keras.optimizers as optimizers
import numpy as np
from sklearn.cross_validation import train_test_split
from lib import data_handler
import argparse
from keras.regularizers import l1, l2, l1l2

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
    datasize_neg = args.negativedatasize

X_train = []
y_train = []
# dummy file_name list
file_name = []

# temporary variable declaration
patchsize = (64, 32)
test_split = 0.1
zca_flag = False
mode = 'L'
###########################################################################################
print 'loading positive pedestrian dataset...'
train_daimler_pos = "/home/tkdrlf9202/DaimlerBenchmark/Data/TrainingData/Pedestrians/48x96"
data_handler.load_data_train(train_daimler_pos, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                               format='pgm', label=(1, 0), datasize=15660)

train_caltech_pos = "/home/tkdrlf9202/CaltechPedestrians/data-USA/images_cropped"
data_handler.load_data_train(train_caltech_pos, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(1, 0), datasize=49055)

train_eth_pos = "/home/tkdrlf9202/CaltechPedestrians/data-ETH/images_cropped"
data_handler.load_data_train(train_eth_pos, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(1, 0), datasize=11941)

train_inria_pos = "/home/tkdrlf9202/CaltechPedestrians/data-INRIA/images_cropped"
data_handler.load_data_train(train_inria_pos, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(1, 0), datasize=1236)

train_tudbrussels_pos = "/home/tkdrlf9202/CaltechPedestrians/data-TudBrussels/images_cropped"
data_handler.load_data_train(train_tudbrussels_pos, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(1, 0), datasize=1207)

total_positive = len(X_train)
print 'total positive data : ' + str(total_positive)

print 'loading negative dataset...'
# caution : file format changed from ppm to png after fp5
"""
train_daimler_neg = "/home/tkdrlf9202/DaimlerBenchmark/Data/false_positive_set1"
data_handler.load_data_train(train_daimler_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='ppm', label=(0, 1), datasize=34539)
"""
train_daimler_neg = "/home/tkdrlf9202/DaimlerBenchmark/Data/false_positive_set2"
data_handler.load_data_train(train_daimler_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='ppm', label=(0, 1), datasize=17938)
train_daimler_neg = "/home/tkdrlf9202/DaimlerBenchmark/Data/false_positive_set3"
data_handler.load_data_train(train_daimler_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='ppm', label=(0, 1), datasize=11723)
train_daimler_neg = "/home/tkdrlf9202/DaimlerBenchmark/Data/false_positive_set4"
data_handler.load_data_train(train_daimler_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='ppm', label=(0, 1), datasize=13667)
train_daimler_neg = "/home/tkdrlf9202/DaimlerBenchmark/Data/false_positive_set5"
data_handler.load_data_train(train_daimler_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(0, 1), datasize=36646)
train_daimler_neg = "/home/tkdrlf9202/DaimlerBenchmark/Data/false_positive_set6"
data_handler.load_data_train(train_daimler_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(0, 1), datasize=3769)
# bad fp set?
"""
train_caltech_neg = "/home/tkdrlf9202/CaltechPedestrians/data-USA/fp3"
data_handler.load_data_train(train_caltech_neg, X_train, y_train, file_name, patchsize=patchsize, mode=mode,
                             format='png', label=(0, 1), datasize=183633)
"""

total_negative = len(X_train) - total_positive
print 'total negative data : ' + str(total_negative)

print 'converting dataset to numpy format...'
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
print 'Total dataset length : ' + str(len(X_train))
# split training data for test usage
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=test_split)
# split training data for validation usage
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_split)
print 'data preparation complete'

print 'training data shape : ' + str(X_train.shape)


print 'building model...'
model = Sequential()
model.add(layers.BatchNormalization(axis=1, input_shape=X_train[0].shape))
# model.add(layers.noise.GaussianNoise(1))
model.add(layers.Convolution2D(nb_filter=64, nb_row=5, nb_col=3, border_mode='valid'))
model.add(layers.core.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.BatchNormalization(axis=1))
#model.add(layers.Dropout(0.25))

model.add(layers.Convolution2D(nb_filter=128, nb_row=3, nb_col=3, border_mode='valid'))
model.add(layers.core.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.BatchNormalization(axis=1))
#model.add(layers.Dropout(0.25))


model.add(layers.Convolution2D(nb_filter=128, nb_row=3, nb_col=3, border_mode='valid'))
model.add(layers.core.Activation('relu'))
model.add(layers.Convolution2D(nb_filter=256, nb_row=3, nb_col=3, border_mode='valid'))
model.add(layers.core.Activation('relu'))
model.add(layers.MaxPooling2D())
model.add(layers.BatchNormalization(axis=1))
#model.add(layers.Dropout(0.25))


model.add(layers.Flatten())
model.add(layers.MaxoutDense(1000, nb_feature=2, init='he_normal',
                             W_regularizer=l2(l=0.001)))
#model.add(layers.core.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.MaxoutDense(500, nb_feature=2, init='he_normal',
                             W_regularizer=l2(l=0.001)))
#model.add(layers.core.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.MaxoutDense(100, nb_feature=2, init='he_normal',
                             W_regularizer=l2(l=0.001)))
#model.add(layers.core.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              # default RMSprop lr is 0.001, set faster rate with batch.norm
              optimizer=optimizers.RMSprop(lr=0.01),
              metrics=['accuracy'])
print 'building complete'
model.summary()

print 'augmenting training files...'
datagen = image.ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   featurewise_std_normalization=False,
                                   samplewise_std_normalization=False,
                                   zca_whitening=zca_flag,
                                   rotation_range=1,
                                   width_shift_range=0.05,
                                   height_shift_range=0.05,
                                   shear_range=0.,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=False,
                                   dim_ordering='th')
datagen.fit(X_train)
if zca_flag:
    print 'zca whitening enabled, saving zca matrix...'
    datagen.principal_components.dump('zca_matrix')
print 'training model...'
model.fit_generator(datagen.flow(X_train, y_train, batch_size=128, shuffle=True),
                    samples_per_epoch=len(X_train),
                    nb_val_samples=len(X_test),
                    class_weight={0: 1, 1: 1},
                    nb_epoch=200,
                    verbose=1,
                    validation_data=datagen.flow(X_val, y_val, shuffle=True),
                    callbacks=[callbacks.EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')])
print 'training complete'

print 'evaluating model...'
score = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print 'test accuracy : ' + str(score[1])

print 'saving architecture and weights...'
json_string = model.to_json()
open('model.json', 'w').write(json_string)
model.save_weights('weights.h5')
