import os
import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, SpatialDropout2D)
from keras.models import Sequential
from keras.optimizers import SGD, Adadelta, Adam
from keras.regularizers import l2
from sklearn.metrics import classification_report
from data_util import get_train_data


os.environ["CUDA_VISIBLE_DEVICES"] = ""
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

train_x, val_x, train_y, val_y = get_train_data(test_size=0.1, one_hot=True)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=l2(0.02)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.02)))

model.add(Conv2D(96, (3, 3), activation='relu', kernel_regularizer=l2(0.04)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.04)))
model.add(MaxPool2D((2, 2), (2, 2)))
model.add(SpatialDropout2D(0.25))
model.add(Conv2D(48, (2, 2), activation='relu', kernel_regularizer=l2(0.02)))
model.add(SpatialDropout2D(0.125))

model.add(Flatten())
model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.125))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(decay=1e-3),
              metrics=[metrics.categorical_accuracy])
print(model.summary())

model.fit(train_x, train_y, epochs=100, batch_size=128, validation_data=(val_x, val_y), verbose=2)
y_predict = model.predict_classes(val_x)
y_true = np.argmax(val_y, axis=1)
report = classification_report(y_true, y_predict)
print(report)
