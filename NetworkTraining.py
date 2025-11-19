# -*- coding: utf-8 -*-
"""
@authors: noam yaakobi, lior ichiely
"""

import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Models import models as model
import pickle

# Tensorflow configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Image configuration
seed = 42
np.random.seed = seed

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

# Images and masks paths
IMG_PATH_train = 'train/Image/'
MASK_PATH_train = 'train/Masks_bit/'
IMG_PATH_test = 'test/Image/'
MASK_PATH_test = 'test/Masks_bit/'
img_ids_train = os.listdir(IMG_PATH_train)
mask_ids_train = os.listdir(MASK_PATH_train)
img_ids_test = os.listdir(IMG_PATH_test)
mask_ids_test = os.listdir(MASK_PATH_test)

# Allocate arrays
X_test = np.zeros((len(img_ids_test), IMG_HEIGHT, IMG_WIDTH, 1))
Y_test = np.zeros((len(img_ids_test), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool_)
X_train = np.zeros((len(img_ids_train), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(img_ids_train), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.bool_)

print('Resizing training & testing images and masks...')
# Training images
for i in range(len(img_ids_train)):
    img = imread(IMG_PATH_train + img_ids_train[i])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True) / 255
    X_train[i] = img
    mask = imread(MASK_PATH_train + mask_ids_train[i])
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    Y_train[i] = mask

# Shuffle data
indices = list(range(len(X_train)))
np.random.shuffle(indices)
X_train = np.array([X_train[i] for i in indices])
Y_train = np.array([Y_train[i] for i in indices])

# Split to train and validation data
X_val = X_train[-10:]
Y_val = Y_train[-10:]
X_train = X_train[:-10]
Y_train = Y_train[:-10]

# Test images
for i in range(len(img_ids_test)):
    img = imread(IMG_PATH_test + img_ids_test[i])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True) / 255
    X_test[i] = img
    mask = imread(MASK_PATH_test + mask_ids_test[i])
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    Y_test[i] = mask

# Define the model and its name (Xnet - UNet++. AttXnet - Attention UNet++)
model_name = 'AttXnet_final'
model = model.AttXnet(use_backbone=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, 'Save', model_name)

# If a saved model exists, load it and load its training history
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    with open('trainHistoryDict', 'rb') as file:
        loaded_history = pickle.load(file)
    history = tf.keras.callbacks.History()
    model.summary()
    history.history = loaded_history
else:
    # Else, prepare, train and save the model
    model = tf.keras.Model(model.input, tf.keras.layers.Activation('linear', dtype='float32')(model.output))
    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy')
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=80, batch_size=4)
    model.save(model_path)
    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

# Plot the training and validation curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Calculate IOU and save segmentations
IOU_Sum = 0
for i in range(0, len(img_ids_test), 10):
    X_pred = model(X_test[i:i+10]).numpy()
    for j in range(10):
        try:
            # Show segmentations
            print(f"Showing image: {img_ids_test[i + j]}")
            plt.imshow(X_test[j + i])
            plt.show()
            plt.imshow(Y_test[j + i])
            plt.show()
            plt.imshow(X_pred[j])
            plt.show()

            # Calculate IoU
            pred = (X_pred[j] > 0.021).astype(np.bool_)
            overlap = pred * Y_test[j + i]
            union = pred + Y_test[j + i]
            IOU = overlap.sum() / float(union.sum())
            IOU_Sum += IOU

            # Save
            img = X_test[i + j]
            imsave(str(j + i) + '.png', img)
            pred = np.where(pred, 255, 0).astype(np.uint8)
            imsave(str(j+i) + '_pred.png', pred)
        except: continue

# Write average IoU
"""
IOU_Avg = IOU_Sum / len(img_ids_test)
file = open(model_name + 'IOU', "w")
file.write(str(IOU_Avg))
file.close()"""