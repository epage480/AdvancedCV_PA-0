# adapted from
# https://github.com/seshuad/IMagenet/blob/master/TinyImagenet.ipynb

import tensorflow as tf
import pandas as pd
import zipfile
import requests
import io
import os
import math

import cv2
import sklearn
import scipy.io


BATCH_SIZE = 20
IMAGE_SIZE = 64
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
TRAIN_IMAGES_FILE = 'train_32x32.mat'
TEST_IMAGES_FILE = 'test_32x32.mat'

TRAIN_IMAGES_URL = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
TEST_IMAGES_URL = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'


# Downloads and Extracts dataset if not already done so
def download_images():
    if os.path.exists(TRAIN_IMAGES_FILE):
        print('Training images already downloaded...')
    else:
        r = requests.get(TRAIN_IMAGES_URL, stream=True)
        print('Downloading ' + TRAIN_IMAGES_URL)
        with open('train_32x32.mat', 'wb') as handle:
            for block in r.iter_content(1024):
                handle.write(block)

    if os.path.exists(TEST_IMAGES_FILE):
        print('Test images already downloaded...')
        return
    else:
        r = requests.get(TEST_IMAGES_URL, stream=True)
        print('Downloading ' + TEST_IMAGES_URL)
        with open('test_32x32.mat', 'wb') as handle:
            for block in r.iter_content(1024):
                handle.write(block)


# Used to load in data in batches
class Dataloader:
    def __init__(self, batch_size=64, file='train_32x32.mat', shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        data = scipy.io.loadmat(file)
        self.images = data['X']
        self.labels = data['y']
        self.n_samples = len(data['y'])
        self.batch_num = 0
        self.total_batches = math.ceil(self.n_samples/batch_size)

        temp = []
        for i in range(0, self.n_samples):
            temp.append(self.images[:, :, :, i])
        self.images = temp
        temp = []
        for i in range(0, self.n_samples):
            temp.append(self.labels[i][0]-1)
        self.labels = temp

    def get_batch(self):
        # Shuffle every epoch
        if self.shuffle == True:
            if self.batch_num == 0:
                self.images, self.labels = sklearn.utils.shuffle(self.images, self.labels)

        index_lower = int(self.batch_num*self.batch_size)
        index_higher = int(min((self.batch_num+1) * self.batch_size, self.n_samples))
        batch = [self.images[index_lower:index_higher], self.labels[index_lower:index_higher]]

        if self.batch_num >= self.total_batches-1:
            self.batch_num = 0
            return batch, False
        else:
            self.batch_num += 1
            return batch, True




