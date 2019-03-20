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


BATCH_SIZE = 20
IMAGE_SIZE = 64
NUM_CLASSES = 200
NUM_IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * NUM_IMAGES_PER_CLASS
TRAINING_IMAGES_DIR = './tiny-imagenet-200/train/'
TRAIN_SIZE = NUM_IMAGES

NUM_VAL_IMAGES = 10000
VAL_IMAGES_DIR = './tiny-imagenet-200/val/'

IMAGES_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'


# Downloads and Extracts dataset if not already done so
def download_images(url=IMAGES_URL, path=TRAINING_IMAGES_DIR):
    if os.path.isdir(path):
        print('Images already downloaded...')
        return
    r = requests.get(url, stream=True)
    print('Downloading ' + url)
    zip_ref = zipfile.ZipFile(io.BytesIO(r.content))
    zip_ref.extractall('./')
    zip_ref.close()


# Returns 2 lists: file paths and labels for the training data
# Labels are ints (0-199) assigned based on the order observed
def load_training_images(image_dir=TRAINING_IMAGES_DIR):
    print("loading training images")
    label_index = 0

    image_files = []
    labels = []

    # Loop through all the label directories
    for labelname in os.listdir(image_dir):
        labelpath = image_dir + labelname + '/images/'
        if os.path.isdir(labelpath):
            images = os.listdir(labelpath)

            # Loop through all the images of a type directory
            for image in images:
                image_files.append(os.path.join(labelpath, image))
                labels.append(label_index)

            label_index += 1
    print("about to return")
    return image_files, labels


def load_val_images(image_dir=VAL_IMAGES_DIR):
    # Generate dictionary to convert class names to labels (ex: 'n04259630': 7)
    c2l_dict, l2c_dict = classes_to_labels()

    # Read in annotations.txt to get class from file name
    val_df = pd.read_csv(VAL_IMAGES_DIR + 'val_annotations.txt', sep='\t', header=None,
                           names=['File', 'Class', 'X', 'Y', 'H', 'W'])

    # Get the image files names and paths from pandas data frame
    images_path = os.path.join(image_dir + 'images')
    image_files = [os.path.join(images_path, image) for image in list(val_df.File)]

    # Get classes the same way, converting to labels
    labels = [c2l_dict[x] for x in val_df.Class]

    return image_files, labels


def classes_to_labels(image_dir=TRAINING_IMAGES_DIR):
    c2l_dict = {}
    l2c_dict = {}
    count = 0
    # Loop through all the label directories
    for labelname in os.listdir(image_dir):
        labelpath = image_dir + labelname + '/images/'
        if os.path.isdir(labelpath):
            c2l_dict[labelname] = count
            l2c_dict[count] = labelname
            count += 1

    return c2l_dict, l2c_dict


# Used to load in data in batches
class Dataloader:
    def __init__(self, batch_size=64, data_percent=1):
        self.files, self.labels = load_training_images()
        self.files, self.labels = sklearn.utils.shuffle(self.files, self.labels)

        self.vfiles, self.vlabels = load_val_images()
        self.batch_size = batch_size
        self.batch_num = 0
        self.n_samples = data_percent*NUM_IMAGES
        self.batches = math.ceil(self.n_samples/batch_size)
        self.vbatches = math.ceil(NUM_VAL_IMAGES/batch_size)


    def get_train_batch(self):
        batch_num = self.batch_num
        batch_size = self.batch_size
        index_lower = int(batch_num*batch_size)
        index_higher = int(min((batch_num+1) * batch_size, self.n_samples))
        image = []
        for i in range(index_lower, index_higher):
            temp = cv2.imread(self.files[i])
            temp2 = cv2.normalize(temp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image.append(temp2)
        batch = [image, self.labels[index_lower: index_higher]]
        if self.batch_num >= self.batches-1:
            self.batch_num = 0
            return batch, False
        else:
            self.batch_num += 1
            return batch, True


    def get_val_batch(self):
        batch_num = self.batch_num
        batch_size = self.batch_size
        index_lower = batch_num*batch_size
        index_higher = min((batch_num+1) * batch_size, NUM_VAL_IMAGES)
        image = []
        for i in range(index_lower, index_higher):
            temp = cv2.imread(self.files[i])
            temp2 = cv2.normalize(temp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            image.append(temp2)
        batch = [image, self.labels[index_lower: index_higher]]
        if self.batch_num >= self.vbatches-1:
            self.batch_num = 0
            return batch, False
        else:
            self.batch_num += 1
            return batch, True
