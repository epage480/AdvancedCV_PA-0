import tensorflow as tf
import ModelFramework2 as fw
import sys
import scipy.io
import cv2
import numpy as np
import SVHN_Dataset as ds
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1024, type=int,
                    help="Batch size, number of samples used per training iteration")
parser.add_argument('--epochs', default=20, type=int,
                    help="Epochs, number of full passes through data set")
parser.add_argument('--save_path', default="Model_Saves/Model1/Save", type=str,
                    help="Model save path")
parser.add_argument('--logs_path', default="tensorflow_logs/", type=str,
                    help="Tensorboard log save path")
args = parser.parse_args()


start_time = time.time()

# Print records of parse
print("batch size:", args.batch_size)
print("epochs:", args.epochs)

# Download dataset if needed and create dataloaders
ds.download_images()
train_data = ds.Dataloader(batch_size=64, file='train_32x32.mat')
test_data = ds.Dataloader(batch_size=64, file='test_32x32.mat', shuffle=False)

# Placeholders
image = tf.placeholder(tf.float32, [None, 32, 32, 3])
label = tf.placeholder(tf.int64, [None])

# Initialize model
myModel = fw.Model(image, label)

# Tensorboard summaries
tf.summary.scalar("Loss", myModel.loss)
tf.summary.scalar("Training Accuracy", myModel.accuracy)
merged_summary_op = tf.summary.merge_all()
sys.stdout.flush()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(args.logs_path + '/', graph=tf.get_default_graph())

    # Output total # of parameters
    print("Number of parameters:", sess.run(myModel.num_params()))
    t1 = time.time()
    i = 0
    print("Epoch   Accuracy   Time")
    for epoch in range(0, args.epochs):
        t1 = time.time()
        cont = True
        j=0
        while cont:
            j+=1
            # Get the batch and train
            mybatch, cont = train_data.get_batch()
            _, summary = sess.run([myModel.optimize, merged_summary_op], feed_dict={image: mybatch[0], label: mybatch[1]})

            # Measurements
            summary_writer.add_summary(summary, i)
            i += 1

            sys.stdout.flush()

        t2 = time.time()

        # Calculate Validation Accuracy by summing up all correct precitions and dividing by # of validation samples
        tally = 0
        cont = True
        while cont:
            j += 1
            mybatch, cont = test_data.get_batch()
            temp = sess.run(myModel.num_equal, feed_dict={image: mybatch[0], label: mybatch[1]})
            tally += sum(temp)
        print(epoch, tally/26032, t2-t1)
        sys.stdout.flush()

        # Save the model after each epoch
        myModel.save(sess, args.save_path, save_num=epoch)

print("Total Elapsed Time:", time.time()-start_time)
