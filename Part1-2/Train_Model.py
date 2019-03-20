import tensorflow as tf
import ModelFramework as fw
import sys
import time
import TinyImageNet_Dataset as ds
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1024, type=int,
                    help="Batch size, number of samples used per training iteration")
parser.add_argument('--epochs', default=20, type=int,
                    help="Epochs, number of full passes through data set")
parser.add_argument('--save_path', default="Model_Saves/Model1/Save", type=str,
                    help="Model save path")
parser.add_argument('--model', default="Model1", type=str,
                    help="Select the model architecture: Model1, Model2, or Model3 (smallest to largest)")
parser.add_argument('--logs_path', default="tensorflow_logs/", type=str,
                    help="Tensorboard log save path")
parser.add_argument('--data_percent', default=1, type=float,
                    help="Percent of the data used to train the network")
args = parser.parse_args()

# Print records of parse
print("model:", args.model)
print("batch size:", args.batch_size)
print("epochs:", args.epochs)
print("data percent:", args.data_percent)
# Record start time of program
start_time = time.time()

# Import Dataset
ds.download_images()
data = ds.Dataloader(batch_size=args.batch_size, data_percent=args.data_percent)

# Placeholders
image = tf.placeholder(tf.float32, [None, 64, 64, 3])
label = tf.placeholder(tf.int64, [None])

# Initialize model
myModel = fw.Model(image, label, architecture=args.model)

# Tensorboard summaries
tf.summary.scalar("Loss", myModel.loss)
tf.summary.scalar("Training Accuracy", myModel.accuracy)
merged_summary_op = tf.summary.merge_all()
sys.stdout.flush()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(args.logs_path + args.model + '/', graph=tf.get_default_graph())

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
            mybatch, cont = data.get_train_batch()
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
            mybatch, cont = data.get_val_batch()
            temp = sess.run(myModel.num_equal, feed_dict={image: mybatch[0], label: mybatch[1]})
            tally += sum(temp)
        print(epoch, tally/10000, t2-t1)
        sys.stdout.flush()

        # Save the model after each epoch
        myModel.save(sess, args.save_path, save_num=epoch)

print("Total Elapsed Time:", time.time()-start_time)
