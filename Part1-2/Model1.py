import tensorflow as tf


def build(img_in, n_classes=200, keep_prob=0.5):
    net = tf.layers.conv2d(img_in, filters=16, kernel_size=5, kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.layers.conv2d(net, filters=32, kernel_size=5, kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.layers.conv2d(net, filters=32, kernel_size=5, kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.layers.conv2d(net, filters=32, kernel_size=5, kernel_initializer=tf.contrib.layers.xavier_initializer())
    net = tf.contrib.layers.flatten(net)
    net = tf.contrib.layers.fully_connected(net, 256)
    net = tf.contrib.layers.fully_connected(net, 256)
    net = tf.contrib.layers.fully_connected(net, n_classes)
    return net
