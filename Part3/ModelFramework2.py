import tensorflow as tf
import propertyLib
import AlexNetModel as alexNet


class Model:

    def __init__(self, x, y, global_step=None, n_classes=10, l_r=1e-6, training=True, scope='MyModel_scope'):
        with tf.variable_scope(scope, reuse=False):
            # x needs to be swaped from rgb to brg and rescaled to 227x227
            self.x = tf.reverse(x, axis=[-1])
            self.x = tf.image.resize_images(self.x, (227, 227))

            self.myvars = []
            self.y = y
            self.l_r = l_r
            self.global_step = global_step
            self.n_classes = n_classes
            self.training = training
            self.scope = scope
            self.inference
            self.loss
            self.optimize
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope))

    # Outputs unmodified logits i.e. without softmax
    @propertyLib.lazy_property
    def inference(self):
        model = alexNet.build(self.x)
        #self.myvars = tf.trainable_variables(scope='MyModel_scope/retrain')
        print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        self.myvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MyModel_scope/retrain')
        print("len myvars:", len(self.myvars))
        return model

    @propertyLib.lazy_property
    def loss(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.inference, labels=self.y))

    # Optimize based on cross entropy
    @propertyLib.lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self.l_r).minimize(self.loss, global_step=self.global_step, var_list=self.myvars)

    @propertyLib.lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.y, 1), tf.argmax(self.inference, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    @propertyLib.lazy_property
    def prediction(self):
        return tf.argmax(self.inference, 1)

    @propertyLib.lazy_property
    def num_equal(self):
        return tf.equal(self.prediction, self.y)

    @propertyLib.lazy_property
    def accuracy(self):
        return tf.reduce_mean(tf.cast(self.num_equal, tf.float32))

    def save(self, sess, model_path, save_num=''):
        if type(save_num) != str:
            save_num=str(save_num)
        model_path = model_path + save_num + ".ckpt"
        return self.saver.save(sess, model_path)

    def restore(self, sess, model_path):
        return self.saver.restore(sess, model_path)

    def num_params(self):
        return tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
