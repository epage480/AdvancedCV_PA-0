import tensorflow as tf
import propertyLib
import Model1
import Model2
import Model3


class Model:

    def __init__(self, x, y, architecture, global_step=None, n_classes=200, l_r=1e-4, training=True, scope='MyModel_scope'):
        with tf.variable_scope(scope, reuse=False):
            self.x = x
            self.y = y
            self.arch = architecture
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
        if self.arch == "Model1":
            model = Model1.build(self.x, n_classes=self.n_classes)
            return model
        elif self.arch == "Model2":
            #raise NotImplementedError
            model = Model2.build(self.x, n_classes=self.n_classes)
            print("model shape:", model.shape)
            return model

        elif self.arch == "Model3":
            #raise NotImplementedError
            model = Model3.build(self.x, n_classes=self.n_classes)
            print("model shape:", model.shape)
            return model

        else:
            model = Model1.build(self.x, n_classes=self.n_classes)
            print("model shape:", model.shape)
            return model

    @propertyLib.lazy_property
    def loss(self):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.inference, labels=self.y))

    # Optimize based on cross entropy
    @propertyLib.lazy_property
    def optimize(self):
        return tf.train.AdamOptimizer(self.l_r).minimize(self.loss, global_step=self.global_step)

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
