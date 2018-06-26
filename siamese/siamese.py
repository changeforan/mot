#import system things
import tensorflow as tf

#import helpers
import inference

MODEL_PATH = '/home/cs/work/mot/save_models/siamese/model'

class Siamese:
    def __init__(self):
        self.sess = tf.Session()
        self.siamese = inference.siamese()
        with self.sess.as_default():
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(self.sess, MODEL_PATH)

    def __del__(self):
        self.sess.close()

    def run(self, *img):
        with self.sess.as_default():
            o1 = self.siamese.o1.eval({
                self.siamese.x1: [i.reshape(-1) / 255. for i in img]
            })
            return o1