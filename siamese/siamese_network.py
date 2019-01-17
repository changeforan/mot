#import system things
import tensorflow as tf

#import helpers
from . import inference

MODEL_PATH = '/Users/cs/work/football_match_mot/mot/save_models/siamese/model'

class Siamese:
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.model = inference.Model()
        saver = tf.train.Saver()
        saver.restore(self.sess, MODEL_PATH)

    def __del__(self):
        self.sess.close()

    def run(self, *img):
        with self.sess.as_default():
            o1 = self.model.o1.eval({self.model.x1: [i.reshape(-1) / 255 for i in img]})
            return o1