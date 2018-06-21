#import system things
import tensorflow as tf
import numpy as np
import os

#import helpers
import inference
from inputdata import Player


if __name__ == "__main__":

    players = Player()
    sess = tf.InteractiveSession()
    siamese = inference.siamese()
    saver = tf.train.Saver()
    model_ckpt = './model.meta'
    if os.path.isfile(model_ckpt):
        saver.restore(sess, './model')

    embed = siamese.o1.eval({siamese.x1: players.test.images})
    embed.tofile('embed.txt')
