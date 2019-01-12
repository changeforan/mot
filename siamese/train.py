
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


#import system things
import tensorflow as tf
import numpy as np
import os

#import helpers
import inference
from inputdata import Player


# prepare data and tf.session
players = Player()
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
load = False
model_ckpt = './model.meta'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model files. Do you want to load it and continue training [yes/no]?")
    if input_var == 'yes':
        load = True

# start training
if load: saver.restore(sess, './model')

for step in range(50000):
    batch_x1, batch_y1 = players.train.next_batch(500)
    batch_x2, batch_y2 = players.train.next_batch(500)
    batch_y = [float(i[0] == i[1]) for i in zip(batch_y1, batch_y2)]

    _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                        siamese.x1: batch_x1,
                        siamese.x2: batch_x2,
                        siamese.y_: batch_y})

    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()

    if step % 100 == 0:
        print ('step %d: loss %.3f' % (step, loss_v))

    if step % 10000 == 0 and step > 0:

        saver.save(sess, './model')




