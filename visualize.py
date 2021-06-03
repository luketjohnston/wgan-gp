import matplotlib
matplotlib.use('tkagg')
import tensorflow as tf
import os
from tensorflow.keras import Model
import gym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import keras_wgan as wgan


def saveImage(gan, filename, width=6, height=6, display=False):
  image_approxs = tf.squeeze(gan.gen.call(width * height,False))
  rows = [[image_approxs[i] for i in range(width*x,width*x+width)] for x in range(height)]
  
  for i,r in enumerate(rows):
    rows[i] = tf.concat(r, axis=0)
  im = tf.concat(rows, axis=1)
  if not display:
    plt.imsave(filename, im, cmap="gray")
  else:
    plt.imshow(im)
    plt.show()


if __name__ == '__main__':
  with tf.device('/device:CPU:0'):

    gan = tf.saved_model.load(wgan.model_savepath)
    saveImage(gan,'tempim.png',display=True)
  
  
  
  
