import tensorflow as tf
import code
import multiprocessing as mp
from datetime import datetime
import matplotlib
import timeit
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import random
import pickle 
import os
from skimage.transform import resize
from tensorflow.keras import Model
import gym
from contextlib import ExitStack
import tensorflow_datasets as tfds
import argparse

from visualize import saveImage

QUICK_TEST = False

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import wgan

RUN_EAGER = False
PROFILE = True

if RUN_EAGER:
  tf.config.run_functions_eagerly(True)

# TODO save and load training params in savepath
# TODO make params command line args rather than global variables

EPOCHS = 20
NUM_CRITIC_PER_GEN = 5 # number of training steps for critic per step for generator
GEN_BATCHES = 1
INITIAL_CRITIC_EPOCHS = 0
BATCH_SIZE = 64
PRINT_CRITIC = 20


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="train WGANGP on mnist")
  parser.add_argument('--load_model', dest="load_model", default='', type=str, help="path to model to load")
  parser.add_argument('--savepath', dest="savepath", default='', type=str, help="path where model will be saved")

  args = parser.parse_args()
  if args.savepath:
    savepath = args.savepath
  elif args.load_model:
    savepath = args.load_model
  else:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    savedir_path = os.path.join(dir_path, 'saves')
    savepath = os.path.join(savedir_path, 'wgan_' + str(datetime.now()))

  picklepath = os.path.join(savepath, 'save.pickle')
 
  ds = wgan.makeDataset()

  if (args.load_model):
    # TODO figure out how to use keras save / load??
    gan = tf.saved_model.load(args.load_model)
  else:
    gan = wgan.WGAN()
    tf.saved_model.save(gan, savepath)
    save = {}
    with open(picklepath, "wb") as fp:
      pickle.dump(save, fp)

  with open(picklepath, "rb") as f: 
    gan_save = pickle.load(f)
    for x in ['gen_losses', 'critic_losses']:
      if not x in gan_save:
        gan_save[x] = []


  def preprocess(x):
    x = tf.cast(x['image'], wgan.FLOAT_TYPE)
    x = x / 255.0
    return x

  ds = ds.map(preprocess)
  ds = ds.cache()
  ds = ds.batch(BATCH_SIZE)
  ds = ds.shuffle(ds.cardinality())
  ds = ds.prefetch(tf.data.AUTOTUNE)

  for epoch in range(EPOCHS):
    print("EPOCH %d/%d" % (epoch, EPOCHS))
    
    for i,real_images in enumerate(ds):
      if epoch == 0 and i == 10 and PROFILE:
        tf.profiler.experimental.start('logdir')
      if epoch == 0 and i == 20 and PROFILE:
        tf.profiler.experimental.stop()
        

      critic_loss = gan.trainCritic(real_images, real_images.shape[0])
      gan_save['critic_losses'] += [critic_loss]

      # train only the critic for the first INITIAL_CRITIC_EPOCHS
      if (not i % NUM_CRITIC_PER_GEN) and (epoch >= INITIAL_CRITIC_EPOCHS):
        gen_loss = gan.trainGen(BATCH_SIZE)
        gan_save['gen_losses'] += [gen_loss]
        loss_str = ''.join('{:6f}, '.format(lossv) for lossv in (critic_loss, gen_loss))
        print('%d/%d: critic, gen: %s' % (i,ds.cardinality(),loss_str))
       
    print('Saving model...')

    # save some sampled images
    im_fn = os.path.join(savepath, 'image_ep' + str(epoch) + '.png')
    saveImage(gan, im_fn)


    # TODO figure out how to use keras save/load??
    tf.saved_model.save(gan, savepath)
    with open(picklepath, "wb") as fp:
      pickle.dump(gan_save, fp)
  
    
  
  
  
