An implementation of WGAN-GP.


python train.py

runs on MNIST with the hyperparameters set in wgan.py and train.py.
Currently if you want to modify the hyperparams you have to do so in those files.

The model will be saved in a directory under saves.

You can change the savepath with the --savepath option
You can load models with the --load_model option

Example images are saved every epoch, here's a gif of the model learning (one image per epoch, over 20 epochs)


![Gif of mnist training](mnist.gif)
