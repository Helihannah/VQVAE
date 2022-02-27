Architecture
=====

.. _Library:

Library
------------

To build the architecture of VQVAE, first import library:

.. code-block:: console

   import os
   import numpy as np
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torchvision import datasets, transforms
   from sklearn.manifold import TSNE
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns
   %matplotlib inline

.. _Hyperparameters:

Hyperparameters
------------

Hyperparameters are parameters whose values are used to control the learning process.

.. code-block:: console

   batch_size = 128
   embedding_dim = 16
   num_embeddings = 128

   epochs = 50
   print_freq = 100

   lr = 1e-3

.. _Dataset:

Dataset
------------

Test on the MNIST dataset and compute the variance of the whole training set to normalise the Mean Squared Error.

The code for normalised part is shown below:

.. code-block:: console

   train_images = []
   for images, labels in train_loader:
      train_images.append(images)
   train_images = torch.cat(train_images, dim=0)
   train_data_variance = torch.var(train_images)
   
.. _Moddel:

Standard VQ-VAE consists of three parts: encoder, decoder and embedding space.

encoder
------------

Encoder consists of three convolutional layers and two activation layers

.. code-block:: console

   class Encoder(nn.Module):
       """Encoder of VQ-VAE"""

       def __init__(self, in_dim=3, latent_dim=16):
           super().__init__()
           self.in_dim = in_dim
           self.latent_dim = latent_dim

           self.convs = nn.Sequential(
               nn.Conv2d(in_dim, 32, 3, stride=2, padding=1),
               nn.ReLU(inplace=True),
               nn.Conv2d(32, 64, 3, stride=2, padding=1),
               nn.ReLU(inplace=True),
               nn.Conv2d(64, latent_dim, 1),
           )

       def forward(self, x):
           return self.convs(x)

decoder
----------------

The structure of encoder and decoder is almost identical except that convolutional layers are replaced by transposed convolution layers. 

.. code-block:: console

   class Decoder(nn.Module):
       """Decoder of VQ-VAE"""

       def __init__(self, out_dim=1, latent_dim=16):
           super().__init__()
           self.out_dim = out_dim
           self.latent_dim = latent_dim

           self.convs = nn.Sequential(
               nn.ConvTranspose2d(latent_dim, 64, 3, stride=2, padding=1, output_padding=1),
               nn.ReLU(inplace=True),
               nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
               nn.ReLU(inplace=True),
               nn.ConvTranspose2d(32, out_dim, 3, padding=1),
           )

       def forward(self, x):
           return self.convs(x)



















