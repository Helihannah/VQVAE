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

Model
------------




















