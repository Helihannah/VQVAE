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

Hyperparameters are parameters whose value are used to control the learning process.

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














Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

