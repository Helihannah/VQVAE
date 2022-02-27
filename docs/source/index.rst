Welcome to VQVAE!
===================================

**VQVAE** (Vector Quantized Variational Autoencoder) was proposed in Neural Discrete Representation Learning by van der Oord et al. In traditional VAEs, the latent space is continuous and is sampled from a Gaussian distribution. It is generally harder to learn such a continuous distribution via gradient descent. VQ-VAEs, on the other hand, operate on a discrete latent space, making the optimization problem simpler. It does so by maintaining a discrete codebook. The codebook is developed by discretizing the distance between continuous embeddings and the encoded outputs. These discrete code words are then fed to the decoder, which is trained to generate reconstructed samples.

Contents
--------

.. toctree::

   model
   Results
