# Topic #3: VAE-based Medical Image Generator

The task for this project is to implement a Variational Autoencoder (VAE) model for the purpose of generating medical images.  

**GPU:** Recommended.

**Your task (0.3):** Train a VAE model to learn the distribution of images in [MedMNIST](https://medmnist.com). For +0.3 bonus, it is okay to focus only on one type of modality.

**Possible extension (0.7):**
For the full bonus (+0.7), implement and evaluate a VAE variant (e.g. Beta VAE, Discrete label VAE, VQ VAE, CVAE) to disentangle the latent space, or to learn the class-conditional distribution for at least three different image classes. Implement a simple user interface (PyQt, Flask+HTML, TkInter,...) where the user can select an image class (PathMNIST, ChestMNIST,...) to sample from.
