# TGM Report 3 Implementation 2

Topic #3: VAE-based Medical Image Generator

---

## Tasks

1. Extend basic version of VAE with different variants of it to see what works best
2. Decide on three datasets we want to use for the final result in order to tweak the models accordingly
3. Have a look at front-end solutions (PyQT, Flask, TkInter)

---

## Who did what

**Felix:**
- More research on VAE variants

**Zixuan:**

- More research on VAE variants

**Ruben:**

- Extension of the boilerplate code and put it on GPU

---

## Problems

1. Generated pictures from a VAE are very blurry

---

## Solutions

1. Known problem to VAE, due to the bottleneck and the way the image gets encoded -> One solution would be to optimize latent vector size

---

## Outlook

- We want to have three working variants of VAE at the end of this implementation cycle
- Basic structure of the front-end

---

## Miscellaneous

[Paper to more variants](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171997)
