# TGM Report 5 Implementation 4

Topic #3: VAE-based Medical Image Generator

---

## Tasks
1. Create presentation
2. Finetune VQ-VAE and $\beta$-VAE and save the models
3. Last modifications on the frontend (flask app) 

---

## Who did what

**Felix:**
- basic flask template
- further investigation on the CVAE, but ultimately deciding not to use it as it does not fit our needs

**Zixuan:**
- improvements on VQ-VAE (parameter tuning and adding capability to also train on 1 channel data)

**Ruben:**

- extended flask template
- improvements on $\beta$-VAE (parameter tuning)

---

## Problems

1. CVAE generalizes to much, overfitts to much on the respective classes

---

## Solutions

1. Not to use the CVAE

---

## Outlook

- Have at least 6 trained models (3 for the VQ-VAE and 3 for the $\beta$-VAE)
- Working frontend with all the features (being able to select a model and a data set and display images from it)
- We want a presentation which describes our work

---

## Miscellaneous

1 and 3 row are ground truth images

2 and 4 are generated images

**VQ-VAE with PATHMNIST:**

<img src="https://cdn.discordapp.com/attachments/969293709063626845/990521037710045214/unknown.png" alt="img" style="zoom:20%;" />

**$\beta$-VAE with BLOODMNIST:**

<img src="https://cdn.discordapp.com/attachments/969293709063626845/992009746527830066/unknown.png" alt="img" style="zoom:60%;" />

**VQ-VAE with ORGANAMNSIT**

<img src="https://cdn.discordapp.com/attachments/969293709063626845/993067632016244796/unknown.png" alt="img" style="zoom:31%;" />
