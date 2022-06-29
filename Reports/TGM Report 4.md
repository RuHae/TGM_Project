# TGM Report 4 

Topic #3: VAE-based Medical Image Generator

---

## Tasks
1. Decide and improve models on the three decided datasets (Finetuning)
2. Decide which model to use for the respective dataset
3. Setup basic flask template

---

## Who did what

**Felix:**
- Extend and implement VAE to CVAE
- Research about Flask

**Zixuan:**
- Extend and implement VAE to Vq-VAE
- Research about Flask

**Ruben:**
- Extend $\beta$-VAE with more layers and latent size
- Research about Flask

**Group**
- Decide on three datasets we want to use for the final result: PathMNIST, BloodMNIST, OrganAMNIST (+OrganCMNIST + OrganSMNIST to generate more traning data)

---

## Problems

1. Model implementations only work for one dataset
2. CUDA assertion error in Colab because of updated CUDA version

---

## Solutions

1. Debug our current implementations to find the root cause (channel/ input dimensions)
2. Install explicit pytorch version for the CUDA version

---

## Outlook

- Work on the tasks
- Get more deeply into the variants of VAE

---

## Miscellaneous

N/A
