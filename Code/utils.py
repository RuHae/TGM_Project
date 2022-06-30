# Here we can put helper functions
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import base64
import io
import medmnist
from medmnist import INFO, Evaluator
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

from tqdm import tqdm

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()

def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12, latent_dim=2):
    w = 28
    img = np.zeros((n*w, n*w,3))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.randint(-1,1, (1,latent_dims), dtype=torch.float, device=device)
            # z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            # print(z)
            x_hat = x_hat.reshape(3, 28, 28).moveaxis(0,2).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w,:] = x_hat
    plt.figure(figsize=(10,10))
    plt.imshow(img, extent=[*r0, *r1])
    plt.show()

def plot_real_vs_constructed(vae, test_dataset, device, plot=True):
    imgs = np.ones((4*29, 8*29, 3))
    diff = 0.
    for i in range(8):
        test_img = test_dataset[i][0][None,:,:,:].to(device)
        lat = vae.encoder.forward(test_img)
        img = vae.decoder.forward(lat)

        imgs[:28, i*28+(i*1):(i+1)*28+(i*1), :] = test_img[0].moveaxis(0, 2).cpu().detach().numpy()
        imgs[28+1:28*2+1, i*28+(i*1):(i+1)*28+(i*1), :] = img[0].moveaxis(0, 2).cpu().detach().numpy()

        diff += ((test_img - img)**2).mean()

    for i in range(8):
        test_img = test_dataset[i+8][0][None,:,:,:].to(device)
        lat = vae.encoder.forward(test_img)
        img = vae.decoder.forward(lat)

        imgs[2*29+1:3*28+3, i*28+(i*1):(i+1)*28+(i*1), :] = test_img[0].moveaxis(0, 2).cpu().detach().numpy()
        imgs[3*28+4:, i*28+(i*1):(i+1)*28+(i*1), :] = img[0].moveaxis(0, 2).cpu().detach().numpy()

        diff += ((test_img - img)**2).mean()

    if plot:
        print((diff/16).item())        
        plt.figure(figsize=(20,20))
        plt.imshow(imgs, cmap="gray")
        plt.plot()

    return imgs

def plot_generated(vae, device, plot=True):
    imgs = np.ones((4*29, 8*29, 3))
    diff = 0.
    for i in range(8):
        img_1 = vae.generate(device)
        img_2 = vae.generate(device)

        imgs[:28, i*28+(i*1):(i+1)*28+(i*1), :] = img_1
        imgs[28+1:28*2+1, i*28+(i*1):(i+1)*28+(i*1), :] = img_2 

    for i in range(8):
        img_1 = vae.generate(device)
        img_2 = vae.generate(device)

        imgs[2*29+1:3*28+3, i*28+(i*1):(i+1)*28+(i*1), :] = img_1
        imgs[3*28+4:, i*28+(i*1):(i+1)*28+(i*1), :] = img_2

    if plot:    
        plt.figure(figsize=(20,20))
        plt.imshow(imgs, cmap="gray")
        plt.plot()

    return imgs

def modify_latent(vae, test_dataset, img_nr=3, device="cpu", plot=True):
    imgs = np.ones((20*29, 20*29, 3))
    # z = torch.torch.distributions.Uniform(torch.Tensor([-1.8]), torch.Tensor([1.8])).sample([1,128]).view(1,128).to(device)
    test_img = test_dataset[img_nr][0][None,:,:,:].to(device)
    z = vae.encoder.forward(test_img)
    z_orig = z.clone()
    r = -1
    for j in range(20):
        z = z_orig.clone()
        idx = j #np.random.randint(0,128)
        # z[0, j*8:(j+1)*8] = -1.8
        if j == 10:
            r = 1
        for i in range(20):
            # z[0, j*8:(j+1)*8] += 0.1
            # z += 0.01 * abs(j-10) * 10 * r
            z += 0.01 * abs(j-10) * abs(i-10) * r
            img_1 = vae.generate(device, z=z)
            imgs[j*(28+1):(j+1)*28+(j*1), i*28+(i*1):(i+1)*28+(i*1), :] = img_1


    if plot:    
        plt.figure(figsize=(100,100))
        plt.imshow(imgs, cmap="gray")
        plt.plot()

    return imgs

def generate_gif(vae, test_dataset, img_nr=3, device="cpu", plot=True):
    test_img = test_dataset[img_nr][0][None,:,:,:].to(device)
    z = vae.encoder.forward(test_img)
    z_orig = z.clone()
    
    gif = []
    for j in range(100):
        z = z_orig.clone()
        z += 0.02 * (j-50)
        img_1 = vae.generate(device, z=z)
        im = Image.fromarray(np.uint8(img_1*255))
        gif.append(im)

    data = io.BytesIO()
    # gif[0].save(data, 'GIF', save_all=True,optimize=False, append_images=gif[1:], loop=0, duration=500)
    imageio.mimwrite(data, gif, "GIF")
    encoded_img_data = base64.b64encode(data.getvalue())
    
    # if plot:    
    #     plt.figure(figsize=(100,100))
    #     plt.imshow(gif)
    #     plt.plot()

    return encoded_img_data.decode('utf-8')

def load_dataset(data_flag, BATCH_SIZE = 64, download = True):
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

    return train_dataset, test_dataset, train_loader, train_loader_at_eval, test_loader