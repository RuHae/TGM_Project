# Here we can put helper functions
import base64
import io

import imageio
import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from medmnist import INFO
from PIL import Image


def plot_real_vs_constructed(vae, test_dataset, device, plot=True, model="beta-vae"):
    '''
    Plots a comparison of test images and images that were created from them by a vae
    Rows 1 and 3 are test images
    Rows 2 and 4 are generated images
    '''
    imgs = np.ones((4*29, 8*29, 3)) # image size
    diff = 0.
    for i in range(8):
        test_img = test_dataset[i][0][None,:,:,:].to(device)

        # make it compatible with beta-vae and vq-vae
        if model == "beta-vae":
            img = vae.forward(test_img)
        elif model == "vq-vae":
            _, img, _ = vae.forward(test_img)
        else:
            print("Error: please supply model name")

        # cut values of at 0 and 1
        img[img > 1] = 1
        img[img < 0] = 0

        # put images into numpy array
        imgs[:28, i*28+(i*1):(i+1)*28+(i*1), :] = test_img[0].moveaxis(0, 2).cpu().detach().numpy()
        imgs[28+1:28*2+1, i*28+(i*1):(i+1)*28+(i*1), :] = img[0].moveaxis(0, 2).cpu().detach().numpy()

        # MSE
        diff += ((test_img - img)**2).mean()

    for i in range(8):
        test_img = test_dataset[i+8][0][None,:,:,:].to(device)

        # make it compatible with beta-vae and vq-vae
        if model == "beta-vae":
            img = vae.forward(test_img)
        elif model == "vq-vae":
            _, img, _ = vae.forward(test_img)
        else:
            print("Error: please supply model name")

        # cut values of at 0 and 1
        img[img > 1] = 1
        img[img < 0] = 0

        # put images into numpy array
        imgs[2*29+1:3*28+3, i*28+(i*1):(i+1)*28+(i*1), :] = test_img[0].moveaxis(0, 2).cpu().detach().numpy()
        imgs[3*28+4:, i*28+(i*1):(i+1)*28+(i*1), :] = img[0].moveaxis(0, 2).cpu().detach().numpy()

        # MSE
        diff += ((test_img - img)**2).mean()

    if plot:
        print((diff/16).item())        
        plt.figure(figsize=(20,20))
        plt.imshow(imgs, cmap="gray")
        plt.plot()

    return imgs

def plot_generated(vae, device, plot=True, u_bound=1.8, l_bound=-1.8):
    '''
    Generate a grid of 4x8 random generated images
    '''
    imgs = np.ones((4*29, 8*29, 3))
    diff = 0.
    for i in range(8):
        img_1 = vae.generate(device, u_bound=u_bound, l_bound=l_bound)
        img_2 = vae.generate(device, u_bound=u_bound, l_bound=l_bound)

        # cut values of at 0 and 1
        img_1[img_1 > 1] = 1
        img_1[img_1 < 0] = 0
        img_2[img_1 > 1] = 1
        img_2[img_1 < 0] = 0

        imgs[:28, i*28+(i*1):(i+1)*28+(i*1), :] = img_1
        imgs[28+1:28*2+1, i*28+(i*1):(i+1)*28+(i*1), :] = img_2 

    for i in range(8):
        img_1 = vae.generate(device, u_bound=u_bound, l_bound=l_bound)
        img_2 = vae.generate(device, u_bound=u_bound, l_bound=l_bound)

        # cut values of at 0 and 1
        img_1[img_1 > 1] = 1
        img_1[img_1 < 0] = 0
        img_2[img_1 > 1] = 1
        img_2[img_1 < 0] = 0

        imgs[2*29+1:3*28+3, i*28+(i*1):(i+1)*28+(i*1), :] = img_1
        imgs[3*28+4:, i*28+(i*1):(i+1)*28+(i*1), :] = img_2

    if plot:    
        plt.figure(figsize=(20,20))
        plt.imshow(imgs, cmap="gray")
        plt.plot()

    return imgs

def modify_latent(vae, test_dataset, img_nr=3, device="cpu", plot=True, model="beta-vae"):
    '''
    Plot a grid of 20x20 images for which the latent space is modified
    '''
    imgs = np.ones((20*29, 20*29, 3))
    test_img = test_dataset[img_nr][0][None,:,:,:].to(device)

    # make it compatible with beta-vae and vq-vae
    if model == "beta-vae":
            z = vae.encoder.forward(test_img)
    elif model == "vq-vae":
        z = vae._encoder(test_img)
    else:
        print("Error: please supply model name")

    z_orig = z.clone()
    r = -1
    for j in range(20):
        z = z_orig.clone()

        if j == 10:
            r = 1
        
        for i in range(20):
            z += 0.01 * abs(j-10) * abs(i-10) * r # generates number between -1 and 1
            img_1 = vae.generate(device, z=z)
            imgs[j*(28+1):(j+1)*28+(j*1), i*28+(i*1):(i+1)*28+(i*1), :] = img_1

    if plot:    
        plt.figure(figsize=(100,100))
        plt.imshow(imgs, cmap="gray")
        plt.plot()

    return imgs

def generate_gif(vae, test_dataset, img_nr=3, device="cpu", plot=True, model="beta-vae"):
    '''
    Generates a base64 encoded GIF of how images changes if latent vector gets modified
    '''
    test_img = test_dataset[img_nr][0][None,:,:,:].to(device)
    
    # make it compatible with beta-vae and vq-vae
    if model == "beta-vae":
            z = vae.encoder.forward(test_img)
    elif model == "vq-vae":
        z = vae._encoder(test_img)
    else:
        print("Error: please supply model name")

    z_orig = z.clone()
    
    gif = []
    for j in range(100):
        z = z_orig.clone()
        z += 0.02 * (j-50) # generates movement between -1 and 1
        img_1 = vae.generate(device, z=z)
        h,w,c = img_1.shape
        if c == 1:
            img_1 = img_1.reshape(h,w)
        
        # cut values of at 0 and 1
        img_1[img_1 > 1] = 1
        img_1[img_1 < 0] = 0

        # convert to PIL images
        im = Image.fromarray(np.uint8(img_1*255))
        gif.append(im)

    data = io.BytesIO() # create store in memory
    imageio.mimwrite(data, gif, "GIF") # genereate gif
    encoded_img_data = base64.b64encode(data.getvalue()) # encode it to base64

    return encoded_img_data.decode('utf-8')

def load_dataset(data_flag, BATCH_SIZE = 64, download = True):
    '''
    Loads and transforms specified data set with data augmentation (random vertical and horizontal flips)
    '''
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
