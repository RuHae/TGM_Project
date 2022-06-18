import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, latent_dims, channels, mode="beta_vae"):
        super(Encoder, self).__init__()
        self.channels = channels
        self.mode = mode
        # 28*28*3
        self.conv1 = nn.Conv2d(self.channels, 6, 3, stride=2)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2)   

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(12)
        
        self.linear1 = nn.Linear(432, 256)
        # self.linear2 = nn.Linear(256, latent_dims)
        self.linear2 = nn.Linear(432, latent_dims)
        # self.linear3 = nn.Linear(256, latent_dims)
        self.linear3 = nn.Linear(432, latent_dims)
        # self.linear1 = nn.Linear(2352, 512)
        # self.linear2 = nn.Linear(512, latent_dims)
        # self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to("cuda:0")#.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to("cuda:0")#.cuda()
        self.kl = 0

    # this is for the beta vae
    def sampling_beta(self, x):
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        # to keep the latent in the region of N(0,1)
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        # self.kl = ((sigma**2 + mu**2)/2 - torch.log(sigma) - 1/2).sum()
        self.kl = 0.5 * torch.sum(torch.exp(sigma) + mu**2 - 1 - sigma)
        return z

    def forward(self, x):
        # x = torch.flatten(x, start_dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(-1, 12 * 6 * 6)
        # x = F.relu(self.linear1(x))

        if self.mode == "beta_vae":
            z = self.sampling_beta(x)
        else:
            z = x
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, channels):
        super(Decoder, self).__init__()
        self.channels = channels
        self.conv1 = nn.ConvTranspose2d(12,6,3, stride=2)
        self.conv2 = nn.ConvTranspose2d(6,3,3, stride=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(3,self.channels,1, stride=1)

        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(3)

        self.linear1 = nn.Linear(latent_dims, 432)
        # self.linear1 = nn.Linear(latent_dims, 256)
        self.linear2 = nn.Linear(256, 432)
        # self.linear1 = nn.Linear(latent_dims, 512)
        # self.linear2 = nn.Linear(512, 2352)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        # z = F.relu(self.linear2(z))
        z = z.reshape(-1, 12, 6, 6)
        z = self.conv1(z)
        # z = self.bn1(z)
        z = F.relu(z)
        z = self.conv2(z)
        # z = self.bn2(z)
        z = F.relu(z)
        z = self.conv3(z)
        return z.reshape((-1, self.channels, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, mode="beta_vae", channels=2):
        super(VariationalAutoencoder, self).__init__()
        self.channels = channels
        self.mode = mode
        self.encoder = Encoder(latent_dims, self.channels, mode)
        self.decoder = Decoder(latent_dims, self.channels)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def train(self, data, epochs=5, lr=0.001, device="cpu", fast=False):
        # only for fast training
        nsamples = 100
        running_loss = 0.
        running_diff = 0.
        running_kl = 0.
        running_bce = 0.
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            i = 0
            for x, y in tqdm(data):
                if fast and i > nsamples:
                    break
                i= i + 1
                # print(x.shape)
                x = x.to(device) # GPU
    
                # for sample in x:
                opt.zero_grad()
                x_hat = self.forward(x)
                diff = ((x - x_hat)**2).sum()
                # diff = ((x - x_hat)**2).mean()
                # bce = F.binary_cross_entropy_with_logits(x_hat, x, weight=torch.Tensor([.25], device=device), reduction="sum")
                # bce = F.cross_entropy(x_hat, x, reduction="sum")
                
                loss = diff + self.encoder.kl             
                # loss = self.encoder.kl                              
                # loss = bce + 10 * self.encoder.kl + diff                        
                loss.backward()
                opt.step()

                # Gather data and report
                running_loss += loss.item()
                running_diff += diff.item()
                running_kl += self.encoder.kl.item()
                # running_bce += bce.item()
            
            last_loss = running_loss / i # loss per batch
            last_diff = running_diff / i # loss per batch
            last_kl = running_kl / i # loss per batch
            last_bce = running_bce / i # loss per batch
            print("Iteration:", epoch, "Loss:", round(last_loss,2), "Diff:", round(last_diff, 2), "KL:", round(last_kl,2), "BCE:", round(last_bce,2))
            running_loss = 0.
            running_diff = 0.
            running_kl = 0.
            running_bce = 0.        