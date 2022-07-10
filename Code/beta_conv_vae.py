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
        modules = []
        # if hidden_dims is None:
        # hidden_dims = [28, 56, 112, 224]
        hidden_dims = [32, 64, 128, 256]

        in_channels = self.channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)

        self.linear_mu = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.linear_sigma = nn.Linear(hidden_dims[-1]*4, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to("cuda:0")#.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to("cuda:0")#.cuda()
        self.kl = 0

    # this is for the beta vae
    def sampling_beta(self, x):
        mu = self.linear_mu(x)
        log_var = self.linear_sigma(x)

        self.kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        if self.mode == "beta_vae":
            z = self.sampling_beta(x)
        else:
            z = x
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, channels):
        super(Decoder, self).__init__()
        self.channels = channels
        modules = []
        # if hidden_dims is None:
        # hidden_dims = [28, 56, 112, 224]
        hidden_dims = [32, 64, 128, 256]

        self.decoder_input = nn.Linear(latent_dims, hidden_dims[-1] * 4)

        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                                hidden_dims[-1],
                                kernel_size=2,
                                stride=2,
                                padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels= self.channels,
                        kernel_size= 3),
            nn.Tanh())

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 256,2,2)
        z = self.decoder(z)
        z = self.final_layer(z)
        return z.reshape((-1, self.channels, 28, 28))


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, mode="beta_vae", channels=2):
        super(VariationalAutoencoder, self).__init__()
        self.channels = channels
        self.mode = mode
        self.latent_dims = latent_dims
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
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            i = 0
            for x, y in tqdm(data):
                if fast and i > nsamples:
                    break
                i= i + 1
                # print(x.shape)
                x = x.to(device) # GPU
    
                # for sample in x:
                self.opt.zero_grad()
                x_hat = self.forward(x)
                diff = ((x - x_hat)**2).sum()

                # loss = diff + self.encoder.kl       
                self.C_max = torch.tensor([25]).to(device)
                self.C_stop_iter = epochs
                self.num_iter = epoch
                self.gamma = .25
                C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
                loss = diff + self.gamma * (self.encoder.kl - C).abs() #  https://arxiv.org/pdf/1804.03599.pdf               
                loss.backward()
                self.opt.step()

                # Gather data and report
                running_loss += loss.item()
                running_diff += diff.item()
                running_kl += self.encoder.kl.item()
            
            last_loss = running_loss / i # loss per batch
            last_diff = running_diff / i # loss per batch
            last_kl = running_kl / i # loss per batch
            print("Iteration:", epoch, "Loss:", round(last_loss,2), "Diff:", round(last_diff, 2), "KL:", round(last_kl,2))
            running_loss = 0.
            running_diff = 0.
            running_kl = 0.
     
    def generate(self, device="cpu", z=None, u_bound=1.8, l_bound=-1.8):
        if z == None:
            z = torch.torch.distributions.Uniform(torch.Tensor([l_bound]), torch.Tensor([u_bound])).sample([1,128]).view(1,128).to(device)
        img = self.decoder.forward(z)
        return img[0].moveaxis(0, 2).cpu().detach().numpy()

    def save_to_file(self, path):
        torch.save(self.state_dict(), path)
        print("Model safed to", path)
        return None

    def load_from_file(self, path):
        self.load_state_dict(torch.load(path))
        print("Model loaded from", path)
        return None