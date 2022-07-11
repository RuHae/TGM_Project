# app.py
import sys
sys.path.insert(1, '../Code')

from flask import Flask, render_template, request  
import torch
import numpy as np

from PIL import Image
import base64
import io

from beta_conv_vae import VariationalAutoencoder as VariationalAutoencoder_beta
from vq_vae_model import VariationalAutoencoder as VariationalAutoencoder_vq
from utils import plot_real_vs_constructed 
from utils import plot_generated  
from utils import generate_gif 
from utils import load_dataset 


# Loading test datasets
_, test_dataset_blood, _, _, _ = load_dataset("bloodmnist")
_, test_dataset_path, _, _, _ = load_dataset("pathmnist")
_, test_dataset_organ, _, _, _ = load_dataset("organamnist")

# Loading vae models
latent_dims = 128 #hyperparameter we can optimze?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# beta-vae
beta_vae_blood = VariationalAutoencoder_beta(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
beta_vae_blood.load_from_file(path="../Code/models/blood_beta_v7")

beta_vae_organ = VariationalAutoencoder_beta(latent_dims, mode="beta_vae", channels=1).to(device) # GPU
beta_vae_organ.load_from_file(path="../Code/models/organ_beta_v2")

beta_vae_path = VariationalAutoencoder_beta(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
beta_vae_path.load_from_file(path="../Code/models/path_beta_v2")

# vq-vae
vq_vae_blood = VariationalAutoencoder_vq(num_hiddens=latent_dims, num_residual_layers=2, num_residual_hiddens=32, \
    num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99, \
        input_channel=3,output_channel=3).to(device)
vq_vae_blood.load_from_file(path="../Code/models/bloodmnist_epochs_100.tar")

vq_vae_organ = VariationalAutoencoder_vq(num_hiddens=latent_dims, num_residual_layers=2, num_residual_hiddens=32, \
    num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99, \
        input_channel=1,output_channel=1).to(device)
vq_vae_organ.load_from_file(path="../Code/models/organamnist_epochs_100.tar")

vq_vae_path = VariationalAutoencoder_vq(num_hiddens=latent_dims, num_residual_layers=2, num_residual_hiddens=32, \
    num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.99, \
        input_channel=3,output_channel=3).to(device)
vq_vae_path.load_from_file(path="../Code/models/pathmnist_epochs_100.tar")

def prepare_imgs(imgs):
    im = Image.fromarray(np.uint8(imgs*255))
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return encoded_img_data.decode('utf-8')

app = Flask(__name__) # name for the Flask app (refer to output)

@app.route("/home", methods=['GET']) # decorator
def home(): # route handler function
    # returning a response
    return render_template('index.html')

@app.route("/results", methods=["GET", "POST"])
def results():
    model_name = request.form.get("model")
    dataset_name = request.form.get("dataset")
    print(request.form.get("dataset"))
    print(request.form.get("model"))


    if dataset_name == "PathMNIST":
        dataset = test_dataset_path
        if model_name == "beta-vae":
            model = beta_vae_path
            l_bound = -2.34
            u_bound = 2.27
        elif model_name == "vq-vae":
            model = vq_vae_path
            l_bound = 0
            u_bound = 13
        else:
            print("Error: model not found")

    elif dataset_name == "BloodMNIST":
        dataset = test_dataset_blood
        if model_name == "beta-vae":
            model = beta_vae_blood
            l_bound = -2.21
            u_bound = 2.26
        elif model_name == "vq-vae":
            model = vq_vae_blood
            l_bound = 0
            u_bound = 13
        else:
            print("Error: model not found")

    elif dataset_name == "OrganAMNIST":
        dataset = test_dataset_organ
        if model_name == "beta-vae":
            model = beta_vae_organ
            l_bound = -2.64
            u_bound = 2.64
        elif model_name == "vq-vae":
            model = vq_vae_organ
            l_bound = 0
            u_bound = 20
        else:
            print("Error: model not found")
    else:
        print("Error: dataset not found")

    

    imgs_real_vs_recon = plot_real_vs_constructed(model, dataset, device, plot=False, model=model_name)
    imgs_gen = plot_generated(model, device, plot=False, u_bound=u_bound, l_bound=l_bound)
    
    img_rvg = prepare_imgs(imgs_real_vs_recon)
    img_gen = prepare_imgs(imgs_gen)
    img_gif = generate_gif(model, dataset, img_nr=45, device=device, model=model_name) # TODO: add random image number

    return render_template('results.html',model_name=str(model_name).upper(), dataset_name=str(dataset_name), img_data_rvg=img_rvg, img_data_gen=img_gen, img_gif=img_gif)