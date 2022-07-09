# app.py
from crypt import methods
import sys

from requests import request
sys.path.insert(1, '../Code')

from flask import Flask, render_template, request  
import torch
import numpy as np

from PIL import Image
import base64
import io

from beta_conv_vae import VariationalAutoencoder
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

beta_vae_blood = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
beta_vae_blood.load_from_file(path="../Code/models/blood_beta_v6")

beta_vae_organ = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=1).to(device) # GPU
beta_vae_organ.load_from_file(path="../Code/models/organ_beta_v1")

beta_vae_path = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
beta_vae_path.load_from_file(path="../Code/models/path_beta_v2")

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
        elif model_name == "vq-vae":
            model = vq_vae_path
        else:
            print("Error: model not found")

    elif dataset_name == "BloodMNIST":
        dataset = test_dataset_blood
        if model_name == "beta-vae":
            model = beta_vae_blood
        elif model_name == "vq-vae":
            model = vq_vae_blood
        else:
            print("Error: model not found")

    elif dataset_name == "OrganAMNIST":
        dataset = test_dataset_organ
        if model_name == "beta-vae":
            model = beta_vae_organ
        elif model_name == "vq-vae":
            model = vq_vae_organ
        else:
            print("Error: model not found")
    else:
        print("Error: dataset not found")

    

    imgs_real_vs_recon = plot_real_vs_constructed(model, dataset, device, plot=False)
    imgs_gen = plot_generated(model, device, plot=False)
    
    img_rvg = prepare_imgs(imgs_real_vs_recon)
    img_gen = prepare_imgs(imgs_gen)
    img_gif = generate_gif(model, dataset, img_nr=45, device=device) # TODO: add random image number

    return render_template('results.html',model_name=str(model_name).upper(), dataset_name=str(dataset_name), img_data_rvg=img_rvg, img_data_gen=img_gen, img_gif=img_gif)