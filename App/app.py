# app.py
import sys
sys.path.insert(1, '../Code')

from flask import Flask, render_template  
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

vae_blood = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
vae_blood.load_from_file(path="../Code/blood_beta_v6")

vae_organ = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=1).to(device) # GPU
vae_organ.load_from_file(path="../Code/organ_beta_v1")

vae_path = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
vae_path.load_from_file(path="../Code/path_beta")

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

@app.route("/results_BloodMNIST", methods=['GET']) 
def results_BloodMNIST():

    imgs_real_vs_recon = plot_real_vs_constructed(vae_blood, test_dataset_blood, device, plot=False)
    imgs_gen = plot_generated(vae_blood, device, plot=False)
    
    img_rvg = prepare_imgs(imgs_real_vs_recon)
    img_gen = prepare_imgs(imgs_gen)
    img_gif = generate_gif(vae_blood, test_dataset_blood, img_nr=45, device=device)

    return render_template('results.html', img_data_rvg=img_rvg, img_data_gen=img_gen, img_gif=img_gif)

@app.route("/results_OrganMNIST", methods=['GET']) 
def results_OrganMNIST():
    
    imgs_real_vs_recon = plot_real_vs_constructed(vae_organ, test_dataset_organ, device, plot=False)
    imgs_gen = plot_generated(vae_organ, device, plot=False)
    
    img_rvg = prepare_imgs(imgs_real_vs_recon)
    img_gen = prepare_imgs(imgs_gen)
    img_gif = generate_gif(vae_organ, test_dataset_organ, img_nr=45, device=device)

    return render_template('results.html', img_data_rvg=img_rvg, img_data_gen=img_gen, img_gif=img_gif)

@app.route("/results_PathMNIST", methods=['GET']) 
def results_PathMNIST():
    
    imgs_real_vs_recon = plot_real_vs_constructed(vae_path, test_dataset_path, device, plot=False)
    imgs_gen = plot_generated(vae_path, device, plot=False)
    
    img_rvg = prepare_imgs(imgs_real_vs_recon)
    img_gen = prepare_imgs(imgs_gen)
    img_gif = generate_gif(vae_path, test_dataset_path, img_nr=45, device=device)

    return render_template('results.html', img_data_rvg=img_rvg, img_data_gen=img_gen, img_gif=img_gif)

if __name__ == "__main__":
    app.run(debug=True)