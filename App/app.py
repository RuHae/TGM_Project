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
from utils import load_dataset 


# Loading test datasets
_, test_dataset_blood, _, _, _ = load_dataset("bloodmnist")
_, test_dataset_path, _, _, _ = load_dataset("pathmnist")
_, test_dataset_organ, _, _, _ = load_dataset("organamnist")
# TODO load the other datesets aswell

# TODO move model loading to here due to performance of page loading


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

@app.route("/results_PathMNIST", methods=['GET']) 
def results_PathMNIST():

    return None

@app.route("/results_BloodMNIST", methods=['GET']) 
def results_BloodMNIST():
    latent_dims = 128 #hyperparameter we can optimze?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    vae = VariationalAutoencoder(latent_dims, mode="beta_vae", channels=3).to(device) # GPU
    vae.load_from_file(path="../Code/blood_beta_v5")

    imgs_real_vs_recon = plot_real_vs_constructed(vae, test_dataset_blood, device, plot=False)
    imgs_gen = plot_generated(vae, device, plot=False)
    
    img_rvg = prepare_imgs(imgs_real_vs_recon)
    img_gen = prepare_imgs(imgs_gen)

    return render_template('results.html', img_data_rvg=img_rvg, img_data_gen=img_gen)



if __name__ == "__main__":
    app.run(debug=True)