# app.py
import sys
sys.path.insert(1, '../Code')

from flask import Flask, render_template  
import torch
import numpy as np

from PIL import Image
import base64
import io

from basic_vae import VariationalAutoencoder
from utils import plot_real_vs_constructed 
from utils import load_dataset 


# Loading test datasets
_, test_dataset_blood, _, _, _ = load_dataset("bloodmnist")
# TODO load the other datesets aswell

# TODO move model loading to here due to performance of page loading


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
    vae.load_from_file(path="../Code/blood_beta")

    # z = torch.randint(-1,1, (1,latent_dims), dtype=torch.float, device=device)
    # img = vae.decoder.forward(z)
    # imgs = img.moveaxis(1,3).cpu().detach().numpy()[0]

    imgs = plot_real_vs_constructed(vae, test_dataset_blood, device, plot=False)

    im = Image.fromarray(np.uint8(imgs*255))
    data = io.BytesIO()
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('results.html', img_data=encoded_img_data.decode('utf-8'))

@app.route("/results_OrganAMNIST", methods=['GET']) 
def results_OrganAMNIST():
    return None

if __name__ == "__main__":
    app.run(debug=True)