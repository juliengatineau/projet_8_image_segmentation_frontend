from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, redirect, url_for
import requests
import re
import os
from PIL import Image
import json
import numpy as np


# Path to the images directory
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/images')
SOURCE_DIR = os.path.join(IMAGES_DIR, 'source')
MASK_DIR = os.path.join(IMAGES_DIR, 'masque')

# Path to the generated directory in the backend
PREDICT_API_URL = 'http://127.0.0.1:8001/predict'

BACKEND_GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend', 'generated')


# Extract image IDs from filenames in the leftimg directory
def extract_image_ids():
    image_ids = []
    image_names = []
    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith('.png'):
            image_names.append(filename)
            match = re.search(r'_(\d{6})_leftImg8bit', filename)
            if match:
                image_ids.append(int(match.group(1)))
    return image_ids, image_names

image_ids, image_names = extract_image_ids()


app = Flask(__name__)

# Route to serve frontend images
@app.route('/images/source/<path:filename>')
def serve_frontend_image(filename):
    return send_from_directory(SOURCE_DIR, filename)

# Route to serve frontend masks
@app.route('/images/masque/<path:filename>')
def serve_frontend_mask(filename):
    return send_from_directory(MASK_DIR, filename)

# Route to serve backend generated images
@app.route('/images/generated/<path:filename>')
def serve_backend_image(filename):
    return send_from_directory(BACKEND_GENERATED_DIR, filename)


# Index page
@app.route('/')
def index():
    return render_template('index.html', image_ids=image_ids, image_names=image_names)



# Prediction route
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the image URL from the form
    image_url = request.form['image_url']
    # Extract the filename
    real_image_filename = os.path.basename(image_url)
    # Get the image name before suffixe
    base_filename = real_image_filename.rsplit('_', 1)[0]
    # Add the mask suffixe
    real_mask_filename = f'{base_filename}_gtFine_color.png'


    # Add the predicted mask suffixe
    predicted_mask_filename = f'{base_filename}_pred.png'
    # Extract the image ID
    image_id = re.search(r'_(\d{6})_leftImg8bit', real_image_filename).group(1)

    response = requests.post(PREDICT_API_URL, json={'image_url': image_url, "predicted_mask_filename": predicted_mask_filename})

    response_data = response.json()
    predicted_mask_path = response_data['predicted_mask_path']
    print('--------------------------------8000-1-------------------------------')
    print(predicted_mask_path)
    return render_template('redirect_post.html', image_id=image_id, real_image_filename=real_image_filename, real_mask_filename=real_mask_filename, predicted_mask_filename=predicted_mask_filename)



# Display route 
@app.route('/display', methods=['POST'])
def display():
    print('--------------------------------8000-3-------------------------------')
    image_id = request.form['image_id']
    real_image_filename = request.form['real_image_filename']
    real_mask_filename = request.form['real_mask_filename']
    predicted_mask_filename = request.form['predicted_mask_filename']
 
    return render_template('display.html', image_id=image_id, real_image=real_image_filename, real_mask=real_mask_filename, predicted_mask=predicted_mask_filename)


    
if __name__ == '__main__':
    app.run(debug=True, port=8000)