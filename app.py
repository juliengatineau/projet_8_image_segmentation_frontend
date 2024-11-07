from flask import Flask, render_template, request, send_from_directory
import requests
import re
import os
from PIL import Image
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# --------------------------------------------------------------------
# VARIABLES
# --------------------------------------------------------------------


# Path to the images directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(ROOT_DIR, 'static/images')
SOURCE_DIR = os.path.join(IMAGES_DIR, 'source')
MASK_DIR = os.path.join(IMAGES_DIR, 'masque')
PRED_DIR = os.path.join(IMAGES_DIR, 'pred')
RESIZE_DIR = os.path.join(IMAGES_DIR, 'resize')

# Path to the generated directory in the backend
PREDICT_API_URL = 'https://projet8backend-dygac2e7f9h6c4ar.westeurope-01.azurewebsites.net/predict'

# Input dimensions expected by your Keras model
MODEL_INPUT_WIDTH = 256
MODEL_INPUT_HEIGHT = 128


# Extract image IDs and names from filenames in the leftimg directory
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


# --------------------------------------------------------------------
# IMAGES AND BASE ROUTE
# --------------------------------------------------------------------

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
@app.route('/images/pred/<path:filename>')
def serve_backend_image(filename):
    return send_from_directory(PRED_DIR, filename)


# Index page
@app.route('/')
def index():
    return render_template('index.html', image_ids=image_ids, image_names=image_names)


# --------------------------------------------------------------------
# PREDICT AND DISPLAY ROUTES
# --------------------------------------------------------------------

# Prediction route
@app.route('/api/predict', methods=['POST'])
def predict():
    logging.info('----------------------------predict-frontend---------------------------')
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

    # Resize and save the image
    image = Image.open(f"{SOURCE_DIR}/{real_image_filename}")
    logging.info('--- image opened')
    image = image.resize((MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
    logging.info('--- image resized')
    image.save(f"{RESIZE_DIR}/{real_image_filename}")
    logging.info('--- image saved')

    # send the image to the API
    with open(f"{RESIZE_DIR}/{real_image_filename}", "rb") as img_file:
        files = {'image': img_file}
        response = requests.post(PREDICT_API_URL, files=files, data={'predicted_mask_filename': predicted_mask_filename})

    logging.info('--- response received')
    img_io = BytesIO(response.content)
    image = Image.open(img_io)
    logging.info('--- image opened')

    resized_image = image.resize((2048,1024), Image.Resampling.LANCZOS)

    save_path = os.path.join(PRED_DIR, predicted_mask_filename)
    resized_image.save(save_path)
    logging.info('--- image saved')

    return render_template('redirect_post.html', image_id=image_id, real_image_filename=real_image_filename, real_mask_filename=real_mask_filename, predicted_mask_filename=predicted_mask_filename)



# Display route 
@app.route('/display', methods=['POST'])
def display():
    logging.info('----------------------------display---------------------------')
    image_id = request.form['image_id']
    real_image_filename = request.form['real_image_filename']
    real_mask_filename = request.form['real_mask_filename']
    predicted_mask_filename = request.form['predicted_mask_filename']

    return render_template('display.html', image_id=image_id, real_image=real_image_filename, real_mask=real_mask_filename, predicted_mask=predicted_mask_filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)