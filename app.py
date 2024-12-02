import os
import numpy as np
import replicate
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import json
import logging
from dotenv import load_dotenv
import requests
from PIL import Image
import requests
import replicate.exceptions


# Import the segmentation model and feature extractor
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch.nn.functional as F

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure Replicate API token is set
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN is not set in the environment.")

# Initialize Replicate client with API token
replicate.Client(api_token=REPLICATE_API_TOKEN)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)
logging.debug(f"Using Replicate API token: {REPLICATE_API_TOKEN}")

def allowed_file(filename):
    """Check if the file is allowed based on the extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the pre-trained segmentation model and feature extractor
feature_extractor = SegformerFeatureExtractor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

@app.route('/')
def upload_image():
    """Render the upload page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and process segmentation."""
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Process the image with the segmentation model
        original_image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=original_image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        # Resize logits to original image size
        upsampled_logits = F.interpolate(
            logits,
            size=original_image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        # Get the segmentation map
        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

        # Save the segmentation data for the client
        seg_path = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{filename}.npy')
        np.save(seg_path, pred_seg.astype(int))

        return redirect(url_for('edit_image', image_filename=filename))
    return redirect(request.url)

@app.route('/edit/<image_filename>')
def edit_image(image_filename):
    """Render the image editing page."""
    return render_template('edit_image.html', image_filename=image_filename)

@app.route('/get_segmentation_mask/<image_filename>')
def get_segmentation_mask(image_filename):
    """Serve the segmentation mask as JSON."""
    seg_path = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{image_filename}.npy')
    if os.path.exists(seg_path):
        pred_seg = np.load(seg_path)
        # Convert the mask to a list of lists for JSON serialization
        mask_list = pred_seg.astype(int).tolist()
        return jsonify({'mask': mask_list})
    else:
        return jsonify({'error': 'Segmentation mask not found.'}), 404

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process the image and send it for inpainting using Replicate API."""
    try:
        image_filename = request.form['image_filename']
        prompt = request.form['prompt']
        mask_data = request.form['mask_data']  # This is the selected segment label

        # Convert mask_data to int if it's not already
        selected_label = int(mask_data)

        # Load the original image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        original_image = Image.open(image_path).convert("RGB")

        # Create a mask image with a fully white background (255 = unchanged areas)
        mask_image = Image.new("L", original_image.size, 255)

        # Load the segmentation mask
        seg_path = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{image_filename}.npy')
        pred_seg = np.load(seg_path)

        # Create a mask where the selected segment is black (0), others are white (255)
        mask_array = np.array(mask_image)

        # Set pixels where pred_seg == selected_label to 0 (black)
        mask_array[pred_seg == selected_label] = 0

        # Convert back to PIL Image
        mask_image = Image.fromarray(mask_array.astype(np.uint8), mode='L')

        # Save the original and mask images temporarily for API request
        init_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'init_' + image_filename)
        mask_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + image_filename)
        original_image.save(init_image_path, format="PNG")
        mask_image.save(mask_image_path, format="PNG")

        # Use replicate.run to process the image using the correct model
        output = replicate.run(
            "ideogram-ai/ideogram-v2-turbo",
            input={
                "prompt": prompt,
                "image": open(init_image_path, "rb"),
                "mask": open(mask_image_path, "rb"),
                "resolution": "None",
                "style_type": "None",
                "aspect_ratio": "1:1",
                "negative_promt": "Altered fabric texture, unrealistic or distorted text, misaligned font, text that doesn't follow the contours of the fabric, unnatural lighting, changes to the original design or color of the clothing, or any disruption to the garment's natural appearance.",
                "magic_prompt_option": "Auto"
            }
        )

        # Handle the output and save it to the output folder
        if isinstance(output, replicate.helpers.FileOutput):
            # Read the file content as bytes
            file_data = output.read()

            # Save the binary data to an image file
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'generated_image.png')
            with open(output_image_path, 'wb') as file:
                file.write(file_data)

            # Render the result page with the image URL
            return render_template('result.html', output_image='generated_image.png')
        elif isinstance(output, list) and len(output) > 0:
            # The output is a URL to the generated image
            output_url = output[0]
            # Download the image
            output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], 'generated_image.png')
            # Use requests to download the image
            import requests
            response = requests.get(output_url)
            with open(output_image_path, 'wb') as f:
                f.write(response.content)
            # Render the result page with the image URL
            return render_template('result.html', output_image='generated_image.png')
        else:
            logging.error(f"Error: No output returned from model.")
            return jsonify({'error': 'No output generated from the model.'}), 500

    except Exception as e:
        # Log the exact error
        error_message = f"Internal Server Error: {str(e)}"
        logging.exception(error_message)  # This logs the full stack trace

        # Return the error message as part of the response for easier debugging
        return jsonify({'error': error_message}), 500

@app.route('/static/outputs/<filename>')
def output_image(filename):
    """Serve the output image."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Ensure the directories exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])

    # Run the Flask app
    app.run(debug=True)
