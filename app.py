import os
import io
import numpy as np
import sqlite3
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, UnidentifiedImageError
import asyncio
import aiofiles
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Add configuration for file upload
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Database configuration
DB_PATH = "DataB.db"

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def load_image(image_data):
    """Asynchronously load an image and convert it to RGB."""
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.mode in ("P", "RGBA", "L"):
            img = img.convert("RGB")
        return img
    except (UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Error loading image: {e}")
        return None

async def extract_features(image_path_or_data):
    """Extract features from an image asynchronously using MobileNetV2."""
    image_data = None

    if isinstance(image_path_or_data, bytes):
        image_data = image_path_or_data
    else:
        if os.path.isfile(image_path_or_data):
            try:
                async with aiofiles.open(image_path_or_data, 'rb') as f:
                    image_data = await f.read()
            except FileNotFoundError:
                print(f"File not found: {image_path_or_data}")
                return None
        else:
            try:
                response = requests.get(image_path_or_data)
                response.raise_for_status()
                image_data = response.content
            except requests.RequestException as e:
                print(f"Error fetching image from URL: {e}")
                return None

    img = await load_image(image_data)
    if img is None:
        return None

    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array).flatten()
    return features

async def load_image_features_from_db(db_path):
    """Load image features from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT product_title, image_url FROM DataB")
    rows = cursor.fetchall()
    conn.close()

    image_features = []
    product_titles = []
    image_urls = []
    
    tasks = []
    for product_title, image_url in rows:
        tasks.append(extract_features(image_url))

    features = await asyncio.gather(*tasks)

    for idx, feature in enumerate(features):
        if feature is not None:
            image_features.append(feature)
            product_titles.append(rows[idx][0])
            image_urls.append(rows[idx][1])

    return np.array(image_features), product_titles, image_urls

def get_similar_images(query_features, image_features, product_titles, image_urls, top_n=5):
    """Find similar images based on cosine similarity"""
    similarities = cosine_similarity(query_features.reshape(1, -1), image_features).flatten()
    similar_indices = np.argsort(similarities)[-top_n:][::-1]
    
    similar_products = []
    for index in similar_indices:
        similar_products.append({
            'similarity_score': float(similarities[index]),
            'title': product_titles[index],
            'image_url': image_urls[index]
        })
    
    return similar_products
@app.route('/similar_images', methods=['POST'])
async def find_similar_images():
    """Endpoint to find similar images"""
    # Debug information
    print("Request Files:", request.files)
    print("Request Form:", request.form)
    print("Content Type:", request.content_type)

    # Check if any file was uploaded
    if 'image' not in request.files:
        return jsonify({
            "error": "No image file uploaded",
            "debug_info": {
                "received_keys": list(request.files.keys()),
                "content_type": request.content_type,
                "expected_key": "image"
            }
        }), 400
    
    file = request.files['image']
    
    # Check if a file was selected
    if file.filename == '':
        return jsonify({
            "error": "No selected file",
            "help": "Please select a file before uploading"
        }), 400

    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            "error": "Invalid file type",
            "allowed_types": list(ALLOWED_EXTENSIONS)
        }), 400

    try:
        # Read and process the image
        image_data = file.read()
        
        # Extract features for the query image
        query_features = await extract_features(image_data)
        if query_features is None:
            return jsonify({"error": "Failed to process image"}), 400

        # Load dataset features
        image_features, product_titles, image_urls = await load_image_features_from_db(DB_PATH)
        
        # Find similar images
        similar_products = get_similar_images(
            query_features, 
            image_features, 
            product_titles, 
            image_urls,
            top_n=5
        )

        return jsonify({
            "status": "success",
            "results": similar_products
        })

    except Exception as e:
        return jsonify({
            "error": "An error occurred while processing the request",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database": os.path.exists(DB_PATH)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)