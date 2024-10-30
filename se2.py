import os
import numpy as np
import sqlite3
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from tkinter import filedialog
import tkinter as tk
import pickle
import asyncio
import aiofiles

# Load MobileNetV2 model pre-trained on ImageNet, without the top layer (for feature extraction)
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Directory to save and load extracted features
FEATURES_DIR = "extracted_features"
os.makedirs(FEATURES_DIR, exist_ok=True)

# SQLite database path
DB_PATH = "DataB.db"

async def load_image(image_path):
    """Asynchronously load an image and convert it to RGB."""
    try:
        async with aiofiles.open(image_path, mode='rb') as f:
            image_data = await f.read()
        img = Image.open(io.BytesIO(image_data))
        if img.mode in ("P", "RGBA", "L"):
            img = img.convert("RGB")  # Convert to RGB for consistency
        return img
    except (UnidentifiedImageError, FileNotFoundError) as e:
        print(f"Error loading {image_path}: {e}. Skipping file.")
        return None

async def extract_features(image_path):
    """Extract features from an image asynchronously using MobileNetV2."""
    if image_path is None:
        print(f"Invalid image path: {image_path}. Skipping file.")
        return None
    
    feature_file = os.path.join(FEATURES_DIR, os.path.basename(image_path) + ".pkl")

    # If features are cached, load them from disk
    if os.path.exists(feature_file):
        with open(feature_file, 'rb') as f:
            return pickle.load(f)

    img = await load_image(image_path)
    if img is None:
        return None

    # Resize and preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get feature vector from MobileNetV2
    features = model.predict(img_array).flatten()

    # Save the extracted features to disk for future use
    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)

    return features

async def load_image_features_from_db(db_path):
    """Load image paths and extract features from the database asynchronously."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT product_title, image_url FROM DataB")
    rows = cursor.fetchall()
    conn.close()

    image_features = []
    product_titles = []
    image_urls = []
    
    # Create a list of tasks for image feature extraction
    tasks = []
    for product_title, image_url in rows:
        tasks.append(extract_features(image_url))

    # Wait for all tasks to complete
    features = await asyncio.gather(*tasks)

    for idx, feature in enumerate(features):
        if feature is not None:
            image_features.append(feature)
            product_titles.append(rows[idx][0])
            image_urls.append(rows[idx][1])

    return np.array(image_features), product_titles, image_urls

def get_similar_images(query_features, image_features, product_titles, image_urls, top_n=5):
    """Find the top N similar images based on cosine similarity."""
    similarities = cosine_similarity(query_features.reshape(1, -1), image_features).flatten()
    similar_indices = np.argsort(similarities)[-top_n:][::-1]  # Sort and take the top N indices
    
    similar_products = []
    for index in similar_indices:
        similar_products.append({
            'title': product_titles[index],
            'image_url': image_urls[index],
            'score': similarities[index]
        })
    
    return similar_products

def display_similar_images(query_image_path, similar_products):
    """Display the query image and similar images using matplotlib."""
    plt.figure(figsize=(10, 5))
    
    # Display query image
    query_image = Image.open(query_image_path)
    plt.subplot(1, len(similar_products) + 1, 1)
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis('off')

    # Display similar images
    for i, product in enumerate(similar_products):
        similar_image = Image.open(product['image_url'])
        plt.subplot(1, len(similar_products) + 1, i + 2)
        plt.imshow(similar_image)
        plt.title(f"{product['title']}\nScore: {product['score']:.4f}")
        plt.axis('off')

    plt.show()

def main():
    # Step 1: Load image dataset and extract features and labels
    loop = asyncio.get_event_loop()
    image_features, product_titles, image_urls = loop.run_until_complete(load_image_features_from_db(DB_PATH))

    # Step 2: Let the user upload an image using tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    query_image_path = filedialog.askopenfilename(
        title="Select a Query Image", 
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not query_image_path:
        print("No query image selected.")
        return

    # Step 3: Extract features for the query image
    query_image_features = loop.run_until_complete(extract_features(query_image_path))

    # Step 4: Find similar images based on cosine similarity
    top_n = 5
    similar_products = get_similar_images(
        query_image_features, 
        image_features, 
        product_titles, 
        image_urls, 
        top_n=top_n
    )

    # Step 5: Display the query image and similar images
    display_similar_images(query_image_path, similar_products)

    # Step 6: Print the result in the desired output format
    print("Similar products with scores:")
    for product in similar_products:
        print(f"Product: {product['title']}, URL: {product['image_url']}, Score: {product['score']}")

if __name__ == "__main__":
    main()