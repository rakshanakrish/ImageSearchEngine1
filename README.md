# ImageSearchEngine1
 A search engine that uses cosine similarity algorithm for finding similar images
 # Image Similarity Search with DeepImageSearch (se.py in SearchEngine)

## Overview

This project implements an image similarity search using DeepImageSearch with a pre-trained ResNet model. The application allows users to upload an image and find similar images from a specified dataset. The results include precision evaluation based on ground truth similar images.

## Features

- Image upload via a Tkinter GUI.
- Image resizing for efficient processing.
- Indexing of images using a pre-trained ResNet model.
- Retrieval of similar images based on the uploaded image.
- Precision score calculation, confusion matrix, and classification report for accuracy evaluation.
- Visualization of retrieved images and the confusion matrix using Matplotlib.

## Requirements

Make sure you have the following packages installed:

- Python 3.x
- Tkinter (usually included with Python installations)
- Pillow
- DeepImageSearch
- Matplotlib
- NumPy
- scikit-learn

You can install the required packages using pip

# Image Similarity Search with MobileNetV2 (se2.py)

## Overview

This project implements an image similarity search application that utilizes the MobileNetV2 model for feature extraction. The application allows users to upload a query image and find similar images from a dataset stored in an SQLite database. The results include the product titles, image URLs, and similarity scores based on cosine similarity.

## Features

- Asynchronous loading of images for efficiency.
- Feature extraction using the MobileNetV2 model pre-trained on ImageNet.
- Storage and retrieval of extracted features using Python's pickle module.
- SQLite database integration for managing image paths and product titles.
- Visualization of query images and similar images using Matplotlib.

## Requirements

Ensure you have the following packages installed:

- Python 3.x
- TensorFlow
- Pillow
- NumPy
- scikit-learn
- Matplotlib
- SQLite (Python's built-in library)

You can install the required packages using pip:
Getting Started
Clone or download this repository.
Prepare your SQLite database (DataB.db) with a table named DataB containing columns product_title and image_url.
Make sure the images referenced in the database are accessible.
Run the script:
bash
Copy code
python your_script_name.py
A file dialog will prompt you to select a query image from your local system.
The program will extract features from the uploaded image and find similar images based on cosine similarity.
Similar images will be displayed along with their titles and similarity scores.
Code Explanation
Feature Extraction: The MobileNetV2 model extracts feature vectors from images for similarity comparisons.
Asynchronous Loading: Images are loaded and processed asynchronously to improve performance.
Cosine Similarity: The application computes cosine similarity between the feature vectors of the query image and those in the dataset to identify similar images.
Display Functionality: Matplotlib is used to visualize the query image alongside the similar images.
Output
The application displays the query image along with the top N similar images found in the dataset. The console output includes the product titles, URLs, and similarity scores for the similar images.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
MobileNetV2 for the convolutional neural network architecture.
TensorFlow for the deep learning framework.
Pillow for image handling.
scikit-learn for cosine similarity computation.


### Note:
- Make sure to replace `"your_script_name.py"` with the actual filename of your script.
- You can add sections like "Contributing," "Contact," or any other relevant information as needed.



