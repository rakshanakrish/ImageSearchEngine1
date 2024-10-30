import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from DeepImageSearch import Load_Data, Search_Setup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score

# Set environment variable to avoid OpenMP runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function to resize image for faster processing
def resize_image(image_path, size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(size)
    return img

# Create a Tkinter window to prompt the user to upload an image
root = tk.Tk()
root.withdraw()  # Hide the main window

# Open a file dialog to allow the user to select an image
user_image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
)

# Resize the user's uploaded image for faster feature extraction
user_image = resize_image(user_image_path)

# Load images from a folder (your existing image dataset)
image_list = Load_Data().from_folder(folder_list=['SearchEngine/images'])

# Use ResNet model for DeepImageSearch
st = Search_Setup(image_list=image_list, model_name='resnet50', pretrained=True, image_count=100)
print("Total Image Count:", len(image_list))
print("Samples:", image_list[:10])

# Run the image indexer
st.run_index()
metadata = st.get_image_metadata_file()

# Display user image path for reference
print(f"User Image Path: {user_image_path}")

# Add a range of images to the index (if necessary)
st.add_images_to_index(image_list[1001:1010])

# Search for similar images based on the user's uploaded image
similar_images = st.get_similar_images(image_path=user_image_path, number_of_images=5)

# Debugging step to see what is returned
print("Similar Images Returned:", similar_images)

# Create a folder to save the similar images if it doesn't exist
output_folder = 'SearchEngine/output_similar_images'
os.makedirs(output_folder, exist_ok=True)

# Labels for ground truth (this is for accuracy calculation, adjust according to your dataset)
true_similar_images = {
    'your_uploaded_image.jpg': ['2700.jpg', '2477.jpg', '2718.jpg'],  # Example, modify to match your dataset
    # Add more entries here for other images
}

# Get the basename of the uploaded image
user_image_key = os.path.basename(user_image_path)  
print(f"User Image Key (used to fetch true similar images): {user_image_key}")

# Now retrieve the true similar images for the user's uploaded image
true_similar_images_list = true_similar_images.get(user_image_key, [])
print("True Similar Images:", true_similar_images_list)

# List to store retrieved similar images
retrieved_images = []

# Save and display similar images
for idx, image_path in similar_images.items():
    print(f"Attempting to open image: {image_path}")
    try:
        img = Image.open(image_path)

        # Save the image to the output folder
        img_output_path = os.path.join(output_folder, f'similar_image_{idx}.jpg')
        img.save(img_output_path)
        print(f"Saved image to: {img_output_path}")
        
        # Append to retrieved images for accuracy score calculation
        retrieved_images.append(os.path.basename(image_path))

        # Display the image using matplotlib
        plt.imshow(img)
        plt.title(f'Similar Image {idx}')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()  # Display the image

    except FileNotFoundError:
        print(f"File does not exist: {image_path}")

print("Retrieved Images:", retrieved_images)

# Calculating accuracy score based on retrieved similar images and ground truth
y_true = [1 if img in true_similar_images_list else 0 for img in retrieved_images]
y_pred = [1] * len(retrieved_images)  # Since all retrieved images are considered as predicted positives

# Precision Score Calculation
accuracy = precision_score(y_true, y_pred, zero_division=0)
print(f"Accuracy (Precision) of Similar Image Retrieval: {accuracy:.2f}")

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_true, y_pred, zero_division=0)
print("\nClassification Report:")
print(class_report)

# Plotting the Confusion Matrix
plt.figure(figsize=(6, 6))
plt.matshow(conf_matrix, cmap='Blues', fignum=1)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()



