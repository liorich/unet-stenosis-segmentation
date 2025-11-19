import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy
import matplotlib.pyplot as plt
from Models import models as model
import pickle
from time import sleep

# Tensorflow Configurations
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
policy = Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Single-slot array for model prediction
X = np.zeros((1, 512, 512, 1))

def calculate_white_pixel_ratio(mask_image, size: int):
    total_white_pixels = np.sum(mask_image != 0)
    total_pixels = size ** 2
    white_pixel_ratio = total_white_pixels / total_pixels
    return white_pixel_ratio

def process_images(input_dir, output_dir, model_path, real_size=-1):
    # Get a list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    with open('white_pixel_ratios.txt', 'w') as file:
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_filename = os.path.splitext(filename)[0] + "_segmented" + os.path.splitext(filename)[1]
            output_path = os.path.join(output_dir, output_filename)
            img = imread(input_path)
            img = resize(img, (512, 512, 1), mode='constant', preserve_range=True)/ 255
            X[0] = img

            model = tf.keras.models.load_model(model_path)

            # Segment the image and calculate ratios
            processed_img = model(X).numpy()[0]
            white_pixel_ratio = calculate_white_pixel_ratio(processed_img, 512)
            if real_size != -1:
                file.write(f"Filename: {output_filename}, Ratio: {white_pixel_ratio:.4f}, "
                           f"Block Area: {white_pixel_ratio * real_size**2:.4f} cm^2\n")
            else:
                file.write(f"Filename: {output_filename}, Ratio: {white_pixel_ratio:.4f}\n")

            # Save the processed image in the output directory
            pred = (processed_img > 0.003).astype(np.bool_)
            pred = np.where(pred, 255, 0).astype(np.uint8)
            imsave(output_path + output_filename, pred)

def browse_input_directory():
    input_dir = filedialog.askdirectory()
    input_entry.delete(0, tk.END)
    input_entry.insert(0, input_dir)

def browse_output_directory():
    output_dir = filedialog.askdirectory()
    output_entry.delete(0, tk.END)
    output_entry.insert(0, output_dir)

def browse_model_path():
    model_path = filedialog.askopenfilename()
    model_entry.delete(0, tk.END)
    model_entry.insert(0, model_path)

def process_images_button():
    input_dir = input_entry.get()
    output_dir = output_entry.get()
    model_path = model_entry.get()
    real_size = float(real_size_entry.get()) if real_size_entry.get() else -1
    process_images(input_dir, output_dir, model_path, real_size)

# Create the main GUI window
root = tk.Tk()
root.title("Image Processing GUI")

# Input directory widgets
input_label = tk.Label(root, text="Input Directory:")
input_label.pack()
input_entry = tk.Entry(root, width=50)
input_entry.pack()
input_button = tk.Button(root, text="Browse", command=browse_input_directory)
input_button.pack()

# Output directory widgets
output_label = tk.Label(root, text="Output Directory:")
output_label.pack()
output_entry = tk.Entry(root, width=50)
output_entry.pack()
output_button = tk.Button(root, text="Browse", command=browse_output_directory)
output_button.pack()

# Model path widget
model_label = tk.Label(root, text="Model Path:")
model_label.pack()
model_entry = tk.Entry(root, width=50)
model_entry.pack()
model_button = tk.Button(root, text="Browse", command=browse_model_path)
model_button.pack()

# Real image size widget
real_size_label = tk.Label(root, text="Real Image Size [cm] (Default: 14cm, Optional)")
real_size_label.pack()
real_size_entry = tk.Entry(root, width=20)
real_size_entry.pack()

# Process button
process_button = tk.Button(root, text="Process Images", command=process_images_button)
process_button.pack()

# Instructions window
instructions = """
INSTRUCTIONS:
1. Select the Input Directory containing the images to be processed.
2. Select the Output Directory where the processed images will be saved.
3. Browse and select the path of the saved model of the trained neural network.
- The model path is the folder which contains: assets/, variables/, keras_metadata.pb, saved_model.pb
4. Optionally, specify the desired Real Image Size in centimeters. Leave empty to use the default size of 14cm.
6. Click on 'Process Images' to start the image processing.
"""
instructions_label = tk.Label(root, text=instructions, justify='left')
instructions_label.pack()

# Run the GUI main loop
root.mainloop()
