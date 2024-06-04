# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 01:48:54 2023

@author: user
"""



#%reset
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import load_model

import cv2
import glob
import time
import gc

#from skimage import exposure
from joblib import Parallel, delayed
#import pandas as pd
from scipy import stats

#from tensorflow.keras.callbacks import LearningRateScheduler
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from PIL import Image
#import matplotlib
#import matplotlib.pyplot as plt



start_time = time.time()

path = r"D:/UC".replace('\\', '/')

INPUT_DIR = path +  '/20240522/NC'
OUTPUT_DIR = INPUT_DIR +  '/crop_4_33'

# Change the current working directory to the input directory
os.chdir(INPUT_DIR)

REQUIRED_DIRS = ['exclude', 'single', 'multi']

for dir_name in REQUIRED_DIRS:
    os.makedirs(os.path.join(OUTPUT_DIR, dir_name), exist_ok=True)

print(OUTPUT_DIR)


def add_border(image):
    """Adds a border to the image."""
    top = bottom = left = right = 12

    reshaped = image.reshape(-1, image.shape[-1])
    red_mode, green_mode, blue_mode = [stats.mode(reshaped[:, i])[0] for i in range(3)]
    border_color = (int(red_mode), int(green_mode), int(blue_mode))

    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_color)

def perform_morphological_operations(image):
    """Performs a series of morphological operations on the image."""
    ball = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, ball)
    subtracted = cv2.subtract(image, background)

    gray = cv2.cvtColor(subtracted, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_value = int(otsu_thresh[0] * 0.8)
    thresh = cv2.threshold(blur, thresh_value, 255, cv2.THRESH_BINARY)[1]

    kernel_size = 7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image_closed = thresh
    #前面開運算這邊閉運算把縫隙填回去
    #但現在圖片都夠亮所以不太需要了
    #for _ in range(1):
    #    image_closed = cv2.morphologyEx(image_closed, cv2.MORPH_CLOSE, kernel)

    filtered_image = cv2.medianBlur(image_closed, 5)

    return filtered_image


def process_image(file):
    
    image = cv2.imread(os.path.join(INPUT_DIR, file))
    filename = file.rsplit('.', 1)[0]

    # 調整亮度和對比度
   # brightness = 1.1# 亮度增量
    #contrast = 1.0 # 對比度增量
    #image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    img = add_border(image)
    img_expand = img.copy()

    processed_image = perform_morphological_operations(img)

    ROI_number = 0
    contours = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            ROI = img_expand[y-11:y+h+11, x-11:x+w+11]
           
            # Check if the ROI is on the border
            is_on_border = y-12 <= 0 or y+h+12 >= img_expand.shape[0] or x-12 <=0 or x+w+12>= img_expand.shape[1]

            # Calculate the area of the contour
            area_bounding  = w*h
            
            # Calculate clarity using variance of Laplacian
            if ROI.size > 0:
                gray_roi = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                clarity_roi = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            else:
                clarity_roi = 0

            if is_on_border:
                # If the ROI is on the border, write the image to the "exclude" directory
                if ROI.size > 0:
                    cv2.imwrite(os.path.join(OUTPUT_DIR, 'exclude', '{}_{}.jpg'.format(filename , ROI_number)), ROI)
                else:
                    print(f"Empty ROI for file {filename} number {ROI_number}")
            elif area_bounding <= 500 or clarity_roi <= 60:
                # If so, write the image to the "exclude" directory
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'exclude', '{}_{}.jpg'.format(filename , ROI_number)), ROI)
            elif area_bounding < 1000:
                # If so, write the image to the "single" directory
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'single', '{}_{}.jpg'.format(filename , ROI_number)), ROI)
            elif area_bounding >= 1000:
                # Otherwise, write the image to the "multi" directory
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'multi', '{}_{}.jpg'.format(filename , ROI_number)), ROI)
            else:
                # If so, write the image to the "single" directory
                cv2.imwrite(os.path.join(OUTPUT_DIR, 'exclude', '{}_{}.jpg'.format(filename , ROI_number)), ROI)

            ROI_number += 1

    return ROI_number






current_directory = os.getcwd()
print(f"當前工作目錄: {current_directory}")

files = glob.glob("*.tiff")

print(f"找到的 .tiff 檔案: {files}")

ROI_numbers = Parallel(n_jobs=2)(delayed(process_image)(file) for file in files)



total_ROIs = sum(ROI_numbers)
print(f"Total ROIs processed: {total_ROIs}")


#Classification_single

# Define main directory
#path = '/content/drive/MyDrive/AI'
#OUTPUT_DIR = path + '/experiment_Data/' + '20230713/Mg5_Pl5_pos/crop'
single_model_path = path + '/model/singleMgPoly_modal_v01_20_tf10'

if os.path.exists(single_model_path):
    print("Loading existing model...")
    model = load_model(single_model_path)

else:
    print("The model not existed...")

# Define paths for the test directories
single_test_path = OUTPUT_DIR + '/single'
mag_folder = single_test_path + '/mg'
poly_folder = single_test_path + '/poly'

# Create directories if they don't exist
os.makedirs(mag_folder, exist_ok=True)
os.makedirs(poly_folder, exist_ok=True)

def process_path(file_path):
    # Convert file_path to Python string using tf.py_function
    file_path = tf.py_function(lambda x: x.decode('utf-8'), [file_path], tf.string)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [20, 20])
    img = img / 255.0  # normalize to [0,1] range
    return img, file_path

def process_images(file_paths_batch):
    predicted_classes = []
    for file_path in file_paths_batch.numpy():
        prediction = tf.py_function(process_file_path, [file_path], tf.int64)
        predicted_classes.append(prediction)
    return tf.stack(predicted_classes)


def process_file_path(file_path):
    img, _ = process_path(file_path)  # Call process_path to get the preprocessed image
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = tf.argmax(prediction[0])
    return predicted_class

file_paths = [os.path.join(single_test_path, filename) for filename in os.listdir(single_test_path) if filename.endswith(".jpg")]

# iterate through each batch of images and their corresponding file paths,
# make predictions for each image.

# Rest of the code remains the same as in your previous code

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64

# 將 test 資料夾底下的檔案路徑加入到 file_paths 列表
test_files = os.listdir(single_test_path)
file_paths = [os.path.join(single_test_path, filename) for filename in test_files if filename.endswith(".jpg")]

# 使用 tf.data.Dataset 來建立資料集
file_paths_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_paths, dtype=tf.string))

# 定義 process_path 函式，用來讀取和預處理影像檔案
def process_path(file_path):
    # Read the image from file_path
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [20, 20])  # Resize to match model input shape
    img = img / 255.0  # normalize to [0,1] range
    return img



# 將 process_path 函式應用到資料集中，並指定 num_parallel_calls 為 AUTOTUNE
images_dataset = file_paths_dataset.map(process_path, num_parallel_calls=AUTOTUNE)


#initialization
MB_number = 0
PB_number = 0


def predict_and_move_files(image):
    global PB_number, MB_number  # Declare PB_number as a global variable
    # Make prediction
    prediction = model(tf.expand_dims(image, axis=0))
    predicted_class = tf.argmax(prediction, axis=-1).numpy()[0]

    # Move the file based on the predicted class
    if predicted_class > 0.5:
        shutil.move(image_path, os.path.join(poly_folder, os.path.basename(image_path)))
        MB_number += 1
        #print(f"Predicted class for {image_path}: {predicted_class}")

    else:
        shutil.move(image_path, os.path.join(mag_folder, os.path.basename(image_path)))
        #print(f"Predicted class for {image_path}: {predicted_class}")
        PB_number += 1

# Iterate through the images and apply prediction and moving logic
for image_path, image in zip(file_paths, images_dataset):
    predict_and_move_files(image)


# Print final counts
print("Final counts:")
print(f"Moved to poly folder (PB_number): {MB_number}")
print(f"Moved to mag folder (MB_number): {PB_number}")




#Classification_multi
# Define main directory
#path = '/content/drive/MyDrive/AI'
#OUTPUT_DIR = path + '/experiment_Data/' + '20230713/Mg5_Pl5_pos/crop'
multi_test_path = OUTPUT_DIR + '/multi'

#test_path = path + '/Mg5_Pl5_pos/multi'

#single_model_path = path + 'model/singleMgPoly_modal_v01_20'
multi_model_path = path + '/model/multi_MgPoly_modal_ResNet_6v04_tf_2_10'



if os.path.exists(multi_model_path):
    print("Loading existing model...")
    model = load_model(multi_model_path)

else:
    print("The model not existed...")

# Define paths for the test directories
test_clump_folder = multi_test_path + '/clump'
test_multi_mg_2_folder = multi_test_path + '/multi_mg_2'
test_multi_mg_3_folder = multi_test_path + '/multi_mg_3'
test_multi_mix_folder = multi_test_path + '/multi_mix'
test_multi_poly_folder = multi_test_path + '/multi_poly'
test_single_MB_folder = multi_test_path + '/single_MB'

# Create directories if they don't exist
os.makedirs(test_clump_folder, exist_ok=True)
os.makedirs(test_multi_mg_2_folder, exist_ok=True)
os.makedirs(test_multi_mg_3_folder, exist_ok=True)
os.makedirs(test_multi_mix_folder, exist_ok=True)
os.makedirs(test_multi_poly_folder, exist_ok=True)
os.makedirs(test_single_MB_folder, exist_ok=True)

def process_path(file_path):
    # Convert file_path to Python string using tf.py_function
    file_path = tf.py_function(lambda x: x.decode('utf-8'), [file_path], tf.string)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [80, 80])
    img = img / 255.0  # normalize to [0,1] range
    return img, file_path

def process_images(file_paths_batch):
    predicted_classes = []
    for file_path in file_paths_batch.numpy():
        prediction = tf.py_function(process_file_path, [file_path], tf.int64)
        predicted_classes.append(prediction)
    return tf.stack(predicted_classes)


def process_file_path(file_path):
    img, _ = process_path(file_path)  # Call process_path to get the preprocessed image
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = tf.argmax(prediction[0])
    return predicted_class

file_paths = [os.path.join(multi_test_path, filename) for filename in os.listdir(multi_test_path) if filename.endswith(".jpg")]

# iterate through each batch of images and their corresponding file paths,
# make predictions for each image.

# Rest of the code remains the same as in your previous code

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64

# 將 test 資料夾底下的檔案路徑加入到 file_paths 列表
test_files = os.listdir(multi_test_path)
file_paths = [os.path.join(multi_test_path, filename) for filename in test_files if filename.endswith(".jpg")]

# 使用 tf.data.Dataset 來建立資料集
file_paths_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_paths, dtype=tf.string))

# 定義 process_path 函式，用來讀取和預處理影像檔案
def process_path(file_path):
    # Read the image from file_path
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [80, 80])
    img = img / 255.0  # normalize to [0,1] range
    return img


# 將 process_path 函式應用到資料集中，並指定 num_parallel_calls 為 AUTOTUNE
images_dataset = file_paths_dataset.map(process_path, num_parallel_calls=AUTOTUNE)


# Assume process_images function returns class probabilities for each image in the batch
def process_images(batch):
    # Your prediction code here, return predicted classes as string tensor
    # For example, you might have:
    # predicted_classes = tf.argmax(model.predict(batch), axis=1)
    # Convert predicted_classes to string tensor
    predicted_classes = tf.argmax(model.predict(batch), axis=1)
    predicted_classes = tf.strings.as_string(predicted_classes)
    return predicted_classes

for batch, file_paths_batch in zip(images_dataset.batch(batch_size), file_paths_dataset.batch(batch_size)):
    probabilities = model.predict(batch)

    for prob, file_path in zip(probabilities, file_paths_batch):
        predicted_label = tf.argmax(prob).numpy()

        file_path = file_path.numpy().decode('utf-8')  # Get the corresponding file_path for this image

        if predicted_label == 0:
            try:
                shutil.move(file_path, os.path.join(test_multi_poly_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 1:
            try:
                shutil.move(file_path, os.path.join(test_single_MB_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 2:
            try:
                shutil.move(file_path, os.path.join(test_multi_mg_2_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 3:
            try:
                shutil.move(file_path, os.path.join(test_multi_mg_3_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 4:
            try:
                shutil.move(file_path, os.path.join(test_clump_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 5:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")


#Classification_PCNC
# Define main directory
#path = '/content/drive/MyDrive/AI'
#OUTPUT_DIR = path + '/experiment_Data/' + '20230713/Mg5_Pl5_pos/crop'
multi_mix_test_path = multi_test_path + '/multi_mix'

#test_path = path + '/Mg5_Pl5_pos/multi'

#single_model_path = path + 'model/singleMgPoly_modal_v01_20'
multi_mix_model_path = path + '/model/multi_mix_modal_ResNet_7v33_tf_2_10'



if os.path.exists(multi_mix_model_path):
    print("Loading existing model...")
    model = load_model(multi_mix_model_path)

else:
    print("The model not existed...")

# Define paths for the test directories
test_multi_mix_b0_1_folder = multi_mix_test_path + '/b0_1'
test_multi_mix_b0_2_folder = multi_mix_test_path + '/b0_2'
test_multi_mix_b0_3_folder = multi_mix_test_path + '/b0_3'
test_multi_mix_b1_folder = multi_mix_test_path + '/b1'
test_multi_mix_b2_folder = multi_mix_test_path + '/b2'
test_multi_mix_b3_folder = multi_mix_test_path + '/b3'
test_multi_mix_clump_folder = multi_mix_test_path + '/clump'


# Create directories if they don't exist
os.makedirs(test_multi_mix_b0_1_folder, exist_ok=True)
os.makedirs(test_multi_mix_b0_2_folder, exist_ok=True)
os.makedirs(test_multi_mix_b0_3_folder, exist_ok=True)
os.makedirs(test_multi_mix_b1_folder, exist_ok=True)
os.makedirs(test_multi_mix_b2_folder, exist_ok=True)
os.makedirs(test_multi_mix_b3_folder, exist_ok=True)
os.makedirs(test_multi_mix_clump_folder, exist_ok=True)

def process_path(file_path):
    # Convert file_path to Python string using tf.py_function
    file_path = tf.py_function(lambda x: x.decode('utf-8'), [file_path], tf.string)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [80, 80])
    img = img / 255.0  # normalize to [0,1] range
    return img, file_path

def process_images(file_paths_batch):
    predicted_classes = []
    for file_path in file_paths_batch.numpy():
        prediction = tf.py_function(process_file_path, [file_path], tf.int64)
        predicted_classes.append(prediction)
    return tf.stack(predicted_classes)


def process_file_path(file_path):
    img, _ = process_path(file_path)  # Call process_path to get the preprocessed image
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = tf.argmax(prediction[0])
    return predicted_class

file_paths = [os.path.join(multi_mix_test_path, filename) for filename in os.listdir(multi_mix_test_path) if filename.endswith(".jpg")]

# iterate through each batch of images and their corresponding file paths,
# make predictions for each image.

# Rest of the code remains the same as in your previous code

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64

# 將 test 資料夾底下的檔案路徑加入到 file_paths 列表
test_files = os.listdir(multi_mix_test_path)
file_paths = [os.path.join(multi_mix_test_path, filename) for filename in test_files if filename.endswith(".jpg")]

# 使用 tf.data.Dataset 來建立資料集
file_paths_dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(file_paths, dtype=tf.string))

# 定義 process_path 函式，用來讀取和預處理影像檔案
def process_path(file_path):
    # Read the image from file_path
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [80, 80])
    img = img / 255.0  # normalize to [0,1] range
    return img


# 將 process_path 函式應用到資料集中，並指定 num_parallel_calls 為 AUTOTUNE
images_dataset = file_paths_dataset.map(process_path, num_parallel_calls=AUTOTUNE)


# Assume process_images function returns class probabilities for each image in the batch
def process_images(batch):
    # Your prediction code here, return predicted classes as string tensor
    # For example, you might have:
    # predicted_classes = tf.argmax(model.predict(batch), axis=1)
    # Convert predicted_classes to string tensor
    predicted_classes = tf.argmax(model.predict(batch), axis=1)
    predicted_classes = tf.strings.as_string(predicted_classes)
    return predicted_classes

for batch, file_paths_batch in zip(images_dataset.batch(batch_size), file_paths_dataset.batch(batch_size)):
    probabilities = model.predict(batch)

    for prob, file_path in zip(probabilities, file_paths_batch):
        predicted_label = tf.argmax(prob).numpy()

        file_path = file_path.numpy().decode('utf-8')  # Get the corresponding file_path for this image

        if predicted_label == 0:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_b0_1_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 1:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_b0_2_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 2:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_b0_3_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 3:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_b1_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 4:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_b2_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 5:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_b3_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        elif predicted_label == 6:
            try:
                shutil.move(file_path, os.path.join(test_multi_mix_clump_folder, os.path.basename(file_path)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")




end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")

#origin = r"D:/UC".replace('\\', '/')

#os.chdir(origin)



