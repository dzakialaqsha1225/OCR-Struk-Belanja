import tensorflow as tf #tensorflow ver 2.18.0
import numpy as np
import cv2 as cv
import urllib.request

from matplotlib import pyplot as plt

#Returns YOLO bounding box array

def object_detection(img_url,
model, model_1, model_2, model_3):
    input_size = 244
    #open url and read the image
    req = urllib.request.urlopen(img_url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_GRAYSCALE) #convert to grayscale

    #preprocess the image
    height, width = img.shape 
    max_size = max(height, width)
    r = max_size / input_size
    new_width = int(width / r)
    new_height = int(height / r)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((input_size, input_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    #second preprocessing
    X = [] 
    new_image = new_image.astype(float) / 255.

    X.append(new_image)

    #third preprocessing
    X = np.array(X)
    
    X = np.expand_dims(X, axis=3)

    X = tf.convert_to_tensor(X, dtype=tf.float32)

    result = X
    #fourth preprocessing
    predictions = model(result) #change
    predictions1 = model_1(result) #change
    predictions2 = model_2(result) #change
    predictions3 = model_3(result) #change

    predicted_box = abs(predictions[1][0] * max_size)
    predicted_box1 = abs(predictions1[1][0] * max_size)
    predicted_box2 = abs(predictions2[1][0] * max_size)
    predicted_box3 = abs(predictions3[1][0] * max_size)
    
    predicted_box = tf.cast(predicted_box, tf.int32)
    predicted_box1 = tf.cast(predicted_box1, tf.int32)
    predicted_box2 = tf.cast(predicted_box2, tf.int32)
    predicted_box3 = tf.cast(predicted_box3, tf.int32)
    
    temp_color_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    
    return(predicted_box, predicted_box1, predicted_box2, predicted_box3)

