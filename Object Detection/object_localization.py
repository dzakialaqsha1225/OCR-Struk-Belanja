import tensorflow as tf #tensorflow ver 2.18.0
import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

def object_localization(img: np.ndarray, model: tf.keras.Model)-> np.ndarray:
    """
    Performs object localization for receipts on an image using a given model.

    Args:
        img: A NumPy array representing the image.
        model: A TensorFlow Keras model for object localization.

    Returns:
        A NumPy array representing the localized objects.
    """
    # Input validation
    if not isinstance(img, np.ndarray):
        raise TypeError("Input image must be a NumPy array.")
    if not isinstance(model, tf.keras.Model):
        raise TypeError("Input model must be a TensorFlow Keras model.")

    # MAIN CODE #
    input_size = 244
    #open url and read the image
    img = cv.imdecode(img, cv.IMREAD_GRAYSCALE) #convert to grayscale
    
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
    
    # Object Localization for 
    predictions = model(result) #change

    predicted_box = abs(predictions[1][0] * max_size)
    
    predicted_box = tf.cast(predicted_box, tf.int32)

    predicted_box = abs(predictions[1][0])
    norm_max = max(predicted_box)
    norm_max = norm_max.numpy()

    predicted_box = (predicted_box / norm_max) * max_size
    
    predicted_box = tf.cast(predicted_box, tf.int32)
    predicted_box_value = predicted_box.numpy()
    
    x_min,y_min,x_max,y_max = predicted_box_value
    cropped_struk = img[y_min:y_max, x_min:x_max]
    cropped_struk = np.array(cropped_struk)
    
    return cropped_struk

