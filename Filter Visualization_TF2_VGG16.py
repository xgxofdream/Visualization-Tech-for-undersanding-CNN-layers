# -*- coding: utf-8 -*-
"""
https://www.sicara.ai/blog/2019-08-28-interpretability-deep-learning-tensorflow

Seeing what is coming out of a layer is great, 
but what if we could understand what makes a kernel activate?

Created on Wed Jan  1 06:42:57 2020

@author: liujie
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Layer name to inspect
layer_name = 'block2_conv1'

epochs = 40
step_size = 1
filter_index = 0

# Create a connection between the input and the target layer
model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()



def deprocess_image(x):
    x = tf.reduce_mean(x)
    x -= x
    x /= (tf.sqrt(x) + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x,0,1)

    x *= 255
    x = np.clip(x,0,255).astype('uint8')
    return x

def generate_pattern(layer_name):
    submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
    # Initiate random noise
    input_img_data = np.random.random((1,150,150,3)) * 20 + 128.
    # Cast random noise from np.float64 to tf.float32 Variable
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    
    # Iterate gradient ascents
    for _ in range(epochs):
        with tf.GradientTape() as tape:
            outputs = submodel(input_img_data)
            loss_value = tf.reduce_mean(outputs[:, :, :, filter_index])
        grads = tape.gradient(loss_value, input_img_data)
        normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_img_data.assign_add(normalized_grads * step_size)
    
    img = input_img_data[0]
    return deprocess_image(img)


size = 64 
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin,3))

for i in range(8):
    for j in range(8):
        filter_img = generate_pattern(layer_name)
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start:horizontal_end,vertical_start:vertical_end,:] = filter_img

results = np.clip(results,0,255).astype('uint8')

plt.figure(figsize=(20,20))
plt.imshow(results)
plt.show()
