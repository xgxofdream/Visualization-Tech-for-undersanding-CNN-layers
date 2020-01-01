# -*- coding: utf-8 -*-
"""
Grad-CAM based on TF2.0

Created on Wed Jan  1 16:32:05 2020

@author: liujie
"""


import ast
import scipy   
import cv2  
import numpy as np   
import matplotlib.pyplot as plt
import tensorflow as tf

img_path = './input_img/dog.jpg'

#-----------------------------------------定义函数---------------------------------------------
# Get Images
def pretrained_path_to_tensor(img_path):
    # 读取一张待判断的图像
    img=cv2.imread(img_path)
    # 转化为数组 
    img=cv2.resize(img,(224, 224))
    print(img.shape)
    # 添加一个纬度，得到1*224*224*3的矩阵 
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    
    return img

# Get features
def get_ResNet():
    # define ResNet50 model
    model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=True)
    # get AMP layer weights
    all_amp_layer_weights = model.layers[-1].get_weights()[0]
    # extract wanted output
    inputs=model.input   
    outputs=[model.layers[-4].output, model.layers[-1].output] 
    ResNet_model = tf.keras.Model(inputs=inputs, outputs=outputs) 
    
    return ResNet_model, all_amp_layer_weights
# Get CAN    
def ResNet_CAM(img_path, model, all_amp_layer_weights):
    # 使用网络预测图像的类别
    last_conv_output, pred_vec = model.predict(pretrained_path_to_tensor(img_path))
    # change dimensions of last convolutional outpu tto 7 x 7 x 2048
    last_conv_output = np.squeeze(last_conv_output) 
    # 计算图片索引
    pred = np.argmax(pred_vec)
    # bilinear upsampling to resize each filtered image to size of original image 
    mat_for_mult = scipy.ndimage.zoom(last_conv_output, (32, 32, 1), order=1) # dim: 224 x 224 x 2048
    # get AMP layer weights
    amp_layer_weights = all_amp_layer_weights[:, pred] # dim: (2048,) 
    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224*224, 2048)), amp_layer_weights).reshape(224,224) # dim: 224 x 224
    # return class activation map
    return final_output, pred
# Plot CAN    
def plot_ResNet_CAM(img_path, ax, model, all_amp_layer_weights):
    # load image, convert BGR --> RGB, resize image to 224 x 224,
    im = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (224, 224))
    # plot image
    ax.imshow(im, alpha=0.5)
    # get class activation map
    CAM, pred = ResNet_CAM(img_path, model, all_amp_layer_weights)
    # plot class activation map
    ax.imshow(CAM, cmap='jet', alpha=0.5)
    # load the dictionary that identifies each ImageNet category to an index in the prediction vector
    with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())
    # obtain the predicted ImageNet category
    ax.set_title(imagenet_classes_dict[pred])    

#-----------------------------------------执行CAM visualization---------------------------------------------
ResNet_model, all_amp_layer_weights = get_ResNet()
fig, ax = plt.subplots()
CAM = plot_ResNet_CAM(img_path, ax, ResNet_model, all_amp_layer_weights)
plt.show()
