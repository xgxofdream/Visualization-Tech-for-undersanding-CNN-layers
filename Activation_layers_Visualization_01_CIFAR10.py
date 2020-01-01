# -*- coding: utf-8 -*-
"""
#------------可视化中间特征层 visualize activation layers------------------------------------
# 理论参考
# https://www.zhihu.com/search?type=content&q=%E5%8F%AF%E8%A7%86%E5%8C%96CNN%E9%9A%90%E8%97%8F%E5%B1%82

# 代码参考
# https://github.com/princekrroshan01/MNIST-dataset-training-using-cnn-visualization/blob/master/MNIST_CNN.ipynb

Created on Wed Jan  1 06:39:14 2020

@author: liuji
"""


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import datasets, layers, models

#-----------------------------模型超参数------------------------------
num_epochs = 5
batch_size = 500
learning_rate = 0.01

#-----------------------------函数设定---------------------
def gen_model_for_Visualization(cnt):
  layer_outputs = [layer.output for layer in model.layers[:cnt]] 
  activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
  return activation_model

def visualize(id):
  layer_names = []
  for layer in model.layers[:cnt]:
      layer_names.append(layer.name)       

  images_per_row = 16

  for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
      n_features = layer_activation.shape[-1] 
      print("number of feature maps",n_features)  # Number of features in the feature map
      size = layer_activation.shape[1] 
      print("size",size) #The feature map has shape (1, size, size, n_features).
      n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
      display_grid = np.zeros((size * n_cols, images_per_row * size))
      for col in range(n_cols): # Tiles each filter into a big horizontal grid
          for row in range(images_per_row):
              channel_image = layer_activation[id,
                                              :, :, 
                                              col * images_per_row + row]
              channel_image -= channel_image.mean() 
              channel_image /= channel_image.std()
              channel_image *= 64
              channel_image += 128
              channel_image = np.clip(channel_image, 0, 255).astype('uint8')
              display_grid[col * size : (col + 1) * size, # Displays the grid
                          row * size : (row + 1) * size] = channel_image
      scale = 1. / size
      plt.figure(figsize=(scale * display_grid.shape[1],
                          scale * display_grid.shape[0]))
      plt.title(layer_name)
      plt.grid(False)
      plt.imshow(display_grid, aspect='auto', cmap='viridis')


#-----------------------------数据获取及预处理------------------------------
#tf.keras.datasets
class CIFARLoader():
    def __init__(self):
        cifar = tf.keras.datasets.cifar10
        (self.train_data, self.train_label), (self.test_data, self.test_label) = cifar.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = self.train_data.astype(np.float32) / 255.0      # [60000, 28, 28, 1]
        self.test_data = self.test_data.astype(np.float32) / 255.0        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

# 测试CIFARLoader()
data_loader = CIFARLoader()
print(data_loader.train_data.shape)
print(data_loader.train_label.shape)

#-----------------------------模型的构建------------------------------
#Keras Sequential API 模式建立模型

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

classes = model.predict_classes(data_loader.train_data, batch_size=10)
print("Predicted class is:",classes)

#---------------------------------Visualization---------------------
cnt=int(input("enter the number of layer you want to visualize"))
output=gen_model_for_Visualization(cnt)
activations =output.predict(data_loader.train_data)

print(activations[0].shape)
     
id=int(input("enter the index of input image to visualize"))
visualize(id)