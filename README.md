# Visualization-Tech-for-undersanding-CNN-layers

tSNE basic usage demo：

Visualization01_tSNEandPCA.py

Visualization02_tSNE.py

Visualization01_tSNE.py


Using tSNE to visualize the last hidden-layer of CNN model @MNIST，CIFAR10 dataset:

CNN_Visualization_Using_tSNE_MNIST.py

CNN_Visualization_Using_tSNE_CIFAR10.py



Credit:

https://becominghuman.ai/visualizing-representations-bd9b62447e38, cite:

In order to obtain the hidden-layer representation, we will first truncate the model at the LSTM layer. Thereafter, we will load the model with the weights that the model has learnt. A better way to do this is create a new model with the same steps (until the layer you want) and load the weights from the model. Layers in Keras models are iterable. 
