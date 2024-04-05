
# Utilizaremos MNIST
# https://github.com/Tensor4Dummies/5_img_mnist
# Tensorflow (Keras) instalación de Tensorflow pip install tensorflow
#
#
#
# Importamos las librerías necesarias.
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
#Cargamos el conjunto de datos MNIST.
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#Normalización de los datos de las imágenes para que sus valores estén en el rango de 0 a 1. 
train_images = train_images / 255.0
test_images = test_images / 255.0

#Convertimos las etiquetas a one-hot enconding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#
# Ejemplo
# train_labels = {0,2,1,2,0}
# despues de to_categorical nos devolvera el siguiente vector.
#   [[1,0,0],
#    [0,0,1],
#    [0,1,0],
#    [0,0,1],
#    [1,0,0]]
# 
# Construimos una red neuronal simple.


model = Sequential([
    Flatten(input_shape=(28, 28)), # Transforma la matriz de 28x28 en un vector de 784.
    Dense(128, activation='relu'), # Una capa densa con 128 neuronas y función de activación ReLu.
    Dense(10, activation='softmax') # Capa de salida con 10 neuronas (una para cada digito) y activación softmax.
])
#
# Compilación del Modelo.
# #
model.compile(optimizer = 'adam',
    loss ='categorical_crossentropy'
    metric = ['accuracy'])
#
# 
# #