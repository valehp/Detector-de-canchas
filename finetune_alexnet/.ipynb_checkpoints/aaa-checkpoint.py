import os
import random
import shutil
import alexnet
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import statistics as stats
import gdal
import os
import cv2
import tensorflow as tf
import geopandas as gpd

"""
# código para escoger aleatoriamente num_val elementos para validacion y escribirlos en un txt para la red
train_c = []
train_nc = []
valid_c = []
valid_nc = []

train = []
valid = []
test = []

files_c = 'C:/Users/ASUS/Desktop/ARAUCO/red/train/canchas/'
files_nc = 'C:/Users/ASUS/Desktop/ARAUCO/red/train/nocanchas/'

for archivo in os.listdir(files_c): train_c.append(archivo)
for archivo in os.listdir(files_nc): train_nc.append(archivo)

num_val = 400 # cantidad de fotos a validar
for i in range(num_val//2): 
    x = random.randint(0, len(train_c)-1)
    y = random.randint(0, len(train_nc)-1)
    valid_c.append(train_c[x])
    valid_nc.append(train_nc[y])
    train_c.pop(x)
    train_nc.pop(y)

for archivo in train_c: train.append(files_c + archivo + " 0\n")
for archivo in train_nc: train.append(files_nc + archivo + " 1\n")

for archivo in valid_c: valid.append(files_c + archivo + " 0\n")
for archivo in valid_nc: valid.append(files_nc + archivo + " 1\n")

num_test = 100
for i in range(num_test): 
    x = random.randint(0, len(train)-1)
    test.append(train[x])
    train.pop(x)

random.shuffle(train)
random.shuffle(valid)

f = open("train.txt", "w")
for archivo in train: f.write(archivo)
f.close()

f = open("valid.txt", "w")
for archivo in valid: f.write(archivo)
f.close()

f = open("test.txt", "w")
for archivo in test: f.write(archivo)
f.close()
"""


# Código para utilizar el test

num_modelo = 100
region_name = '20059'
img_files = []
f = open("test.txt", "r")
for linea in f:
    linea = linea.split("\n")
    img_files.append(linea[0])
f.close()

#print("img_files: ", img_files)

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
checkpoint_path = "./tmp/"

#load all images
imgs = []
tags = []
for linea in img_files:
    f, tag = linea.split()
    imgs.append(cv2.imread(f))
    tags.append(tag)

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 256, 256, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = alexnet.AlexNet(x, keep_prob, 2, [])

#define activation of last layer as score
score = model.fc8

saver = tf.train.Saver()

#create op to calculate softmax 
softmax = tf.nn.softmax(score)
class_names = ['Cancha', 'No cancha']
predict = []
archivo = open("out.txt", "w")

canchas = []
buenos = 0
malos = 0
with tf.Session() as sess:
    print("sesion lista")
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    # Load the pretrained weights into the model
    checkpoint_name = os.path.join(checkpoint_path, 'final_model_'+str(num_modelo)+'.ckpt')
    saver.restore(sess, checkpoint_name)

    # Create figure handle
    fig2 = plt.figure(figsize=(15,6))
    
    # Loop over all images
    for i, image in enumerate(imgs):
        
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (256,256))
        
        # Subtract the ImageNet mean
        img -= imagenet_mean
        
        # Reshape as needed to feed into model
        img = img.reshape((1,256,256,3))
        
        # Run the session and calculate the class probability
        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        
        # Get the class name of the class with the highest probability
        class_name = class_names[np.argmax(probs)]
        archivo.write(img_files[i] + " : " + class_name + " " + str(np.argmax(probs)) + "\tReal: "+tags[i] +"\n")
        #print("imagen", i+1, " - clase: ", class_name)
        if class_name == class_names[int(tags[i])]: buenos += 1
        else: malos += 1
        if class_name == class_names[0]: canchas.append(img_files[i])

archivo.write("Buenos: {} | Malos: {} | Totales: {}".format(buenos, malos, len(img_files)))
archivo.close()
