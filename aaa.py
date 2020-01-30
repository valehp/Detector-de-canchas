import geopandas as gpd
import time
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import numpy as np
import statistics as stats
from funciones import *
import random

"""
# Código para crear los pedazos de imagenes con figuras en shape files
raster = "C:/Users/ASUS/Desktop/ARAUCO/MOSAICO/20059_mosaico.tif"
shp = 'C:/Users/ASUS/Desktop/ARAUCO/arauco/shapeFiles/20059/curvas.shp'
data = gpd.read_file(shp)
canchas = data["geometry"]
canchas = gpd.GeoSeries(canchas)
#print(canchas)
cortar_imagen(raster, '20059', 'train/nocanchas', 230, 'train', canchas, True, True)
# curvas inicio -> 230
# guatas inicio -> 224
"""
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

imgs_files = []
f = open("train.txt", "r")
for linea in f:
    linea = linea.split("\n")
    imgs_files.append(linea)
f.close()

print("archivo para guardar: ", txt)
#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
checkpoint_path = "./finetune_alexnet/tmp/"

#load all images
imgs = []
tags = []
for f, tag in img_files:
    imgs.append(cv2.imread(f))
    tags.append(tag)

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 256, 256, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = alexnet.AlexNet(x, keep_prob, 2, [], "./finetune_alexnet/bvlc_alexnet.npy")

#define activation of last layer as score
score = model.fc8

saver = tf.train.Saver()

#create op to calculate softmax 
softmax = tf.nn.softmax(score)
class_names = ['Cancha', 'No cancha']
predict = []
archivo = open(txt, "w")

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
        archivo.write(img_files[i] + " : " + class_name + " " + str(np.argmax(probs)) + "\n")
        #print("imagen", i+1, " - clase: ", class_name)
        if np.argmax(probs) == tags[i]: buenos += 1
        else: malos += 1
        if class_name == class_names[0]: canchas.append(img_files[i])
archivo.close()

centros = gpd.GeoDataFrame(columns=["centro"],geometry = "centro")
centros['centro'] = canchas
out = "C:/Users/ASUS/Desktop/ARAUCO/red2/finetune_alexnet/{}_canchas_test.shp".format(region_name) # windows
centros.to_file(out)
