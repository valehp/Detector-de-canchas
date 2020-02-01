import math
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import numpy as np
import statistics as stats
import gdal
from preprocess import *
import os
import cv2
import tensorflow as tf
from finetune_alexnet import alexnet


def cortar_imagen(file_tif, region_name, save_file, inicio, tipo='predict', canchas, img_normal=True, img_flip=False):
	print("Save file: ", save_file)
	coords = []
	dif = 65
	#for puntos in canchas:
	for i in canchas.index:
		centro = canchas[i]
		minx, miny = centro.x-dif , centro.y-dif
		maxx, maxy = centro.x+dif , centro.y+dif
		coords.append( ( maxx, miny, minx, maxy, i) )

	ids = [i for i in range(inicio, len(coords)+inicio, 1)]
	id_name = {}

	print("iniciando raster")
	raster = gdal.Open(file_tif)
	print("iniciando prep")
	prep = preprocessing(raster, raster, region_name, "mask", None)

	for i in range(len(coords)):
		coord = coords[i]
		coord1 = coord[0], coord[1]
		coord2 = coord[2], coord[3]
		prep.set_options(str(ids[i]), img_normal, img_flip, save_file)
		prep.crop_rasters(coord1, coord2, 256, tipo)
		id_name["cancha_{}_{}.jpg".format(region_name, str(ids[i]))] = coord[4]
		#id_name.append( ( coord[4], "cancha_{}_{}_00.jpg".format(region_name, str(ids[i])) ) )
	return id_name


def usar_red(txt, num_modelo):
	print("archivo para guardar: ", txt)
    #mean of imagenet dataset in BGR
	imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

	current_dir = os.getcwd()
	image_dir = os.path.join(current_dir, 'Predict')
	checkpoint_path = "./finetune_alexnet/tmp/"
	
	#get list of all images
	img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

	#load all images
	imgs = []
	for f in img_files:
		imgs.append(cv2.imread(f))

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
			if class_name == class_names[0]: canchas.append(img_files[i])
	archivo.close()
	return canchas