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


def distancia(p1, p2):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    d = math.sqrt( dx**2 + dy**2 )
    return d

def centro2(puntos):
   # print("haciendo poly")
    poly = Polygon([[p[0], p[1]] for p in puntos])
    return poly
   
def funcAnchos(cam, x, y, start, finish):
	lengthComp = 3 # cuantas lineas compara
	lengthAlg = 20 # largo de puntos a revisar
	sensProp = 1.4 # cuanto tiene que cambiar el ancho (proporcion)

	canchas = []
	newStart = start - lengthAlg
	if newStart < 0:
		newStart = 0
	newFinish = finish + lengthAlg
	if newFinish >= len(x):
		newFinish = len(x)-1

	difs = [0]*lengthComp
	for i in range(newStart, newFinish):
		difx = x[i+1]-x[i]
		dify = y[i+1]-y[i]

		if difx == 0:
			m = float('inf')
		else:
			m = dify/difx
		if m == 0:
			perpm = float('inf')
		else:
			perpm = -1/m
		px = (x[i+1]+x[i])/2
		py = (y[i+1]+y[i])/2
		a = py - perpm*px
		if perpm < 1 and perpm > -1:
			finalx1 = px - 50
			finalx2 = px + 50
			finaly1 = perpm*finalx1 + a
			finaly2 = perpm*finalx2 + a
		else:
			finaly1 = py - 50
			finaly2 = py + 50
			if perpm == float('inf'):
				finalx1 = px;
				finalx2 = px;
			else:
				finalx1 = (finaly1 - a)/perpm
				finalx2 = (finaly2 - a)/perpm
		line = LineString([(finalx1, finaly1), (finalx2, finaly2)])
		if not line.is_valid:
			continue

		inter = cam.intersection(line)
		interSelect = 0
		pt = Point(px, py)
		maxLen = 0
		try:
			for j in inter:
				if pt.distance(j) < 0.1 and j.length > maxLen:
					interSelect = j
					maxLen = j.length
		except TypeError:
			interSelect = inter
		if interSelect == 0:
			continue

		if difs[i%lengthComp] != 0:
			widthProp = interSelect.length / difs[i%lengthComp]
			if widthProp > sensProp:
				canchas.append(interSelect.centroid)

		for comps in range(lengthComp):
			if comps == i%lengthComp:
				difs[comps] = interSelect.length
			elif difs[comps] != 0 and interSelect.length / difs[comps] < sensProp:
				difs[comps] = 0
	#print("canchitas: ", canchas, type(canchas))
	return canchas
    
def separar_canchas(canchas):
    sensProp = 1.4
    lengthComp = 3
    separateTol = 25
    sepCanchas = []
    while canchas:
    	auxX = [canchas[0].x]
    	auxY = [canchas[0].y]
    	minX = canchas[0]
    	maxX = canchas[0]
    	minY = canchas[0]
    	maxY = canchas[0]
    	canchas.pop(0)
    	selectedPoints = []
    	for j in range(len(canchas)):
    		if minX.distance(canchas[j]) < separateTol or maxX.distance(canchas[j]) < separateTol or minY.distance(canchas[j]) < separateTol or minY.distance(canchas[j]) < separateTol:
    			if minX.x > canchas[j].x:
    				minX = canchas[j]
    			if maxX.x < canchas[j].x:
    				maxX = canchas[j]
    			if minY.y > canchas[j].y:
    				minY = canchas[j]
    			if maxY.y < canchas[j].y:
    				maxY = canchas[j]
    			auxX.append(canchas[j].x)
    			auxY.append(canchas[j].y)
    			selectedPoints.append(j)
    	meanX = stats.mean(list(auxX))
    	meanY = stats.mean(list(auxY))
    	sepCanchas.append(Point(meanX, meanY))
    	popCount = 0
    	for j in selectedPoints:
    		canchas.pop(j - popCount)
    		popCount += 1
    return sepCanchas
    
def postProcFunc(caminos, centros, separateTol=25):
	canchList = []
	for cam in caminos:
		canchList.append(cam)
	MPcanch = MultiPolygon(canchList)

	trueCentros = []
	while centros:
		auxX = [centros[0].x]
		auxY = [centros[0].y]
		minX = centros[0]
		maxX = centros[0]
		minY = centros[0]
		maxY = centros[0]
		centros.pop(0)
		selectedPoints = []
		for j in range(len(centros)):
			if minX.distance(centros[j]) < separateTol or maxX.distance(centros[j]) < separateTol or minY.distance(centros[j]) < separateTol or minY.distance(centros[j]) < separateTol:
				lin1 = LineString([centros[j], minX])
				lin2 = LineString([centros[j], maxX])
				lin3 = LineString([centros[j], minY])
				lin4 = LineString([centros[j], maxY])
				if not MPcanch.contains(lin1) or not MPcanch.contains(lin2) or not MPcanch.contains(lin3) or not MPcanch.contains(lin4):
					break
				if minX.x > centros[j].x:
					minX = centros[j]
				if maxX.x < centros[j].x:
					maxX = centros[j]
				if minY.y > centros[j].y:
					minY = centros[j]
				if maxY.y < centros[j].y:
					maxY = centros[j]
				auxX.append(centros[j].x)
				auxY.append(centros[j].y)
				selectedPoints.append(j)
		meanX = stats.mean(list(auxX))
		meanY = stats.mean(list(auxY))
		trueCentros.append(Point(meanX, meanY))
		popCount = 0
		for j in selectedPoints:
			centros.pop(j - popCount)
			popCount += 1

	return trueCentros


def centros_points(caminos, lengthComp=3, sensProp=1.4):
	pointsCanchas = []
	canchas = []
	for cam in caminos:
		difs = [0]*lengthComp

		# Concatenamos tanto los puntos interiores como exteriores.
		x, y = cam.exterior.coords.xy
		camInt = list(cam.interiors)
		if camInt:
			for it in camInt:
				xint, yint = it.coords.xy
				x.extend(xint)
				y.extend(yint)

		for i in range(len(x)-1):
			difx = x[i+1]-x[i]
			dify = y[i+1]-y[i]

			# Todo esto es para crear una linea perpendicular a la line analizada.
			if difx == 0:
				m = float('inf')
			else:
				m = dify/difx
			if m == 0:
				perpm = float('inf')
			else:
				perpm = -1/m
			px = (x[i+1]+x[i])/2
			py = (y[i+1]+y[i])/2
			a = py - perpm*px
			if perpm < 1 and perpm > -1:
				finalx1 = px - 50
				finalx2 = px + 50
				finaly1 = perpm*finalx1 + a
				finaly2 = perpm*finalx2 + a
			else:
				finaly1 = py - 50
				finaly2 = py + 50
				if perpm == float('inf'):
					finalx1 = px;
					finalx2 = px;
				else:
					finalx1 = (finaly1 - a)/perpm
					finalx2 = (finaly2 - a)/perpm
			line = LineString([(finalx1, finaly1), (finalx2, finaly2)])
			if not line.is_valid:
				continue

			# Se detectan las intersecciones, si hay mas de una, se usa la que este mas cerca del punto en analisis,
			# si no hay interseccion cerca del punto, se imprime QUEEEEEEEEEEE.
			inter = cam.intersection(line)
			interSelect = 0
			pt = Point(px, py)
			maxLen = 0
			try:
				for j in inter:
					if pt.distance(j) < 0.1 and j.length > maxLen:
						interSelect = j
						maxLen = j.length
			except TypeError:
				interSelect = inter
			if interSelect == 0:
				continue

			# Se almacenan anchos anteriores en difs, los cuales se vuelven 0 si algun ancho siguiente no cumple con
			# la sensibilidad establecida, si se cumplen por las veces establecidas el aumento de ancho, se añade como
			# cancha.
			if difs[i%lengthComp] != 0:
				widthProp = interSelect.length / difs[i%lengthComp]
				if widthProp > sensProp:
					canchas.append(interSelect.centroid)
					currentPoints = []
					for back in range(lengthComp):
						currentPoints.append((x[i+1-back], y[i+1-back]))
					pointsCanchas.append(currentPoints)
			for comps in range(lengthComp):
				if comps == i%lengthComp:
					difs[comps] = interSelect.length
				elif difs[comps] != 0 and interSelect.length / difs[comps] < sensProp:
					difs[comps] = 0

	return (canchas, pointsCanchas)


def centros_por_distancia(caminos, distancia_minima = 0.1, puntos_minimos = 9):
    #print("caminos: ", caminos)
    distancias = []
    menores = []
    m = []
    puntos = []
    contador = 0
    
    centritos = []
    poligonitos = []
    
    inicio = -1
    fin = -1
    
    for camino in caminos:
        
        m = []
        menores = []
        added= False # True si el punto directamente anterior se añadio a m
        anterior = 0.0 # distancia anterior a comparar
        x = [i[0] for i in camino]
        y = [i[1] for i in camino]
        #pointList = [Point(i[0], i[1]) for i in camino]
        pol = Polygon(camino)
        for j in range(1, len(x), 1):
            p1 = (x[j-1], y[j-1])
            p2 = (x[j], y[j])
            d = distancia(p1, p2)
            distancias.append( (p1, p2, d) )
            if j == 1:
                anterior = d
                added = True
            else:
                comp = abs(anterior - d)
                if comp <= distancia_minima:
                    if p1 not in m:
                        if added:
                            if not menores: inicio = j-1
                            else: fin = j-1
                            menores.append(p1)
                        else:
                            if len(menores) > puntos_minimos: m.append((menores, inicio, fin))
                            menores = []
                    if p2 not in m:
                        if not menores: inicio = j
                        else: fin = j
                        menores.append(p2)
                        added = True
                else: added = False
            anterior = d
            
        for lista, inicio, fin in m:
            if len(lista) <= 2: continue
            poly = centro2(lista)
            pto = poly.centroid 
            if pto != False:
                if pto.within(pol):
                    centritos.append(pto)
                    contador += 1
    return centritos
                    
                    
def cortar_imagen(file_tif, region_name, save_file, inicio, tipo, canchas, img_normal, img_flip, puntos):
	print("Save file: ", save_file)
	coords = []
	dif = 65
	#for puntos in canchas:
	for i in canchas.index:
		if puntos: centro = canchas[i]
		else: 
			puntos = canchas[i]
			centro = puntos.centroid
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


def calcular_errores(lista_centros):
    tp_files = ['../shapeFiles/20059/canchas.shp', '../shapeFiles/20059/guatas.shp']
    fp_files = ['../shapeFiles/20059/bifurcaciones.shp', '../shapeFiles/20059/curvas.shp']
    
    tp_polygons = [ gpd.GeoSeries(gpd.read_file(path)["geometry"]) for path in tp_files]
    fp_polygons = [ gpd.GeoSeries(gpd.read_file(path)["geometry"]) for path in fp_files]
    
    tp_polygons = [ poly for list in tp_polygons for poly in list ]
    fp_polygons = [ poly for list in fp_polygons for poly in list ]
    
    #centros = gpd.GeoSeries(gpd.read_file('../vale/shapefiles/59_centros_merge.shp')["geometry"])
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    num_tp = len(tp_polygons)
    num_fp = len(fp_polygons)
    print(num_tp, num_fp, len(lista_centros))
    
    for centro in lista_centros:
        found = False
        for poly in tp_polygons:
            if centro.within(poly):
                tp += 1
                tp_polygons.remove(poly)
                found = True
                break
        if not found:
            fp += 1
    exp_tp = tp / num_tp
    if tp != 0: exp_fp = fp / tp
    else: exp_fp = -1
    return (exp_tp, exp_fp, tp, fp)