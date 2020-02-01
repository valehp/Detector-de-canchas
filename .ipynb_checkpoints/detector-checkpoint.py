# -*- coding: utf-8 -*-
import geopandas as gpd
import time
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import numpy as np
import statistics as stats
from DCPA import *
from funciones import *
#from estadisticas2 import *


# Comentar si se usa CPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.keras.backend.set_session(tf.Session(config=config))
# comentar hasta aquí

region_name = "20015"

fp = 'C:/Users/ASUS/Desktop/ARAUCO/arauco/shapeFiles/20015/20015.shp'

data = gpd.read_file(fp)
caminos = data[data.TIPOUSO == "FCAM"]
caminos = caminos["geometry"]
caminos = gpd.GeoSeries(caminos)

start = time.time()

lengthComp = 0

parametros = {
        "1.1": 3, 
        #"1.52": 9, 
        #"1.8": 15
        }

canchasList = []
caminitos = []
sensProp = 1.1
lengthComp = 3
data = gpd.read_file(fp)
caminos = data[data.TIPOUSO == "FCAM"]
caminos = caminos["geometry"]
caminos = gpd.GeoSeries(caminos)
saltados = gpd.GeoDataFrame(columns=['puntoMarcado'], geometry='puntoMarcado')
print("length comp: ", lengthComp)

start = time.time()
separateTol = 25

saltadosCount = 0
caminoCount = 1

anchosCanchas = []

canchas = []
canchas, _ = detCanchasAncho(caminos, lengthComp, sensProp)
sepCanchas = postJuntarPuntos(canchas)

centroides = gpd.GeoDataFrame(columns=["centro"],geometry = "centro")
centroides['centro'] = sepCanchas
canchas_posibles = centroides["centro"]

tif = "C:/Users/ASUS/Desktop/ARAUCO/MOSAICO/20015_mosaico.tif"
imgs = cortar_imagen(tif, '20015', '/Predict', 1, 'predict', canchas_posibles, True, False, True)

canchas = []
ultimos_centros = usar_red("canchas_20015.txt", 50)
for u in ultimos_centros:
	u = u.split("\\")
	i = imgs[u[len(u)-1]]
	canchas.append(canchas_posibles[i])

if canchas:
	centros = gpd.GeoDataFrame(columns=["centro"],geometry = "centro")
	centros["centro"] = canchas
	out = "C:/Users/ASUS/Desktop/ARAUCO/red2/{}_canchas_1.shp".format(region_name) # windows
	centros.to_file(out)
else: print("ERROR: no hay canchas :(")

end = time.time()

print("Tiempo a canchas 1: ", end - start)


canchas2 = separar_canchas(canchas)
if canchas2:
	centros = gpd.GeoDataFrame(columns=["centro"],geometry = "centro")
	centros["centro"] = canchas2
	out = "C:/Users/ASUS/Desktop/ARAUCO/red2/{}_canchas_2.shp".format(region_name) # windows
	centros.to_file(out)
else: print("ERROR: no hay canchas :(")


end = time.time()

print("Tiempo: ", end - start)







